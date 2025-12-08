#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cblas.h>
#include <errno.h>

/* OpenBLAS-specific */
extern void openblas_set_num_threads(int num_threads);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double mean(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return s / n;
}

static double stdev(const double *v, int n, double m) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = v[i] - m;
        s += d * d;
    }
    return sqrt(s / n);
}

static double target_cv(void) {
    const char *s = getenv("BENCH_CV");
    if (!s || !*s) return 0.03; /* 3% default */
    double v = atof(s);
    if (v < 0.0005) v = 0.0005;
    return v;
}

static int max_runs(void) {
    const char *s = getenv("BENCH_MAX_RUNS");
    if (!s || !*s) return 50;  /* run longer by default to reduce noise */
    int v = atoi(s);
    if (v < 3) v = 3;
    return v;
}

static void *alloc_aligned(size_t bytes, size_t align) {
    void *p = NULL;
    if (posix_memalign(&p, align, bytes)) return NULL;
    return p;
}

static size_t parse_size_token(const char *p, char **end) {
    errno = 0;
    unsigned long long v = strtoull(p, end, 10);
    if (errno || p == *end) return 0;
    switch (**end) {
        case 'g': case 'G': v *= 1024ULL;
        case 'm': case 'M': v *= 1024ULL;
        case 'k': case 'K': v *= 1024ULL; (*end)++; break;
        default: break;
    }
    return (size_t)v;
}

static int parse_size_list(const char *env, size_t *out, int max) {
    int n = 0;
    const char *p = env;
    while (*p && n < max) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        size_t v = parse_size_token(p, &end);
        if (v > 0) out[n++] = v;
        p = end ? end : p + 1;
        while (*p && *p != ',') p++;
    }
    return n;
}

static int build_size_list(size_t *sizes, int max) {
    const char *env = getenv("BENCH_SIZES");
    int n = 0;
    if (env && *env) {
        n = parse_size_list(env, sizes, max);
        if (n > 0) return n;
    }
    size_t defaults[] = { 32 * 1024UL, 256 * 1024UL, 8 * 1024 * 1024UL, 64 * 1024 * 1024UL };
    for (size_t i = 0; i < sizeof(defaults)/sizeof(defaults[0]) && n < max; i++) {
        sizes[n++] = defaults[i];
    }
    return n;
}

static int parse_int_list(const char *env, int *out, int max) {
    int n = 0;
    const char *p = env;
    while (*p && n < max) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end && v > 0) out[n++] = (int)v;
        p = end ? end : p + 1;
        while (*p && *p != ',') p++;
    }
    return n;
}

static int build_instances_list(int *vals, int max) {
    const char *env = getenv("BENCH_INSTANCES");
    int n = 0;
    if (env && *env) n = parse_int_list(env, vals, max);
    if (n > 0) return n;

    int defaults[] = {1, 2, 4, 8, 12, 16};
    int tmax = omp_get_max_threads();
    for (size_t i = 0; i < sizeof(defaults)/sizeof(defaults[0]) && n < max; i++) {
        if (defaults[i] > tmax * 2) break; /* avoid wild oversubscription */
        vals[n++] = defaults[i];
    }
    return n;
}

static int build_inner_threads(int *vals, int max) {
    const char *env = getenv("BENCH_INNER_THREADS");
    int n = 0;
    if (env && *env) n = parse_int_list(env, vals, max);
    if (n > 0) return n;

    int defaults[] = {1, 2, 4};
    int tmax = omp_get_max_threads();
    for (size_t i = 0; i < sizeof(defaults)/sizeof(defaults[0]) && n < max; i++) {
        if (defaults[i] > tmax) continue;
        vals[n++] = defaults[i];
    }
    if (n == 0) vals[n++] = 1;
    return n;
}

static void fill_pattern(float *p, size_t elems) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < elems; i++) p[i] = (float)(i % 97);
}

static double run_memcpy_case(float **dst, float **src, size_t bytes, int instances, int runs_limit, double cv_target, double *std_pct_out, int *runs_out) {
    double times[64];
    size_t elems = bytes / sizeof(float);
    for (int r = 0; r < runs_limit && r < 64; r++) {
        double t0 = now_sec();
        #pragma omp parallel for num_threads(instances) schedule(static)
        for (int i = 0; i < instances; i++) {
            memcpy(dst[i], src[i], elems * sizeof(float));
        }
        double t1 = now_sec();
        times[r] = t1 - t0;
        if (r >= 2) {
            double m = mean(times, r + 1);
            double sd = stdev(times, r + 1, m);
            if (sd / m <= cv_target) { r++; break; }
        }
    }
    int runs = 0;
    while (runs < runs_limit && runs < 64 && times[runs] > 0) runs++;
    double m = mean(times, runs);
    double sd = stdev(times, runs, m);
    *std_pct_out = (m > 0.0) ? (sd / m * 100.0) : 0.0;
    *runs_out = runs;
    double total_bytes = (double)bytes * (double)instances;
    return total_bytes / m / 1e9;
}

static double run_scopy_case(float **dst, float **src, size_t bytes, int instances, int inner_threads, int runs_limit, double cv_target, double *std_pct_out, int *runs_out) {
    double times[64];
    size_t elems = bytes / sizeof(float);
    openblas_set_num_threads(inner_threads);
    for (int r = 0; r < runs_limit && r < 64; r++) {
        double t0 = now_sec();
        #pragma omp parallel for num_threads(instances) schedule(static)
        for (int i = 0; i < instances; i++) {
            cblas_scopy((int)elems, src[i], 1, dst[i], 1);
        }
        double t1 = now_sec();
        times[r] = t1 - t0;
        if (r >= 2) {
            double m = mean(times, r + 1);
            double sd = stdev(times, r + 1, m);
            if (sd / m <= cv_target) { r++; break; }
        }
    }
    int runs = 0;
    while (runs < runs_limit && runs < 64 && times[runs] > 0) runs++;
    double m = mean(times, runs);
    double sd = stdev(times, runs, m);
    *std_pct_out = (m > 0.0) ? (sd / m * 100.0) : 0.0;
    *runs_out = runs;
    double total_bytes = (double)bytes * (double)instances;
    return total_bytes / m / 1e9;
}

int main(void) {
    size_t sizes[16];
    int instances_list[12];
    int inner_list[8];

    int nsizes = build_size_list(sizes, (int)(sizeof(sizes)/sizeof(sizes[0])));
    int ninst  = build_instances_list(instances_list, (int)(sizeof(instances_list)/sizeof(instances_list[0])));
    int ninner = build_inner_threads(inner_list, (int)(sizeof(inner_list)/sizeof(inner_list[0])));

    int max_inst = 0;
    for (int i = 0; i < ninst; i++) if (instances_list[i] > max_inst) max_inst = instances_list[i];

    /* Preallocate buffers for worst case to reduce allocator noise */
    float **src = calloc((size_t)max_inst, sizeof(float*));
    float **dst = calloc((size_t)max_inst, sizeof(float*));
    if (!src || !dst) {
        fprintf(stderr, "alloc pointer table failed\n");
        return 1;
    }

    double cv_target = target_cv();
    int runs_limit = max_runs();

    printf("test,size_bytes,instances,inner_threads,agg_gbps,per_instance_gbps,std_pct,runs\n");
    for (int s = 0; s < nsizes; s++) {
        size_t bytes = sizes[s];
        size_t elems = bytes / sizeof(float);
        for (int i = 0; i < max_inst; i++) {
            src[i] = alloc_aligned(bytes, 64);
            dst[i] = alloc_aligned(bytes, 64);
            if (!src[i] || !dst[i]) {
                fprintf(stderr, "alloc failed for size %zu bytes (instance %d)\n", bytes, i);
                return 1;
            }
            fill_pattern(src[i], elems);
            memset(dst[i], 0, bytes);
        }

        for (int k = 0; k < ninst; k++) {
            int inst = instances_list[k];
            double std_pct, bw;
            int runs;

            bw = run_memcpy_case(dst, src, bytes, inst, runs_limit, cv_target, &std_pct, &runs);
            printf("memcpy,%zu,%d,%d,%.3f,%.3f,%.2f,%d\n",
                   bytes, inst, 0, bw, bw / inst, std_pct, runs);

            for (int t = 0; t < ninner; t++) {
                int inner = inner_list[t];
                bw = run_scopy_case(dst, src, bytes, inst, inner, runs_limit, cv_target, &std_pct, &runs);
                printf("scopy,%zu,%d,%d,%.3f,%.3f,%.2f,%d\n",
                       bytes, inst, inner, bw, bw / inst, std_pct, runs);
            }
        }

        for (int i = 0; i < max_inst; i++) {
            free(src[i]);
            free(dst[i]);
            src[i] = dst[i] = NULL;
        }
    }

    free(src);
    free(dst);
    return 0;
}
