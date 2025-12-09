#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Benchmark NUMA-aware memcpy-style bandwidth in several scenarios:
 *  - local:    source and destination on the same NUMA node as the worker
 *  - remote-r: worker on node B copies from buffers on node A into buffers on B (remote read)
 *  - remote-w: worker on node B copies from buffers on B into buffers on A (remote write)
 *  - bidir:    half the threads on node A copy to node B while the other half copy to node A
 *
 * The goal is to expose core<->RAM and cross-socket traffic limits that hurt
 * multi-instance OpenBLAS runs under USE_TLS=1.
 *
 * Environment knobs (all optional):
 *   BENCH_SIZES   comma list like "64K,8M,64M" (defaults span caches->RAM)
 *   BENCH_THREADS thread counts, e.g. "1,2,4,8" (defaults: 1,2,4)
 *   BENCH_RUNS    max repetitions before giving up (default 30)
 *   BENCH_CV      target coefficient of variation (default 0.03)
 *   BENCH_CASES   subset of cases to run: "local,remote-r,remote-w,bidir"
 *   BENCH_NODEA/B override node ids (defaults: 0 and 1 when available)
 */

typedef enum {
    CASE_LOCAL,
    CASE_REMOTE_READ,
    CASE_REMOTE_WRITE,
    CASE_BIDIR,
    CASE_MAX
} bench_case_t;

static const char *case_name[CASE_MAX] = { "local", "remote-r", "remote-w", "bidir" };

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double mean(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return s / (double)n;
}

static double stdev(const double *v, int n, double m) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = v[i] - m;
        s += d * d;
    }
    return sqrt(s / (double)n);
}

static int max_runs(void) {
    const char *s = getenv("BENCH_RUNS");
    if (!s || !*s) return 30;
    int v = atoi(s);
    if (v < 3) v = 3;
    return v;
}

static double target_cv(void) {
    const char *s = getenv("BENCH_CV");
    if (!s || !*s) return 0.03; /* 3% default */
    double v = atof(s);
    if (v < 0.0005) v = 0.0005;
    return v;
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

static size_t read_cache_sysfs(int index) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[64];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);
    char *end = NULL;
    return parse_size_token(buf, &end);
}

static int build_size_list(size_t *sizes, int max) {
    const char *env = getenv("BENCH_SIZES");
    int n = 0;
    if (env && *env) {
        n = parse_size_list(env, sizes, max);
        if (n > 0) return n;
    }
    size_t l2 = read_cache_sysfs(2);
    size_t l3 = read_cache_sysfs(3);
    size_t defaults[] = {
        8 * 1024,                  /* L1-ish */
        64 * 1024,
        l2 ? l2 / 2 : 256 * 1024,
        l2 ? l2     : 512 * 1024,
        l3 ? l3 / 2 : 4 * 1024 * 1024,
        l3 ? l3     : 16 * 1024 * 1024,
        l3 ? l3 * 2 : 64 * 1024 * 1024
    };
    for (size_t i = 0; i < sizeof(defaults)/sizeof(defaults[0]) && n < max; i++) {
        if (n > 0 && sizes[n-1] == defaults[i]) continue;
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

static int build_thread_list(int *threads, int max) {
    const char *env = getenv("BENCH_THREADS");
    int n = 0;
    if (env && *env) n = parse_int_list(env, threads, max);
    if (n > 0) return n;
    int tmax = omp_get_max_threads();
    int defaults[] = {1, 2, 4, 8};
    for (size_t i = 0; i < sizeof(defaults)/sizeof(defaults[0]) && n < max; i++) {
        if (defaults[i] > tmax) continue;
        threads[n++] = defaults[i];
    }
    return n;
}

static bool case_enabled(bench_case_t c) {
    const char *env = getenv("BENCH_CASES");
    if (!env || !*env) return true;
    char *tmp = strdup(env);
    bool ok = false;
    for (char *tok = strtok(tmp, ","); tok; tok = strtok(NULL, ",")) {
        for (int i = 0; i < CASE_MAX; i++) {
            if (strcmp(tok, case_name[i]) == 0) { ok = (c == i) ? true : ok; }
        }
    }
    free(tmp);
    return ok;
}

static void pin_to_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        perror("sched_setaffinity");
    }
}

static int pick_cpu_from_mask(const struct bitmask *bm, int idx) {
    int count = 0;
    for (int cpu = bm->size - 1; cpu >= 0; cpu--) {
        if (numa_bitmask_isbitset(bm, cpu)) {
            if (count == idx) return cpu;
            count++;
        }
    }
    return -1;
}

static void touch_buffer(char *p, size_t len) {
    size_t step = 4096;
    for (size_t i = 0; i < len; i += step) p[i] = (char)(i & 0xFF);
    if (len) p[len - 1] = 1;
}

static double run_case(bench_case_t which, size_t bytes, int threads, int node_a, int node_b) {
    double times[64];
    int maxr = max_runs();
    double cv_target = target_cv();
    if (threads > 64) threads = 64; /* guard */

    /* Allocate per-thread buffers */
    char **src = calloc((size_t)threads, sizeof(char *));
    char **dst = calloc((size_t)threads, sizeof(char *));
    if (!src || !dst) { fprintf(stderr, "alloc failed\n"); exit(1); }

    for (int t = 0; t < threads; t++) {
        int src_node = node_a, dst_node = node_a;
        switch (which) {
            case CASE_LOCAL:      src_node = dst_node = node_a; break;
            case CASE_REMOTE_READ: src_node = node_a; dst_node = node_b; break;
            case CASE_REMOTE_WRITE: src_node = node_b; dst_node = node_a; break;
            case CASE_BIDIR:
                /* even threads copy A->B, odd threads copy B->A */
                src_node = (t % 2 == 0) ? node_a : node_b;
                dst_node = (t % 2 == 0) ? node_b : node_a;
                break;
            default: break;
        }
        src[t] = numa_alloc_onnode(bytes, src_node);
        dst[t] = numa_alloc_onnode(bytes, dst_node);
        if (!src[t] || !dst[t]) {
            fprintf(stderr, "numa_alloc_onnode failed (node %d/%d)\n", src_node, dst_node);
            exit(1);
        }
        touch_buffer(src[t], bytes);
        touch_buffer(dst[t], bytes);
    }

    /* Choose CPUs for pinning */
    struct bitmask *mask_a = numa_allocate_cpumask();
    struct bitmask *mask_b = numa_allocate_cpumask();
    numa_node_to_cpus(node_a, mask_a);
    numa_node_to_cpus(node_b, mask_b);

    int r;
    for (r = 0; r < maxr && r < (int)(sizeof(times)/sizeof(times[0])); r++) {
        double t0 = now_sec();
#pragma omp parallel num_threads(threads)
        {
            int tid = omp_get_thread_num();
            bool use_a = true;
            switch (which) {
                case CASE_LOCAL: use_a = true; break;
                case CASE_REMOTE_READ: use_a = false; break; /* workers on node_b */
                case CASE_REMOTE_WRITE: use_a = false; break; /* workers on node_b */
                case CASE_BIDIR: use_a = (tid % 2 == 0); break;
                default: break;
            }
            int cpu = pick_cpu_from_mask(use_a ? mask_a : mask_b, tid % numa_num_task_cpus());
            if (cpu >= 0) pin_to_cpu(cpu);

            memcpy(dst[tid], src[tid], bytes);
        }
        double t1 = now_sec();
        times[r] = t1 - t0;
        if (r >= 2) {
            double m = mean(times, r + 1);
            double sd = stdev(times, r + 1, m);
            if (sd / m <= cv_target) { r++; break; }
        }
    }
    int runs = r;
    if (runs == 0) runs = 1;
    double m = mean(times, runs);
    double sd = stdev(times, runs, m);

    double total_bytes = (double)bytes * (double)threads;
    double gbps = total_bytes / m / 1e9;
    double std_pct = (m > 0.0) ? (sd / m * 100.0) : 0.0;

    printf("%s,%zu,%d,%.3f,%.3f,%d,%.2f,%d\n",
           case_name[which], bytes, threads, gbps, gbps / threads,
           (which == CASE_BIDIR) ? 2 : 1, std_pct, runs);

    for (int t = 0; t < threads; t++) {
        numa_free(src[t], bytes);
        numa_free(dst[t], bytes);
    }
    free(src);
    free(dst);
    numa_free_cpumask(mask_a);
    numa_free_cpumask(mask_b);
    return gbps;
}

int main(void) {
    if (numa_available() < 0) {
        fprintf(stderr, "numa_available() failed; this benchmark needs libnuma and NUMA hardware\n");
        return 1;
    }
    int max_node = numa_max_node();
    if (max_node < 0) {
        fprintf(stderr, "no NUMA nodes detected\n");
        return 1;
    }

    int node_a = 0;
    int node_b = (max_node >= 1) ? 1 : 0;
    const char *env_a = getenv("BENCH_NODEA");
    const char *env_b = getenv("BENCH_NODEB");
    if (env_a && *env_a) node_a = atoi(env_a);
    if (env_b && *env_b) node_b = atoi(env_b);
    if (node_a > max_node) node_a = max_node;
    if (node_b > max_node) node_b = max_node;

    size_t sizes[16];
    int nsizes = build_size_list(sizes, (int)(sizeof(sizes)/sizeof(sizes[0])));

    int threads_list[12];
    int nthreads = build_thread_list(threads_list, (int)(sizeof(threads_list)/sizeof(threads_list[0])));

    printf("case,size_bytes,threads,agg_gbps,per_thread_gbps,directions,std_pct,runs\n");
    for (int s = 0; s < nsizes; s++) {
        size_t bytes = sizes[s];
        for (int t = 0; t < nthreads; t++) {
            int nth = threads_list[t];
            for (int c = 0; c < CASE_MAX; c++) {
                bench_case_t bc = (bench_case_t)c;
                if (!case_enabled(bc)) continue;
                if (bc == CASE_BIDIR && (nth % 2)) continue; /* need pairs */
                /* Skip remote cases if only one node exists */
                if ((bc == CASE_REMOTE_READ || bc == CASE_REMOTE_WRITE || bc == CASE_BIDIR) &&
                    node_a == node_b) continue;
                run_case(bc, bytes, nth, node_a, node_b);
            }
        }
    }
    return 0;
}
