#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <omp.h>
#include <cblas.h>

static double target_cv(void) {
  const char *s = getenv("BENCH_CV");
  if (!s || !*s) return 0.03; /* 3% default */
  double v = atof(s);
  if (v <= 0.0001) v = 0.0001;
  return v;
}

static int max_runs(void) {
  const char *s = getenv("BENCH_MAX_RUNS");
  if (!s || !*s) return 50; /* default to more samples for tighter stats */
  int v = atoi(s);
  if (v < 3) v = 3;
  return v;
}

static double mean(const double *t, int n) {
  double s = 0.0;
  for (int i = 0; i < n; i++) s += t[i];
  return s / n;
}

static double stdev(const double *t, int n, double m) {
  double s = 0.0;
  for (int i = 0; i < n; i++) {
    double d = t[i] - m;
    s += d * d;
  }
  return sqrt(s / n);
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *alloc_aligned(size_t bytes, size_t align) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, align, bytes)) return NULL;
  return ptr;
}

static size_t parse_size_token(const char *p, char **end) {
  /* Parses tokens like 32K, 8M, 1G, or raw bytes. */
  errno = 0;
  unsigned long long v = strtoull(p, end, 10);
  if (errno || p == *end) return 0;
  switch (tolower(**end)) {
    case 'g': v *= 1024ULL;
    case 'm': v *= 1024ULL;
    case 'k': v *= 1024ULL; (*end)++; break;
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
  /* Try Linux sysfs cache info: index2=L2, index3=L3 */
  char path[128];
  snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char buf[64];
  if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
  fclose(f);

  char *end = NULL;
  size_t v = parse_size_token(buf, &end);
  return v;
}

static int build_size_list(size_t *sizes, int max) {
  const char *env = getenv("BENCH_SIZES");
  int n = 0;
  if (env && *env) {
    n = parse_size_list(env, sizes, max);
    if (n > 0) return n;
  }

  /* Detect cache sizes to ensure we exercise L1/L2/L3 and RAM paths. */
  size_t l2 = read_cache_sysfs(2);
  size_t l3 = read_cache_sysfs(3);

  size_t defaults[] = {
    8 * 1024,                 /* L1-sized */
    64 * 1024,                /* L1/L2 boundary */
    l2  ? l2 / 2 : 512 * 1024,
    l2  ? l2 : 1 * 1024 * 1024,
    l3  ? l3 / 2 : 8 * 1024 * 1024,
    l3  ? l3 : 32 * 1024 * 1024,
    l3  ? l3 * 2 : 128 * 1024 * 1024
  };

  for (size_t i = 0; i < sizeof(defaults)/sizeof(defaults[0]) && n < max; i++) {
    size_t v = defaults[i];
    /* Deduplicate adjacent values */
    if (n > 0 && sizes[n-1] == v) continue;
    sizes[n++] = v;
  }
  return n;
}

static int parse_thread_list(const char *env, int *out, int max) {
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
  if (env && *env) n = parse_thread_list(env, threads, max);
  if (n > 0) return n;

  int tmax = omp_get_max_threads();
  int defaults[3];
  defaults[0] = 1;
  defaults[1] = tmax >= 2 ? tmax / 2 : 1;
  defaults[2] = tmax;
  for (int i = 0; i < 3 && n < max; i++) {
    if (defaults[i] < 1) continue;
    if (n > 0 && threads[n-1] == defaults[i]) continue;
    threads[n++] = defaults[i];
  }
  return n;
}

static double checksum_bytes(const uint8_t *p, size_t n) {
  double acc = 0.0;
  for (size_t i = 0; i < n; i += 4096) acc += p[i];
  if (n) acc += p[n - 1];
  return acc;
}

static double bench_memcpy_mt(size_t bytes, int threads, double *std_gbps, int *runs_out) {
  uint8_t *src = alloc_aligned(bytes, 64);
  uint8_t *dst = alloc_aligned(bytes, 64);
  if (!src || !dst) return 0.0;

  /* Fill source to avoid lazy allocation penalties later. */
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < bytes; i++) src[i] = (uint8_t)(i * 13 + 7);

  double times[64];
  int maxr = max_runs();
  double cv_target = target_cv();
  int r;
  for (r = 0; r < maxr && r < (int)(sizeof(times)/sizeof(times[0])); r++) {
    #pragma omp barrier
    double t0 = now_sec();
    #pragma omp parallel num_threads(threads)
    {
      int tid = omp_get_thread_num();
      size_t chunk = bytes / threads;
      size_t start = chunk * tid;
      size_t len = (tid == threads - 1) ? (bytes - start) : chunk;
      memcpy(dst + start, src + start, len);
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
  double m = mean(times, runs);
  double sd = stdev(times, runs, m);
  double bw = (double)bytes / m / 1e9;
  *std_gbps = (sd > 0 ? (double)bytes / 1e9 / m * (sd / m) : 0.0); /* relative sd times bw */
  *runs_out = runs;
  free(src);
  free(dst);
  return bw;
}

static double bench_scopy_outer(size_t bytes, int threads, double *std_gbps, int *runs_out) {
  size_t elems = bytes / sizeof(float);
  float *src = alloc_aligned(elems * sizeof(float), 64);
  float *dst = alloc_aligned(elems * sizeof(float), 64);
  if (!src || !dst) return 0.0;

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < elems; i++) src[i] = (float)(i % 97);

  openblas_set_num_threads(1); /* Outer OpenMP drives parallelism. */

  double times[64];
  int maxr = max_runs();
  double cv_target = target_cv();
  int r;
  for (r = 0; r < maxr && r < (int)(sizeof(times)/sizeof(times[0])); r++) {
    double t0 = now_sec();
    #pragma omp parallel num_threads(threads)
    {
      int tid = omp_get_thread_num();
      size_t chunk = elems / threads;
      size_t start = chunk * tid;
      size_t len = (tid == threads - 1) ? (elems - start) : chunk;
      cblas_scopy((int)len, src + start, 1, dst + start, 1);
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
  double m = mean(times, runs);
  double sd = stdev(times, runs, m);
  double bw = (double)bytes / m / 1e9;
  *std_gbps = (sd > 0 ? (double)bytes / 1e9 / m * (sd / m) : 0.0);
  *runs_out = runs;
  free(src);
  free(dst);
  return bw;
}

static double bench_scopy_inner(size_t bytes, int threads, double *std_gbps, int *runs_out) {
  size_t elems = bytes / sizeof(float);
  float *src = alloc_aligned(elems * sizeof(float), 64);
  float *dst = alloc_aligned(elems * sizeof(float), 64);
  if (!src || !dst) return 0.0;

  for (size_t i = 0; i < elems; i++) src[i] = (float)(i % 97);

  openblas_set_num_threads(threads);

  double times[64];
  int maxr = max_runs();
  double cv_target = target_cv();
  int r;
  for (r = 0; r < maxr && r < (int)(sizeof(times)/sizeof(times[0])); r++) {
    double t0 = now_sec();
    cblas_scopy((int)elems, src, 1, dst, 1);
    double t1 = now_sec();
    times[r] = t1 - t0;
    if (r >= 2) {
      double m = mean(times, r + 1);
      double sd = stdev(times, r + 1, m);
      if (sd / m <= cv_target) { r++; break; }
    }
  }

  int runs = r;
  double m = mean(times, runs);
  double sd = stdev(times, runs, m);
  double bw = (double)bytes / m / 1e9;
  *std_gbps = (sd > 0 ? (double)bytes / 1e9 / m * (sd / m) : 0.0);
  *runs_out = runs;
  free(src);
  free(dst);
  return bw;
}

int main(void) {
  size_t sizes[16];
  int nsizes = build_size_list(sizes, (int)(sizeof(sizes)/sizeof(sizes[0])));
  int thread_sets[8];
  int nthreads = build_thread_list(thread_sets, (int)(sizeof(thread_sets)/sizeof(thread_sets[0])));

  printf("test,size_bytes,threads,gbps,std_gbps,runs,checksum\n");
  for (int s = 0; s < nsizes; s++) {
    size_t bytes = sizes[s];
    double cs = checksum_bytes((uint8_t *)&bytes, sizeof(bytes));
    for (int t = 0; t < nthreads; t++) {
      int thr = thread_sets[t];
      if (thr < 1) continue;
      double std, bw;
      int runs;

      bw = bench_memcpy_mt(bytes, thr, &std, &runs);
      printf("memcpy,%zu,%d,%.3f,%.3f,%d,%.1f\n", bytes, thr, bw, std, runs, cs);

      bw = bench_scopy_outer(bytes, thr, &std, &runs);
      printf("scopy_outer,%zu,%d,%.3f,%.3f,%d,%.1f\n", bytes, thr, bw, std, runs, cs);

      bw = bench_scopy_inner(bytes, thr, &std, &runs);
      printf("scopy_inner,%zu,%d,%.3f,%.3f,%d,%.1f\n", bytes, thr, bw, std, runs, cs);
      /* Emit checksums so the work cannot be optimized away. */
    }
  }
  return 0;
}
