#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cblas.h>

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

static double checksum_bytes(const uint8_t *p, size_t n) {
  double acc = 0.0;
  for (size_t i = 0; i < n; i += 4096) acc += p[i];
  if (n) acc += p[n - 1];
  return acc;
}

static double bench_memcpy_mt(size_t bytes, int threads) {
  uint8_t *src = alloc_aligned(bytes, 64);
  uint8_t *dst = alloc_aligned(bytes, 64);
  if (!src || !dst) return 0.0;

  /* Fill source to avoid lazy allocation penalties later. */
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < bytes; i++) src[i] = (uint8_t)(i * 13 + 7);

  double best = 1e30;
  for (int r = 0; r < 5; r++) {
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
    double elapsed = t1 - t0;
    if (elapsed < best) best = elapsed;
  }

  double bw = (double)bytes / best / 1e9;
  free(src);
  free(dst);
  return bw;
}

static double bench_scopy_outer(size_t bytes, int threads) {
  size_t elems = bytes / sizeof(float);
  float *src = alloc_aligned(elems * sizeof(float), 64);
  float *dst = alloc_aligned(elems * sizeof(float), 64);
  if (!src || !dst) return 0.0;

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < elems; i++) src[i] = (float)(i % 97);

  openblas_set_num_threads(1); /* Outer OpenMP drives parallelism. */

  double best = 1e30;
  for (int r = 0; r < 5; r++) {
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
    double elapsed = t1 - t0;
    if (elapsed < best) best = elapsed;
  }

  double bw = (double)bytes / best / 1e9;
  free(src);
  free(dst);
  return bw;
}

static double bench_scopy_inner(size_t bytes, int threads) {
  size_t elems = bytes / sizeof(float);
  float *src = alloc_aligned(elems * sizeof(float), 64);
  float *dst = alloc_aligned(elems * sizeof(float), 64);
  if (!src || !dst) return 0.0;

  for (size_t i = 0; i < elems; i++) src[i] = (float)(i % 97);

  openblas_set_num_threads(threads);

  double best = 1e30;
  for (int r = 0; r < 5; r++) {
    double t0 = now_sec();
    cblas_scopy((int)elems, src, 1, dst, 1);
    double t1 = now_sec();
    if (t1 - t0 < best) best = t1 - t0;
  }

  double bw = (double)bytes / best / 1e9;
  free(src);
  free(dst);
  return bw;
}

int main(void) {
  const size_t sizes[] = {
    8 * 1024,          /* L1 sized copy (core-core) */
    64 * 1024,         /* L1/L2 boundary */
    1 * 1024 * 1024,   /* L2/L3 boundary */
    32 * 1024 * 1024,  /* LLC / memory */
    128 * 1024 * 1024  /* RAM-dominated */
  };
  int threads_max = omp_get_max_threads();
  int thread_sets[3];
  thread_sets[0] = 1;
  thread_sets[1] = threads_max >= 2 ? threads_max / 2 : 1;
  thread_sets[2] = threads_max;

  printf("test,size_bytes,threads,gbps,checksum\n");
  for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
    size_t bytes = sizes[s];
    for (int t = 0; t < 3; t++) {
      int thr = thread_sets[t];
      if (thr < 1) continue;
      double bw_memcpy = bench_memcpy_mt(bytes, thr);
      double bw_scopy_outer  = bench_scopy_outer(bytes, thr);
      double bw_scopy_inner  = bench_scopy_inner(bytes, thr);
      /* Emit checksums so the work cannot be optimized away. */
      uint8_t sample = (uint8_t)((bytes ^ thr) & 0xFF);
      double cs = checksum_bytes(&sample, 1);
      printf("memcpy,%zu,%d,%.3f,%.1f\n", bytes, thr, bw_memcpy, cs);
      printf("scopy_outer,%zu,%d,%.3f,%.1f\n", bytes, thr, bw_scopy_outer, cs);
      printf("scopy_inner,%zu,%d,%.3f,%.1f\n", bytes, thr, bw_scopy_inner, cs);
    }
  }
  return 0;
}
