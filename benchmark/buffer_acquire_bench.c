#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>

#include <math.h>

static double target_cv(void) {
  const char *s = getenv("BENCH_CV");
  if (!s || !*s) return 0.03; /* 3% default */
  double v = atof(s);
  if (v <= 0.0001) v = 0.0001;
  return v;
}

static int max_runs(void) {
  const char *s = getenv("BENCH_MAX_RUNS");
  if (!s || !*s) return 30;
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

static double now(void) { return omp_get_wtime(); }

static void *aligned_alloc64(size_t bytes) {
  void *p = NULL;
  if (posix_memalign(&p, 64, bytes)) return NULL;
  return p;
}

int main(void) {
  const int N = 96;          /* balance between threading and runtime */
  const int inner_threads = 2;
  const int outer_list[] = {2, 4, 6};
  const int iters_per_thread = 3;

  size_t bytes = (size_t)N * N * sizeof(double);
  double *A = aligned_alloc64(bytes);
  double *B = aligned_alloc64(bytes);
  double *C = aligned_alloc64(bytes);
  if (!A || !B || !C) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  for (int i = 0; i < N * N; i++) {
    A[i] = 1.0;
    B[i] = 1.0;
    C[i] = 0.0;
  }

  openblas_set_num_threads(inner_threads);

  printf("outer_threads,inner_threads,calls,time_ms,us_per_call,std_us,runs\n");
  for (int idx = 0; idx < (int)(sizeof(outer_list)/sizeof(outer_list[0])); idx++) {
    int outer = outer_list[idx];
    int total_calls = outer * iters_per_thread;
    double times[64];
    int maxr = max_runs();
    double cv_target = target_cv();
    int r;
    for (r = 0; r < maxr && r < (int)(sizeof(times)/sizeof(times[0])); r++) {
      double t0 = now();
#pragma omp parallel for num_threads(outer) schedule(static)
      for (int k = 0; k < total_calls; k++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, B, N, 0.0, C, N);
      }
      double t1 = now();
      times[r] = (t1 - t0) * 1e6 / total_calls; /* us per call */
      if (r >= 2) {
        double m = mean(times, r + 1);
        double sd = stdev(times, r + 1, m);
        if (sd / m <= cv_target) { r++; break; }
      }
    }
    int runs = r;
    double m_us = mean(times, runs);
    double sd_us = stdev(times, runs, m_us);
    double ms_total = m_us * total_calls / 1e3;
    printf("%d,%d,%d,%.2f,%.2f,%.2f,%d\n",
           outer, inner_threads, total_calls, ms_total, m_us, sd_us, runs);
  }

  free(A); free(B); free(C);
  return 0;
}
