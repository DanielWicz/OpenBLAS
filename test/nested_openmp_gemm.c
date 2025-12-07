/*
 * Nested OpenMP + OpenBLAS GEMM micro-benchmark
 *
 * Goal: show the impact of nested-team handling inside OpenBLAS GEMM.
 *
 * Two runs:
 *  - "legacy": emulate the pre-patch behavior by forcing the per-thread
 *    override openblas_set_num_threads_local(1), which collapses inner
 *    GEMM parallelism to a single thread when called from a nested region.
 *  - "patched": clear the override (openblas_set_num_threads_local(0))
 *    so OpenBLAS uses the OpenMP runtime's per-level defaults, allowing
 *    the inner GEMM to use the configured inner team size.
 *
 * The program prints wall-clock time and achieved GFLOP/s for each run.
 *
 * Build (from repository root, after building libopenblas_*):
 *   gcc -O2 -fopenmp -I. test/nested_openmp_gemm.c \
 *       libopenblas_nehalemp-r0.3.30.dev.a -lm -lpthread -o nested_omp_gemm
 *
 * Suggested env for local 16c dev box (adjust for target):
 *   export OMP_NUM_THREADS="4,2"
 *   export OMP_NESTED=TRUE
 *   export OMP_MAX_ACTIVE_LEVELS=2
 *
 * On the target 384c machine use the user-provided settings (e.g.
 * OMP_NUM_THREADS="32,12") to see the larger gap.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "cblas.h"

/* Problem size; keep moderate so it finishes quickly on 16c dev boxes. */
#ifndef GEMM_N
#define GEMM_N 512
#endif
#define OUTER_THREADS 4
#define INNER_REPS 2

static int getenv_int(const char *name, int fallback) {
  const char *v = getenv(name);
  if (!v || !*v) return fallback;
  int parsed = atoi(v);
  return parsed > 0 ? parsed : fallback;
}

static void fill_matrix(double *a, int n, double v) {
  for (int i = 0; i < n * n; i++) a[i] = v;
}

static double run_case(const char *label, int local_override,
                       int outer_threads, int inner_reps) {
  /* local_override:
   *   1 -> emulate legacy bug (force inner threads=1)
   *   0 -> patched/default behavior (follow OpenMP nested defaults)
   */
  int inner_max = 1;
#pragma omp parallel num_threads(outer_threads)
  {
    if (omp_get_thread_num() == 0)
      inner_max = omp_get_max_threads();
  }

  openblas_set_num_threads(inner_max);           /* allow full inner team */
  openblas_set_num_threads_local(local_override);/* set/clear local override */

  printf("  [%s] openblas_get_num_threads() reported: %d\n",
         label, openblas_get_num_threads());
  double *A = NULL, *B = NULL, *C = NULL;
  if (posix_memalign((void **)&A, 64, sizeof(double) * GEMM_N * GEMM_N) ||
      posix_memalign((void **)&B, 64, sizeof(double) * GEMM_N * GEMM_N) ||
      posix_memalign((void **)&C, 64, sizeof(double) * GEMM_N * GEMM_N)) {
    fprintf(stderr, "Allocation failure\n");
    exit(1);
  }
  fill_matrix(A, GEMM_N, 1.0);
  fill_matrix(B, GEMM_N, 1.0);
  memset(C, 0, sizeof(double) * GEMM_N * GEMM_N);

  double t0 = omp_get_wtime();
#pragma omp parallel num_threads(outer_threads)
  {
    if (omp_get_thread_num() == 0) {
      /* This matches what num_cpu_avail() observes inside OpenBLAS. */
      int inner_max = omp_get_max_threads();
      printf("  [%s] omp_get_max_threads() seen inside outer region: %d\n",
             label, inner_max);
    }
    for (int r = 0; r < inner_reps; ++r) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  GEMM_N, GEMM_N, GEMM_N, 1.0, A, GEMM_N, B, GEMM_N, 0.0, C, GEMM_N);
    }
  }
  double t1 = omp_get_wtime();

  /* 2*N^3 flops per GEMM */
  double gflops = (2.0 * GEMM_N * GEMM_N * GEMM_N * outer_threads * inner_reps) / 1e9;
  double secs = t1 - t0;
  printf("%s: override=%d  time=%.3f s  perf=%.2f GFLOP/s\n",
         label, local_override, secs, gflops / secs);

  free(A);
  free(B);
  free(C);
  return secs;
}

int main(void) {
  omp_set_nested(1);
  omp_set_max_active_levels(2);

  int outer_threads = getenv_int("OUTER_THREADS", OUTER_THREADS);
  int inner_reps = getenv_int("INNER_REPS", INNER_REPS);

  printf("N=%d, outer=%d, inner env from OMP_NUM_THREADS, reps=%d\n",
         GEMM_N, outer_threads, inner_reps);

  double t_legacy = run_case("legacy", 1, outer_threads, inner_reps);
  double t_patched = run_case("patched", 0, outer_threads, inner_reps);

  printf("Speedup patched vs legacy: %.2fx\n", t_legacy / t_patched);
  return 0;
}
