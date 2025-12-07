/*
 * Regression test for nested OpenMP + OpenBLAS deadlock (USE_TLS=1, NUM_PARALLEL>1)
 *
 * The historical issue: with nested OpenMP enabled and blas_omp_threads_local set
 * to follow the runtime (value 0), each outer OpenMP thread could spawn an inner
 * OpenBLAS team, exhausting the limited NUM_PARALLEL buffer slots and hanging in
 * exec_blas() while trying to acquire a buffer. The fix restored the legacy
 * behavior (blas_omp_threads_local=1) so inner parallelism is suppressed when
 * called from inside an OpenMP region.
 *
 * This test exercises the deadlock scenario: an outer OpenMP team calls GEMM
 * repeatedly while nested OpenMP is allowed. If the deadlock regresses, the
 * program will hang; a POSIX alarm terminates it with a non‑zero exit, causing
 * the test suite to fail instead of stalling indefinitely.
 */

#define _POSIX_C_SOURCE 200809L
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cblas.h"

#ifndef DEADLOCK_TIMEOUT_SEC
#define DEADLOCK_TIMEOUT_SEC 10
#endif

#ifndef OUTER_THREADS_DEFAULT
#define OUTER_THREADS_DEFAULT 4
#endif

#ifndef GEMM_N
#define GEMM_N 96
#endif

static void on_alarm(int sig) {
  (void)sig;
  fprintf(stderr, "deadlock_nested_omp: timed out (possible deadlock)\n");
  _Exit(1);
}

static int getenv_int(const char *name, int fallback) {
  const char *v = getenv(name);
  if (!v || !*v) return fallback;
  int parsed = atoi(v);
  return parsed > 0 ? parsed : fallback;
}

int main(void) {
  /* Set an alarm so the test fails fast instead of hanging the entire suite. */
  struct sigaction sa = {.sa_handler = on_alarm};
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGALRM, &sa, NULL);
  alarm(DEADLOCK_TIMEOUT_SEC);

  /* Enable nested OpenMP; outer team size is fixed, inner is determined by
   * the OpenBLAS nested handling (the bug was here). */
  omp_set_nested(1);
  omp_set_max_active_levels(2);

  int outer_threads = getenv_int("OUTER_THREADS", OUTER_THREADS_DEFAULT);
  int calls_per_thread = getenv_int("CALLS_PER_THREAD", 4);

  size_t bytes = (size_t)GEMM_N * GEMM_N * sizeof(double);
  double *A = NULL, *B = NULL, *C = NULL;
  if (posix_memalign((void **)&A, 64, bytes) ||
      posix_memalign((void **)&B, 64, bytes) ||
      posix_memalign((void **)&C, 64, bytes)) {
    fprintf(stderr, "deadlock_nested_omp: allocation failed\n");
    return 1;
  }
  for (int i = 0; i < GEMM_N * GEMM_N; i++) {
    A[i] = 1.0;
    B[i] = 1.0;
    C[i] = 0.0;
  }

  /* Outer parallel region; each thread issues several GEMMs. If OpenBLAS tries
   * to spawn inner teams here and the buffer-slot accounting regresses, this
   * loop will hang and the alarm will abort the test. */
#pragma omp parallel num_threads(outer_threads)
  {
    for (int r = 0; r < calls_per_thread; r++) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  GEMM_N, GEMM_N, GEMM_N, 1.0,
                  A, GEMM_N, B, GEMM_N, 0.0, C, GEMM_N);
    }
  }

  alarm(0); /* success */
  free(A);
  free(B);
  free(C);
  printf("deadlock_nested_omp: completed without hang (outer=%d, calls/thread=%d)\n",
         outer_threads, calls_per_thread);
  return 0;
}
