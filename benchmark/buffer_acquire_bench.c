#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>

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

  printf("outer_threads,inner_threads,calls,time_ms,us_per_call\n");
  for (int idx = 0; idx < (int)(sizeof(outer_list)/sizeof(outer_list[0])); idx++) {
    int outer = outer_list[idx];
    int total_calls = outer * iters_per_thread;
    double t0 = now();
#pragma omp parallel for num_threads(outer) schedule(static)
    for (int k = 0; k < total_calls; k++) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    }
    double t1 = now();
    double ms = (t1 - t0) * 1e3;
    double us_per = ms * 1e3 / total_calls;
    printf("%d,%d,%d,%.2f,%.2f\n", outer, inner_threads, total_calls, ms, us_per);
  }

  free(A); free(B); free(C);
  return 0;
}
