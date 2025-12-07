#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "cblas.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int env_int(const char *name, int def) {
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

static size_t round_up(size_t val, size_t align) {
    return (val + align - 1) / align * align;
}

int main(void) {
    const int outer_threads = env_int("OUTER_THREADS", 2);
    const int inner_reps    = env_int("INNER_REPS", 4);
    const int M = env_int("BENCH_M", 2048);
    const int N = env_int("BENCH_N", 1024);
    const int K = env_int("BENCH_K", 16);

    printf("membw_nested_gemm: OUTER_THREADS=%d INNER_REPS=%d M=%d N=%d K=%d\n",
           outer_threads, inner_reps, M, N, K);
    printf("OpenBLAS configured threads (outer runtime view): %d\n", openblas_get_num_threads());

    const size_t bytes_A = (size_t)M * K * sizeof(double);
    const size_t bytes_B = (size_t)K * N * sizeof(double);
    const size_t bytes_C = (size_t)M * N * sizeof(double);

    double wall_start = now_sec();
    double copy_bytes_total = 0.0;
    double flop_total = 0.0;

#pragma omp parallel num_threads(outer_threads) reduction(+:copy_bytes_total, flop_total)
    {
        int tid = omp_get_thread_num();
        double *A = NULL, *B = NULL, *C = NULL;
        if (posix_memalign((void **)&A, 64, round_up(bytes_A, 64)) ||
            posix_memalign((void **)&B, 64, round_up(bytes_B, 64)) ||
            posix_memalign((void **)&C, 64, round_up(bytes_C, 64))) {
            fprintf(stderr, "tid %d: allocation failed\n", tid);
            exit(1);
        }
        /* Initialize once so first touch happens on the worker thread. */
        for (size_t i = 0; i < bytes_A / sizeof(double); i += 64) A[i] = 1.0;
        for (size_t i = 0; i < bytes_B / sizeof(double); i += 64) B[i] = 1.0;
        memset(C, 0, bytes_C);

        double t0 = now_sec();
        for (int r = 0; r < inner_reps; ++r) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0, A, M, B, K, 0.0, C, M);
            copy_bytes_total += (double)(bytes_A + bytes_B);
            flop_total += 2.0 * (double)M * (double)N * (double)K;
        }
        double t1 = now_sec();

        printf("tid %d: time %.3f s, bytes %.1f MB\n", tid, t1 - t0, (double)(bytes_A + bytes_B) * inner_reps / 1e6);

        free(A);
        free(B);
        free(C);
    }

    double wall = now_sec() - wall_start;
    double gflops = flop_total / wall / 1e9;
    double gbytes = copy_bytes_total / wall / 1e9;
    printf("Total wall %.3f s  ~%.2f GFLOPS  copy_bw ~%.2f GB/s (approx pack traffic)\n",
           wall, gflops, gbytes);
    return 0;
}
