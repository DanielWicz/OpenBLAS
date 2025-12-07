#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include <omp.h>

/* 
 * Test case to verify stability when OMP_NUM_THREADS > MAX_CPU_NUMBER (compile-time limit).
 * This simulates the "BIGNUMA" or high-core-count scenario where the thread ID 
 * returned by OpenMP exceeds the static buffer array size in blas_server_omp.c.
 */

int main(int argc, char **argv) {
    int i;
    int n = 2048;
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));
    
    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    for (i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    /* 
     * Force OpenBLAS to try and use more threads than the likely build-time MAX_CPU_NUMBER (often 8 or 16).
     * We set a very high number to be safe.
     */
    int num_threads = 64; 
    printf("Requesting %d threads via openblas_set_num_threads...\n", num_threads);
    openblas_set_num_threads(num_threads);

    printf("Running DGEMM (N=%d)...\n", n);
    
    // Perform matrix multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1.0, A, n, B, n, 0.0, C, n);

    printf("DGEMM completed successfully.\n");

    free(A);
    free(B);
    free(C);
    return 0;
}
