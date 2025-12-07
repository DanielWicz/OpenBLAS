#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>
#include "cblas.h"

#define MAX_REPEATS 20
#define MIN_REPEATS 5
#define MIN_TIME 0.5 // seconds

// 64-byte alignment for AVX
#define ALIGNMENT 64

void* aligned_alloc_wrapper(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, ALIGNMENT, size) != 0) {
        return NULL;
    }
    return ptr;
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void bench_op_single(size_t n, int op_type) {
    double *x, *y;
    size_t size_bytes = n * sizeof(double);
    
    fprintf(stderr, "DEBUG: bench_op_single N=%zu Op=%d\n", n, op_type);

    x = aligned_alloc_wrapper(size_bytes);
    y = aligned_alloc_wrapper(size_bytes);
    
    if (!x || !y) {
        fprintf(stderr, "Error: Allocation failed for size %zu\n", size_bytes);
        if(x) free(x);
        if(y) free(y);
        return;
    }
    
    // Initialize
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        x[i] = (double)i;
        y[i] = 0.0;
    }

    double start, end, time_avg;
    int reps = 0;
    
    // Warmup
    if (op_type == 0) cblas_dcopy(n, x, 1, y, 1);
    else if (op_type == 1) cblas_daxpy(n, 1.0, x, 1, y, 1);
    else if (op_type == 2) cblas_ddot(n, x, 1, y, 1);
    else if (op_type == 3) cblas_dscal(n, 0.5, x, 1);
    else if (op_type == 4) cblas_dnrm2(n, x, 1);
    else if (op_type == 5) cblas_dasum(n, x, 1);

    start = get_time();
    while (get_time() - start < MIN_TIME || reps < MIN_REPEATS) {
        if (op_type == 0) cblas_dcopy(n, x, 1, y, 1);
        else if (op_type == 1) cblas_daxpy(n, 1.0, x, 1, y, 1);
        else if (op_type == 2) cblas_ddot(n, x, 1, y, 1);
        else if (op_type == 3) cblas_dscal(n, 0.5, x, 1);
        else if (op_type == 4) cblas_dnrm2(n, x, 1);
        else if (op_type == 5) cblas_dasum(n, x, 1);
        reps++;
    }
    end = get_time();
    
    time_avg = (end - start) / reps;
    
    double bw = 0.0;
    double gflops = 0.0;
    
    if (op_type == 0) { // COPY
        // Read X, Write Y = 2 * size
        bw = (2.0 * size_bytes) / time_avg / 1e9;
    } else if (op_type == 1) { // AXPY
        // Read X, Read Y, Write Y = 3 * size (usually Y is RMW)
        bw = (3.0 * size_bytes) / time_avg / 1e9;
        // Y = a*X + Y. 1 mul, 1 add = 2 FLOPs per element
        gflops = (2.0 * n) / time_avg / 1e9;
    } else if (op_type == 2) { // DOT
        // Read X, Read Y = 2 * size
        bw = (2.0 * size_bytes) / time_avg / 1e9;
        // sum += x*y. 1 mul, 1 add = 2 FLOPs per element
        gflops = (2.0 * n) / time_avg / 1e9;
    } else if (op_type == 3) { // SCAL
        // Read/Write X = 2 * size
        bw = (2.0 * size_bytes) / time_avg / 1e9;
        gflops = (1.0 * n) / time_avg / 1e9; // one multiply per element
    } else if (op_type == 4) { // NRM2
        // Read X once
        bw = (1.0 * size_bytes) / time_avg / 1e9;
        gflops = (2.0 * n) / time_avg / 1e9; // mul+add per element
    } else if (op_type == 5) { // ASUM
        // Read X once
        bw = (1.0 * size_bytes) / time_avg / 1e9;
        gflops = (1.0 * n) / time_avg / 1e9; // abs+add ~1 op counted
    }

    printf("N=%zu, Size=%.2f KB, Time=%.6f us, BW=%.2f GB/s, GFLOPS=%.2f\n", 
           n, size_bytes / 1024.0, time_avg * 1e6, bw, gflops);

    free(x);
    free(y);
}

typedef struct {
    size_t n;
    int iterations;
    int op_type;
    double time_total;
} thread_arg_t;

void* thread_func(void* arg) {
    thread_arg_t* t_arg = (thread_arg_t*)arg;
    size_t n = t_arg->n;
    size_t size_bytes = n * sizeof(double);
    
    double *x = aligned_alloc_wrapper(size_bytes);
    double *y = aligned_alloc_wrapper(size_bytes);
    
    if (!x || !y) {
        fprintf(stderr, "Error: Allocation failed in thread for size %zu\n", size_bytes);
        if(x) free(x);
        if(y) free(y);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        x[i] = (double)i;
        y[i] = (double)i;
    }

    // Warmup
    if (t_arg->op_type == 0) cblas_dcopy(n, x, 1, y, 1);
    else if (t_arg->op_type == 1) cblas_daxpy(n, 1.0, x, 1, y, 1);
    else if (t_arg->op_type == 2) cblas_ddot(n, x, 1, y, 1);
    else if (t_arg->op_type == 3) cblas_dscal(n, 0.5, x, 1);
    else if (t_arg->op_type == 4) cblas_dnrm2(n, x, 1);
    else if (t_arg->op_type == 5) cblas_dasum(n, x, 1);

    double start = get_time();
    for (int i = 0; i < t_arg->iterations; i++) {
        if (t_arg->op_type == 0) cblas_dcopy(n, x, 1, y, 1);
        else if (t_arg->op_type == 1) cblas_daxpy(n, 1.0, x, 1, y, 1);
        else if (t_arg->op_type == 2) cblas_ddot(n, x, 1, y, 1);
        else if (t_arg->op_type == 3) cblas_dscal(n, 0.5, x, 1);
        else if (t_arg->op_type == 4) cblas_dnrm2(n, x, 1);
        else if (t_arg->op_type == 5) cblas_dasum(n, x, 1);
    }
    double end = get_time();

    t_arg->time_total = end - start;

    free(x);
    free(y);
    return NULL;
}

void bench_op_concurrent(size_t n, int num_threads, int op_type) {
    fprintf(stderr, "DEBUG: bench_op_concurrent N=%zu Threads=%d Op=%d\n", n, num_threads, op_type);
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t* args = malloc(num_threads * sizeof(thread_arg_t));
    
    int iterations = 100;
    // Adjust iterations for small/large N
    if (n > 1000000) iterations = 10;
    if (n > 10000000) iterations = 5;

    for (int i = 0; i < num_threads; i++) {
        args[i].n = n;
        args[i].iterations = iterations;
        args[i].op_type = op_type;
        pthread_create(&threads[i], NULL, thread_func, &args[i]);
    }

    double total_time_sum = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_time_sum += args[i].time_total;
    }

    double avg_time_per_op = (total_time_sum / num_threads) / iterations;
    double total_size_bytes = n * sizeof(double);
    
    double agg_bw = 0.0;
    double agg_gflops = 0.0;
    
    if (op_type == 0) { // COPY
        agg_bw = (2.0 * total_size_bytes * num_threads) / avg_time_per_op / 1e9;
    } else if (op_type == 1) { // AXPY
        agg_bw = (3.0 * total_size_bytes * num_threads) / avg_time_per_op / 1e9;
        agg_gflops = (2.0 * n * num_threads) / avg_time_per_op / 1e9;
    } else if (op_type == 2) { // DOT
        agg_bw = (2.0 * total_size_bytes * num_threads) / avg_time_per_op / 1e9;
        agg_gflops = (2.0 * n * num_threads) / avg_time_per_op / 1e9;
    } else if (op_type == 3) { // SCAL
        agg_bw = (2.0 * total_size_bytes * num_threads) / avg_time_per_op / 1e9;
        agg_gflops = (1.0 * n * num_threads) / avg_time_per_op / 1e9;
    } else if (op_type == 4) { // NRM2
        agg_bw = (1.0 * total_size_bytes * num_threads) / avg_time_per_op / 1e9;
        agg_gflops = (2.0 * n * num_threads) / avg_time_per_op / 1e9;
    } else if (op_type == 5) { // ASUM
        agg_bw = (1.0 * total_size_bytes * num_threads) / avg_time_per_op / 1e9;
        agg_gflops = (1.0 * n * num_threads) / avg_time_per_op / 1e9;
    }
    
    printf("Concurrent: Threads=%d, N=%zu, Size=%.2f KB, AvgTime=%.6f us, AggBW=%.2f GB/s, AggGFLOPS=%.2f\n", 
           num_threads, n, total_size_bytes / 1024.0, avg_time_per_op * 1e6, agg_bw, agg_gflops);

    free(threads);
    free(args);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <op_type> <mode> [args]\n", argv[0]);
        printf("Op Types: 0=COPY, 1=AXPY, 2=DOT, 3=SCAL, 4=NRM2, 5=ASUM\n");
        printf("Mode 0: Single Sweep\n");
        printf("Mode 1: Concurrent Sweep <num_concurrent_threads>\n");
        return 1;
    }

    int op_type = atoi(argv[1]);
    int mode = atoi(argv[2]);

    // Sizes to test: 4KB to 256MB
    size_t sizes[] = {
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, // Small (L1/L2)
        131072, 262144, 524288, 1048576, 2097152, 4194304, // Medium (L3)
        8388608, 16777216, 33554432, 67108864 // Large (RAM)
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    openblas_set_num_threads(omp_get_max_threads());

    if (mode == 0) {
        printf("Running Single Thread/Process Sweep (Op=%d)...\n", op_type);
        printf("OpenBLAS Threads: %d\n", openblas_get_num_threads());
        for (int i = 0; i < num_sizes; i++) {
            bench_op_single(sizes[i], op_type);
        }
    } else if (mode == 1) {
        if (argc < 4) {
            printf("Need num_concurrent_threads for mode 1\n");
            return 1;
        }
        int conc_threads = atoi(argv[3]);
        printf("Running Concurrent Sweep with %d threads (Op=%d)...\n", conc_threads, op_type);
        
        for (int i = 0; i < num_sizes; i++) {
            bench_op_concurrent(sizes[i], conc_threads, op_type);
        }
    }

    return 0;
}
