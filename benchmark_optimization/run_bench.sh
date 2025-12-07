#!/bin/bash
set -e

# Compile the benchmark
if [ -f "../libopenblas.a" ]; then
    LIB="../libopenblas.a"
    INC="-I.."
elif [ -f "libopenblas.a" ]; then
    LIB="libopenblas.a"
    INC="-I."
else
    echo "Error: libopenblas.a not found"
    exit 1
fi

echo "Compiling benchmark..."
gcc -O3 -fopenmp $INC benchmark_optimization/bench_copy.c -o benchmark_optimization/bench_copy $LIB -lpthread -lm

# Op Types: 0=COPY, 1=AXPY, 2=DOT

echo "=== COPY Benchmark ==="
echo "Running Single Thread Sweep (COPY)..."
OMP_NUM_THREADS=8 ./benchmark_optimization/bench_copy 0 0 > benchmark_optimization/result_copy_single.txt
echo "Running Concurrent Sweep (COPY, 8 threads)..."
OMP_NUM_THREADS=1 ./benchmark_optimization/bench_copy 0 1 8 > benchmark_optimization/result_copy_conc_8.txt

echo "=== AXPY Benchmark ==="
echo "Running Single Thread Sweep (AXPY)..."
OMP_NUM_THREADS=8 ./benchmark_optimization/bench_copy 1 0 > benchmark_optimization/result_axpy_single.txt
echo "Running Concurrent Sweep (AXPY, 8 threads)..."
OMP_NUM_THREADS=1 ./benchmark_optimization/bench_copy 1 1 8 > benchmark_optimization/result_axpy_conc_8.txt

echo "=== DOT Benchmark ==="
echo "Running Single Thread Sweep (DOT)..."
OMP_NUM_THREADS=8 ./benchmark_optimization/bench_copy 2 0 > benchmark_optimization/result_dot_single.txt
echo "Running Concurrent Sweep (DOT, 8 threads)..."
OMP_NUM_THREADS=1 ./benchmark_optimization/bench_copy 2 1 8 > benchmark_optimization/result_dot_conc_8.txt

echo "Done. Results in benchmark_optimization/result_*.txt"