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

# Number of full sweeps to average (default 3). Set SWEEPS=1 to keep old behavior.
SWEEPS=${SWEEPS:-3}

average_results() {
    python3 - "$@" <<'PY'
import re, sys
files = sys.argv[1:]
lines = [open(f).read().splitlines() for f in files]
if not lines:
    sys.exit("No result files")
count = len(lines)
out = []
metric_re = re.compile(r'(N=\d+.*Time=)([0-9.]+)(\s+us,\s+BW=)([0-9.]+)(\s+GB/s,\s+GFLOPS=)([0-9.]+)')
conc_re   = re.compile(r'(Concurrent:.*AvgTime=)([0-9.]+)(\s+us,\s+AggBW=)([0-9.]+)(\s+GB/s,\s+AggGFLOPS=)([0-9.]+)')
rows = len(lines[0])
for i in range(rows):
    first = lines[0][i]
    m = metric_re.match(first)
    mc = conc_re.match(first)
    if m or mc:
        vals = []
        for lset in lines:
            cur = lset[i]
            mm = metric_re.match(cur) or conc_re.match(cur)
            if not mm:
                sys.exit(f"Line mismatch on index {i}: {cur}")
            vals.append(tuple(float(mm.group(j)) for j in (2,4,6)))
        avg = [sum(v[j] for v in vals)/count for j in range(3)]
        g = m or mc
        out.append(f"{g.group(1)}{avg[0]:.6f}{g.group(3)}{avg[1]:.2f}{g.group(5)}{avg[2]:.2f}")
    else:
        if first.strip():
            out.append(first)
print("\n".join(out))
PY
}

run_and_avg() {
    out="$1"; shift
    tmpdir=$(mktemp -d)
    for s in $(seq 1 $SWEEPS); do
        ./benchmark_optimization/bench_copy "$@" > "$tmpdir/run$s.txt"
    done
    average_results "$tmpdir"/run*.txt > "$out"
    rm -rf "$tmpdir"
}

# Op Types: 0=COPY, 1=AXPY, 2=DOT, 3=SCAL, 4=NRM2, 5=ASUM

echo "=== COPY Benchmark ==="
echo "Running Single Thread Sweep (COPY)..."
OMP_NUM_THREADS=8 run_and_avg benchmark_optimization/result_copy_single.txt 0 0
echo "Running Concurrent Sweep (COPY, 8 threads)..."
OMP_NUM_THREADS=1 run_and_avg benchmark_optimization/result_copy_conc_8.txt 0 1 8

echo "=== AXPY Benchmark ==="
echo "Running Single Thread Sweep (AXPY)..."
OMP_NUM_THREADS=8 run_and_avg benchmark_optimization/result_axpy_single.txt 1 0
echo "Running Concurrent Sweep (AXPY, 8 threads)..."
OMP_NUM_THREADS=1 run_and_avg benchmark_optimization/result_axpy_conc_8.txt 1 1 8

echo "=== DOT Benchmark ==="
echo "Running Single Thread Sweep (DOT)..."
OMP_NUM_THREADS=8 run_and_avg benchmark_optimization/result_dot_single.txt 2 0
echo "Running Concurrent Sweep (DOT, 8 threads)..."
OMP_NUM_THREADS=1 run_and_avg benchmark_optimization/result_dot_conc_8.txt 2 1 8

echo "=== SCAL Benchmark ==="
echo "Running Single Thread Sweep (SCAL)..."
OMP_NUM_THREADS=8 run_and_avg benchmark_optimization/result_scal_single.txt 3 0
echo "Running Concurrent Sweep (SCAL, 8 threads)..."
OMP_NUM_THREADS=1 run_and_avg benchmark_optimization/result_scal_conc_8.txt 3 1 8

echo "=== NRM2 Benchmark ==="
echo "Running Single Thread Sweep (NRM2)..."
OMP_NUM_THREADS=8 run_and_avg benchmark_optimization/result_nrm2_single.txt 4 0
echo "Running Concurrent Sweep (NRM2, 8 threads)..."
OMP_NUM_THREADS=1 run_and_avg benchmark_optimization/result_nrm2_conc_8.txt 4 1 8

echo "=== ASUM Benchmark ==="
echo "Running Single Thread Sweep (ASUM)..."
OMP_NUM_THREADS=8 run_and_avg benchmark_optimization/result_asum_single.txt 5 0
echo "Running Concurrent Sweep (ASUM, 8 threads)..."
OMP_NUM_THREADS=1 run_and_avg benchmark_optimization/result_asum_conc_8.txt 5 1 8

echo "Done. Results in benchmark_optimization/result_*.txt"
