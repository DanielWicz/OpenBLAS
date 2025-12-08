#!/usr/bin/env bash
set -euo pipefail

# Usage: benchmark/compare_bench.sh <dirA> <dirB>
# Runs memcopy_bench, buffer_acquire_bench, and (if present) numa_copy_bench
# in both trees and prints a side-by-side speed comparison.

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <current_repo_dir> <compare_repo_dir>" >&2
  exit 1
fi

DIR_A=$(cd "$1" && pwd)
DIR_B=$(cd "$2" && pwd)
TMP=$(mktemp -d)
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

compile_and_run() {
  local dir="$1" label="$2"
  pushd "$dir" >/dev/null
  gcc -O2 -fopenmp -I. benchmark/memcopy_bench.c libopenblas_nehalemp-r0.3.30.dev.a -lm -lpthread -o benchmark/memcopy_bench
  gcc -O2 -fopenmp -I. benchmark/buffer_acquire_bench.c libopenblas_nehalemp-r0.3.30.dev.a -lm -lpthread -o benchmark/buffer_acquire_bench
  OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} ./benchmark/memcopy_bench > "$TMP/memcopy_${label}.csv"
  OMP_NESTED=TRUE OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=${OMP_NESTED_THREADS:-4} ./benchmark/buffer_acquire_bench > "$TMP/buffer_${label}.csv"
  if [ -f benchmark/numa_copy_bench.c ]; then
    gcc -O2 -fopenmp -I. benchmark/numa_copy_bench.c -lm -lnuma -o benchmark/numa_copy_bench
    OMP_NUM_THREADS=${OMP_NUMA_THREADS:-8} OMP_PROC_BIND=spread ./benchmark/numa_copy_bench > "$TMP/numa_${label}.csv"
  fi
  popd >/dev/null
}

compile_and_run "$DIR_A" "A"
compile_and_run "$DIR_B" "B"

python - "$TMP" <<'PY'
import csv, sys, os
TMP = sys.argv[1]
from collections import defaultdict

def load_mem(path):
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row['test'], int(row['size_bytes']), int(row['threads']))
            rows[key] = float(row['gbps'])
    return rows

def load_buf(path):
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (int(row['outer_threads']), int(row['inner_threads']))
            rows[key] = float(row['us_per_call'])
    return rows

memA = load_mem(f"{TMP}/memcopy_A.csv")
memB = load_mem(f"{TMP}/memcopy_B.csv")
bufA = load_buf(f"{TMP}/buffer_A.csv")
bufB = load_buf(f"{TMP}/buffer_B.csv")

def load_numa(label):
    path = f"{TMP}/numa_{label}.csv"
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            key = (row['case'], int(row['size_bytes']), int(row['threads']))
            rows[key] = float(row['agg_gbps'])
    return rows

numaA = load_numa("A")
numaB = load_numa("B")

print("=== memcopy_bench (GB/s) ===")
print("test,size,thr,A,B,speedup")
for key in sorted(memA.keys()):
    a = memA[key]
    b = memB.get(key, 0.0)
    speed = (a / b) if b else 0.0
    print(f"{key[0]},{key[1]},{key[2]},{a:.3f},{b:.3f},{speed:.2f}x")

print("\n=== buffer_acquire_bench (lower is better, us/call) ===")
print("outer,inner,A_us,B_us,improvement")
for key in sorted(bufA.keys()):
    a = bufA[key]
    b = bufB.get(key, 0.0)
    speed = (b / a) if a else 0.0
    print(f"{key[0]},{key[1]},{a:.2f},{b:.2f},{speed:.2f}x")

if numaA:
    print("\n=== numa_copy_bench (GB/s) ===")
    print("case,size,thr,A,B,speedup")
    for key in sorted(numaA.keys()):
        a = numaA[key]
        b = numaB.get(key, 0.0)
        speed = (a / b) if b else 0.0
        print(f"{key[0]},{key[1]},{key[2]},{a:.3f},{b:.3f},{speed:.2f}x")
PY
