# Agent Notes

- Always read `not_works.md` before attempting performance changes; it lists recent failed experiments and benchmarks to avoid repeating regressions. If you fail - you must also update the file with new records of detailed implementation, failure and what you did (very detailed).
- Use the existing `benchmark/multi_instance_copy_bench` and `benchmark_optimization/run_bench.sh` to validate multi-instance and Level-1 changes under `USE_TLS=1`.
- Keep untracked benchmark outputs out of commits; only add new artifacts that are explicitly requested.
- New benchmarks:
  - `benchmark/numa_copy_bench.c`: NUMA-aware memcpy microbench. Measures local, remote-read, remote-write, and bidirectional copy bandwidth with thread pinning; exercises cache/L3/RAM and cross-socket traffic to spot multi-instance bottlenecks under `USE_TLS=1`. Build: `gcc -O2 -fopenmp -lnuma -I. benchmark/numa_copy_bench.c -o benchmark/numa_copy_bench`; run with `OMP_NUM_THREADS=...` (set `BENCH_SIZES`, `BENCH_THREADS`, `BENCH_CASES` to shape coverage).
