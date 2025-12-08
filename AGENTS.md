# Agent Notes

- Always read `not_works.md` before attempting performance changes; it lists recent failed experiments and benchmarks to avoid repeating regressions. If you fail - you must also update the file with new records of detailed implementation, failure and what you did (very detailed).
- Use the existing `benchmark/multi_instance_copy_bench` and `benchmark_optimization/run_bench.sh` to validate multi-instance and Level-1 changes under `USE_TLS=1`.
- Keep untracked benchmark outputs out of commits; only add new artifacts that are explicitly requested.
- Benchmark catalogue (keep binaries/results untracked):
  - `benchmark/multi_instance_copy_bench.c`: Stresses concurrent instances (instances × inner threads) for memcpy vs BLAS scopy across problem sizes; CV-based repeat-until-stable. Build: `gcc -O2 -fopenmp -I. benchmark/multi_instance_copy_bench.c libopenblas*.a -lpthread -lm -o benchmark/multi_instance_copy_bench`; run with `OMP_NESTED=TRUE OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=<outer> ./benchmark/multi_instance_copy_bench`.
  - `benchmark/buffer_acquire_bench.c`: Measures TLS buffer slot acquisition latency under nested OpenMP (NUM_PARALLEL contention). Build similarly; run with nested OpenMP to profile exec_blas spin/slot behavior.
  - `benchmark/memcopy_bench.c`: Single-team memcpy vs BLAS copy sweep across cache/L3/RAM sizes and thread counts; configurable via `BENCH_SIZES`/`BENCH_THREADS`.
  - `benchmark_optimization/run_bench.sh`: Wrapper that compiles `bench_copy.c` and runs copy/axpy/dot/scal/nrm2/asum in single-thread and 8-way concurrent modes, averaging multiple sweeps (controlled by `SWEEPS`).
  - `benchmark/numa_copy_bench.c`: NUMA-aware memcpy microbench. Measures local, remote-read, remote-write, and bidirectional bandwidth with thread pinning; exercises cache/L3/RAM and cross-socket paths to spot multi-instance bottlenecks under `USE_TLS=1`. Build: `gcc -O2 -fopenmp -I. benchmark/numa_copy_bench.c -lm -lnuma -o benchmark/numa_copy_bench`; run with `OMP_NUM_THREADS=...` (tune `BENCH_SIZES`, `BENCH_THREADS`, `BENCH_CASES`, `BENCH_NODEA/B`).
