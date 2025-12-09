# Agent Notes

- Always read `not_works.md` before attempting performance changes; it lists recent failed experiments and benchmarks to avoid repeating regressions. If you fail - you must also update the file with new records of detailed implementation, failure and what you did (very detailed).
- Use the existing `benchmark/multi_instance_copy_bench` and `benchmark_optimization/run_bench.sh` to validate multi-instance and Level-1 changes under `USE_TLS=1`.
- Keep untracked benchmark outputs out of commits; only add new artifacts that are explicitly requested.
- Benchmark catalogue (keep binaries/results untracked):
  - `benchmark/multi_instance_copy_bench.c`: Stresses concurrent instances (instances × inner threads) for memcpy vs BLAS scopy across problem sizes; CV-based repeat-until-stable with default `BENCH_MAX_RUNS=50` and std% emitted for variance checks. Build: `gcc -O2 -fopenmp -I. benchmark/multi_instance_copy_bench.c libopenblas*.a -lpthread -lm -o benchmark/multi_instance_copy_bench`; run with `OMP_NESTED=TRUE OMP_MAX_ACTIVE_LEVELS=2 OMP_PROC_BIND=spread OMP_NUM_THREADS=<outer> ./benchmark/multi_instance_copy_bench`.
  - `benchmark/buffer_acquire_bench.c`: Measures TLS buffer slot acquisition latency under nested OpenMP (NUM_PARALLEL contention); defaults to 50 samples and reports std% relative to mean to expose contention variance. Build similarly; run with nested OpenMP to profile exec_blas spin/slot behavior.
  - `benchmark/memcopy_bench.c`: Single-team memcpy vs BLAS copy sweep across cache/L3/RAM sizes and thread counts; defaults to 50 samples per point, emits std% for each bandwidth to compare variance; configurable via `BENCH_SIZES`/`BENCH_THREADS`. Reports both GB/s and derived GFLOPS (copy counted as 1 flop/element).
  - `benchmark_optimization/run_bench.sh`: Wrapper that compiles `bench_copy.c` and runs copy/axpy/dot/scal/nrm2/asum in single-thread and 8-way concurrent modes, averaging multiple sweeps (controlled by `SWEEPS`).
  - `benchmark/numa_copy_bench.c`: NUMA-aware memcpy microbench. Measures local, remote-read, remote-write, and bidirectional bandwidth with thread pinning; exercises cache/L3/RAM and cross-socket paths to spot multi-instance bottlenecks under `USE_TLS=1`. Build: `gcc -O2 -fopenmp -I. benchmark/numa_copy_bench.c -lm -lnuma -o benchmark/numa_copy_bench`; run with `OMP_NUM_THREADS=...` (tune `BENCH_SIZES`, `BENCH_THREADS`, `BENCH_CASES`, `BENCH_NODEA/B`). Outputs GB/s and GFLOPS plus per-thread breakdowns.
  - `benchmark/multi_instance_copy_bench.c` now also reports aggregate/per-instance GFLOPS alongside GB/s to make cross-size comparisons easier.
- Benchmarking workflow (keep artifacts untracked):
  - Build benches after every code change: `gcc -O2 -fopenmp -I. benchmark/<bench>.c libopenblas*.a -lpthread -lm -o benchmark/<bench>` (add `-lnuma` for `numa_copy_bench`).
  - Run with `BENCH_MAX_RUNS=50` and `OMP_PROC_BIND=spread`; enable nested OpenMP (`OMP_NESTED=TRUE OMP_MAX_ACTIVE_LEVELS=2`) for multi-instance and buffer_acquire.
  - For A/B: run both current tree and `/home/daniel-wiczew/build/OpenBLAS-develop`, saving CSVs as `benchmark/results_*_current*.csv` and `..._develop*.csv`; compare GB/s and GFLOPS across key sizes (32 KB–64 MB) and instances (4/8/12) with inner threads 1/2/4.
  - Keep all CSVs/binaries out of commits; only commit code or doc changes explicitly requested.

## Changelog

### Performance Optimization: OpenMP Buffer Allocation (2025-12-09)

**Optimizations Implemented:**
1.  **TLS Buffer Caching:** Modified `driver/others/blas_server_omp.c` to cache the per-thread BLAS working buffer using `static __thread void *tls_buffer`. This avoids the overhead of `blas_memory_alloc` (lock acquisition, table lookup) for the majority of calls.
2.  **False Sharing Mitigation:** Padded the `blas_buffer_inuse` array to 64 bytes (`blas_buffer_inuse_t`) to prevent cache line bouncing during slot acquisition in `exec_blas`.
3.  **Lazy Allocation:** Removed eager global buffer pre-allocation in `adjust_thread_buffers`, relying on the lazy thread-local allocation pattern to ensure NUMA locality.
4.  **Recursion Handling:** Implemented a fallback to standard allocation if the TLS buffer is already in use (`tls_in_use` flag).

**Results:**
-   **Multi-Instance Scalability:** In `benchmark/multi_instance_copy_bench.c` (8 instances, nested OpenMP), aggregate memcpy bandwidth improved from **~11.2 GB/s** to **~31.5 GB/s** (~2.8x speedup).
-   **Single-Thread Recovery:** Single-threaded `scopy` performance, previously degraded by allocation overhead, recovered from ~18 GB/s to **~26-31 GB/s** (matching or exceeding baseline).
-   **Verification:** Passed all 126 unit tests in `utest` covering BLAS level 1/2/3 and fork safety.