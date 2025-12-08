# What didn’t work (Dec 8, 2025)

Context: experiments aimed at improving multi‑instance OpenBLAS performance (8×2‑thread instances, USE_TLS=1, TARGET=NEHALEM, NUM_THREADS=2, NUM_PARALLEL=8, USE_OPENMP=1).

- Added a per‑size OpenMP team cap for Level‑1 “big vector” fast paths (env `OPENBLAS_L1_MAX_TEAM`) and spun up with `num_threads(team)` inside copy/axpy/dot/asum/nrm2/scal`. Result: regressions versus develop in multi‑instance copy. Bench `benchmark/results_multi_instance_copy_current_cap2.csv` shows 64 MB, 8 instances, inner=2 → ~35 GB/s vs develop ~45 GB/s (see `/home/daniel-wiczew/build/OpenBLAS-develop/benchmark/results_multi_instance_copy_develop.csv`). Reverted.
- Added backoff/pause/yield in `exec_blas` slot acquisition to reduce spin when `MAX_PARALLEL_NUMBER` is exceeded. No measurable improvement; risk of extra latency once slots free. Reverted.
- No NUMA-aware memcpy path or TLS buffer changes were committed; bandwidth bottleneck remains memory, not locking (buffer_acquire bench ~33 µs/call, unchanged).

Status: tree reset to commit `8bcfc506c` (only new `benchmark/multi_instance_copy_bench` exists). Keep these notes to avoid repeating the same regressions. Baseline build/tests pass with specified flags.
