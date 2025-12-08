# Agent Notes

- Always read `not_works.md` before attempting performance changes; it lists recent failed experiments and benchmarks to avoid repeating regressions.
- Use the existing `benchmark/multi_instance_copy_bench` and `benchmark_optimization/run_bench.sh` to validate multi-instance and Level-1 changes under `USE_TLS=1`.
- Keep untracked benchmark outputs out of commits; only add new artifacts that are explicitly requested.
