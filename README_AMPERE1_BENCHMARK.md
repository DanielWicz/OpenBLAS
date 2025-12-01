# Benchmarking AmpereOne Optimization

This branch `ampereone-a1-optimization` contains optimizations for the AmpereOne processor, specifically targeting Bfloat16 GEMM (SBGEMM/BGEMM) performance and correctness.

## 1. Build

To build for AmpereOne (on an AmpereOne machine):

```bash
make TARGET=AMPERE1 USE_OPENMP=1 -j$(nproc)
```

If you are cross-compiling, ensure you have the correct toolchain and flags:
```bash
make TARGET=AMPERE1 HOSTCC=gcc CC=aarch64-linux-gnu-gcc FC=aarch64-linux-gnu-gfortran USE_OPENMP=1 -j$(nproc)
```

## 2. Verify Correctness

The previous implementation had a bug in the tail reduction of the SBGEMM kernel causing test failures.
Run the tests to verify the fix:

```bash
cd test
make test_sbgemm
./test_sbgemm
```
(Or simply `make` in the `test` directory to run all tests).

## 3. Benchmark

To measure performance of the new kernel:

```bash
cd benchmark
make sbgemm.goto
./sbgemm.goto
```

You can also benchmark standard SGEMM/DGEMM which have been tuned for AmpereOne's L2 cache size:
```bash
make sgemm.goto
./sgemm.goto
```

## 4. Tuning Parameters

The `param.h` file has been updated with specific blocking parameters for AmpereOne:
- `SBGEMM_DEFAULT_Q` increased to 2048 to better utilize the 2MB L2 cache.
- `SBGEMM_DEFAULT_P` set to 512 for better threading granularity on high-core-count systems.

You can further tune these in `param.h` if needed based on specific workload characteristics.
