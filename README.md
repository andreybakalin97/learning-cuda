# Learning CUDA

CUDA exercises from *Programming Massively Parallel Processors* (PMPP), with correctness tests (GoogleTest) and benchmarks against optimized libraries (Google Benchmark + Thrust/cuBLAS).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Run benchmarks

```bash
./build/exercises/bench_vecadd
```

## Adding a new exercise

1. Create `exercises/name/` with:
   - `name.cuh` — your kernel implementation
   - `test_name.cu` — correctness tests (GoogleTest)
   - `bench_name.cu` — benchmarks vs optimized library
2. Add `add_exercise(name)` in `exercises/CMakeLists.txt`

## Project structure

```
common/              Shared utilities (error checking, timing, device helpers)
exercises/
  vecadd/            Vector addition — kernel, test, benchmark
  ...
```
