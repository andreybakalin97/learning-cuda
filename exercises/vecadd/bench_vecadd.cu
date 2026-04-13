#include "vecadd.cuh"
#include "cuda_utils.cuh"

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

// Benchmark our kernel
static void BM_VecAdd_Custom(benchmark::State& state) {
    const int n = state.range(0);
    const int blockSize = state.range(1);
    thrust::device_vector<float> d_A(n, 1.0f);
    thrust::device_vector<float> d_B(n, 2.0f);
    thrust::device_vector<float> d_C(n);

    for (auto _ : state) {
        vecadd(thrust::raw_pointer_cast(d_A.data()),
               thrust::raw_pointer_cast(d_B.data()),
               thrust::raw_pointer_cast(d_C.data()), n, blockSize);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Report throughput: 3 arrays * n floats * 4 bytes each
    state.SetBytesProcessed(state.iterations() * 3L * n * sizeof(float));
}

// Benchmark Thrust (optimized library baseline)
static void BM_VecAdd_Thrust(benchmark::State& state) {
    const int n = state.range(0);
    thrust::device_vector<float> d_A(n, 1.0f);
    thrust::device_vector<float> d_B(n, 2.0f);
    thrust::device_vector<float> d_C(n);

    for (auto _ : state) {
        thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(),
                          thrust::plus<float>());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    state.SetBytesProcessed(state.iterations() * 3L * n * sizeof(float));
}

BENCHMARK(BM_VecAdd_Custom)->ArgsProduct({
    {1 << 16, 1 << 20, 1 << 24},
    {1, 32, 64, 128, 256}
});
BENCHMARK(BM_VecAdd_Thrust)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 24);
