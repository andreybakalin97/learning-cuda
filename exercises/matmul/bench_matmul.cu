#include "matmul.cuh"
#include "cuda_utils.cuh"

#include <benchmark/benchmark.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

// Benchmark our kernel
static void BM_Matmul_Custom(benchmark::State& state) {
    const int N = state.range(0);
    thrust::device_vector<float> d_A(N * N, 1.0f);
    thrust::device_vector<float> d_B(N * N, 1.0f);
    thrust::device_vector<float> d_C(N * N);

    for (auto _ : state) {
        matmul(thrust::raw_pointer_cast(d_A.data()),
               thrust::raw_pointer_cast(d_B.data()),
               thrust::raw_pointer_cast(d_C.data()), N, N, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 2*N^3 FLOPs for an N×N matrix multiply
    state.SetItemsProcessed(state.iterations() * 2LL * N * N * N);
}

// Benchmark cuBLAS (highly optimized library baseline)
static void BM_Matmul_cuBLAS(benchmark::State& state) {
    const int N = state.range(0);
    thrust::device_vector<float> d_A(N * N, 1.0f);
    thrust::device_vector<float> d_B(N * N, 1.0f);
    thrust::device_vector<float> d_C(N * N);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;

    for (auto _ : state) {
        // cuBLAS uses column-major order, so we compute C = B^T * A^T
        // which is equivalent to C = A * B in row-major
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    thrust::raw_pointer_cast(d_B.data()), N,
                    thrust::raw_pointer_cast(d_A.data()), N,
                    &beta,
                    thrust::raw_pointer_cast(d_C.data()), N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    state.SetItemsProcessed(state.iterations() * 2LL * N * N * N);

    cublasDestroy(handle);
}

#define SIZES Arg(256)->Arg(512)->Arg(1024)->Arg(2048)

BENCHMARK(BM_Matmul_Custom)->SIZES;
BENCHMARK(BM_Matmul_cuBLAS)->SIZES;
