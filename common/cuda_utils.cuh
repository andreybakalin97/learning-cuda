#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Check CUDA call and abort on error
#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Synchronize and check for kernel launch errors
#define CHECK_LAST_KERNEL()                                                    \
    do {                                                                        \
        CHECK_CUDA(cudaGetLastError());                                         \
        CHECK_CUDA(cudaDeviceSynchronize());                                    \
    } while (0)

// Simple RAII timer using CUDA events (prints elapsed ms on destruction)
struct CudaTimer {
    cudaEvent_t start, stop;
    const char* label;

    CudaTimer(const char* label = "kernel") : label(label) {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
    }

    float stop_and_get_ms() {
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }

    ~CudaTimer() {
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
};

// Allocate device memory and copy from host
template <typename T>
T* to_device(const T* host_data, size_t count) {
    T* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, count * sizeof(T)));
    CHECK_CUDA(
        cudaMemcpy(d_ptr, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    return d_ptr;
}

// Copy device memory back to host
template <typename T>
void to_host(T* host_dst, const T* d_src, size_t count) {
    CHECK_CUDA(
        cudaMemcpy(host_dst, d_src, count * sizeof(T), cudaMemcpyDeviceToHost));
}
