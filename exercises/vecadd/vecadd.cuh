#pragma once

#include <cuda_runtime.h>

__global__
void vecadd_kernel(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

inline void vecadd(const float* d_A, const float* d_B, float* d_C, int n, int blockSize = 256) {
    vecadd_kernel<<<(n + blockSize - 1) / blockSize, blockSize>>>(d_A, d_B, d_C, n);
}
