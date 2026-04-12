#pragma once

#include <cuda_runtime.h>

__global__ void vecadd_kernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

inline void vecadd(const float* d_A, const float* d_B, float* d_C, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vecadd_kernel << <grid_size, block_size >> > (d_A, d_B, d_C, n);
}
