#pragma once

#include <cuda_runtime.h>

__global__
void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    // TODO: implement matrix multiplication kernel
}

inline void matmul(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K) {
    // TODO: choose grid/block dimensions and launch matmul_kernel
}
