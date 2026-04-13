#pragma once

#include <cuda_runtime.h>

__global__
void matmul_kernel_one_elem(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[N * i + col];
        }
        C[row * N + col] = sum;
    }
}

inline void matmulOneElem(const float* d_A, const float* d_B, float* d_C,
                          int M, int N, int K) {
    int threadPerDim = 16;
    dim3 blockSize(threadPerDim, threadPerDim);
    dim3 gridSize((N + threadPerDim - 1) / threadPerDim, ((M + threadPerDim - 1) / threadPerDim));
    matmul_kernel_one_elem<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
}

__global__
void matmul_kernel_one_row(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        for (int i = 0; i < N; ++i) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + i];
            }
            C[row * N + i] = sum;
        }
    }
}

inline void matmulOneRow(const float* d_A, const float* d_B, float* d_C,
                          int M, int N, int K) {
    int threadPerBlock = 16;
    matmul_kernel_one_row<<<(M + threadPerBlock - 1) / threadPerBlock, threadPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

__global__
void matmul_kernel_one_col(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int i = 0; i < N; ++i) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + col];
            }
            C[i * N + col] = sum;
        }
    }
}

inline void matmulOneCol(const float* d_A, const float* d_B, float* d_C,
                          int M, int N, int K) {
    int threadPerBlock = 16;
    matmul_kernel_one_row<<<(N + threadPerBlock - 1) / threadPerBlock, threadPerBlock>>>(d_A, d_B, d_C, M, N, K);
}
