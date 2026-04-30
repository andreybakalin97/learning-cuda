#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

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
        for (int i = 0; i < M; ++i) {
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
    matmul_kernel_one_col<<<(N + threadPerBlock - 1) / threadPerBlock, threadPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

constexpr int TILE_WIDTH = 16;

__global__
void matmul_kernel_tiled(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    __shared__ float block[TILE_WIDTH * TILE_WIDTH * 2];
    float* As = block;
    float* Bs = block + TILE_WIDTH * TILE_WIDTH;

    int tc = threadIdx.x, tr = threadIdx.y;
    int bc = blockIdx.x, br = blockIdx.y;

    int row = (br * blockDim.y) + tr;
    int col = (bc * blockDim.x) + tc;

    float sum = 0;
    int numBlocks = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int b = 0; b < numBlocks; ++b) {
        if (row < M && b * TILE_WIDTH + tc < K) {
            As[tr * TILE_WIDTH + tc] = A[row * K + b * TILE_WIDTH + tc];
        } else {
            As[tr * TILE_WIDTH + tc] = 0;
        }
        if (b * TILE_WIDTH + tr < K && col < N) {
            Bs[tr * TILE_WIDTH + tc] = B[(b * TILE_WIDTH + tr) * N + col];
        } else {
            Bs[tr * TILE_WIDTH + tc] = 0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[tr * TILE_WIDTH + k] * Bs[k * TILE_WIDTH + tc];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

inline void matmulTiled(const float* d_A, const float* d_B, float* d_C,
                          int M, int N, int K) {
    dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    matmul_kernel_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
}
