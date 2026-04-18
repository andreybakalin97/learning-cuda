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

__global__
void matmul_kernel_tiled(const float* A, const float* B, float* C,
                         int M, int N, int K, int tileWidth) {
    extern __shared__ char block[];
    float* As = (float*)block;
    float* Bs = (float*)(block + (tileWidth * tileWidth * sizeof(float)) / 2);

    int tc = threadIdx.x, tr = threadIdx.y;
    int bc = blockIdx.x, br = blockIdx.y;

    int row = (br * blockDim.y) + tr;
    int col = (bc * blockDim.x) + tc;

    float sum = 0;
    int numBlocks = (K + tileWidth - 1) / tileWidth;
    for (int b = 0; b < numBlocks; ++b) {
        if (row < M && b * tileWidth + tc < K) {
            As[tr * tileWidth + tc] = A[row * K + b * tileWidth + tc];
        } else {
            As[tr * tileWidth + tc] = 0;
        }
        if (b * tileWidth + tr < K && col < N) {
            Bs[tr * tileWidth + tc] = B[(b * tileWidth + tr) * N + col];
        } else {
            Bs[tr * tileWidth + tc] = 0;
        }
        __syncthreads();
        if (row < M && col < N) {
            for (int k = 0; k < tileWidth; ++k) {
                sum += As[tr * tileWidth + k] * Bs[k * tileWidth + tc];
            }
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#include <stdio.h>

inline void matmulTiled(const float* d_A, const float* d_B, float* d_C,
                          int M, int N, int K) {
    int tileWidth = 16, sharedSize = tileWidth * tileWidth * sizeof(float);
    dim3 blockSize((N + tileWidth - 1) / tileWidth, (M + tileWidth - 1) / tileWidth);
    dim3 threadSize(tileWidth, tileWidth);
    matmul_kernel_tiled<<<blockSize, threadSize, sharedSize>>>(d_A, d_B, d_C, M, N, K, tileWidth);
}
