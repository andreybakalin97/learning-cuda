#include "matmul.cuh"
#include "cuda_utils.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

// CPU reference implementation
static void matmul_cpu(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

class MatmulTest : public ::testing::Test {
  protected:
    virtual void run_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) = 0;

    void run_test(int M, int N, int K) {
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
        std::vector<float> ref(M * N);

        for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(i % 7) * 0.1f;
        for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(i % 5) * 0.1f;

        matmul_cpu(h_A.data(), h_B.data(), ref.data(), M, N, K);

        float* d_A = to_device(h_A.data(), M * K);
        float* d_B = to_device(h_B.data(), K * N);
        float* d_C = nullptr;
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        run_kernel(d_A, d_B, d_C, M, N, K);
        CHECK_LAST_KERNEL();

        to_host(h_C.data(), d_C, M * N);

        for (int i = 0; i < M * N; i++) {
            ASSERT_NEAR(h_C[i], ref[i], 1e-3f)
                << "Mismatch at index " << i
                << " (row=" << i / N << ", col=" << i % N << ")";
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }
};

class MatmulTestOneElem : public MatmulTest {
protected:
    void run_kernel(const float* d_A, const float* d_B, float* d_C,
                              int M, int N, int K) override {
        matmulOneElem(d_A, d_B, d_C, M, N, K);
    }
};

class MatmulTestOneRow : public MatmulTest {
protected:
    void run_kernel(const float* d_A, const float* d_B, float* d_C,
                              int M, int N, int K) override {
        matmulOneRow(d_A, d_B, d_C, M, N, K);
    }
};

class MatmulTestOneCol : public MatmulTest {
protected:
    void run_kernel(const float* d_A, const float* d_B, float* d_C,
                              int M, int N, int K) override {
        matmulOneCol(d_A, d_B, d_C, M, N, K);
    }
};

TEST_F(MatmulTestOneElem, Square_Small)    { run_test(32, 32, 32); }
TEST_F(MatmulTestOneElem, Square_Medium)   { run_test(512, 512, 512); }
TEST_F(MatmulTestOneElem, Rectangular)     { run_test(128, 256, 64); }
TEST_F(MatmulTestOneElem, NonMultiple32)   { run_test(100, 100, 100); }

TEST_F(MatmulTestOneRow, Square_Small)    { run_test(32, 32, 32); }
TEST_F(MatmulTestOneRow, Square_Medium)   { run_test(512, 512, 512); }
TEST_F(MatmulTestOneRow, Rectangular)     { run_test(128, 256, 64); }
TEST_F(MatmulTestOneRow, NonMultiple32)   { run_test(100, 100, 100); }

TEST_F(MatmulTestOneCol, Square_Small)    { run_test(32, 32, 32); }
TEST_F(MatmulTestOneCol, Square_Medium)   { run_test(512, 512, 512); }
TEST_F(MatmulTestOneCol, Rectangular)     { run_test(128, 256, 64); }
TEST_F(MatmulTestOneCol, NonMultiple32)   { run_test(100, 100, 100); }
