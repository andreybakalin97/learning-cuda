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

// Declares test cases shared across all fixtures via X-macro.
// Each entry: (TestName, M, N, K)
#define MATMUL_TEST_CASES(F)                                                        \
    /* tile-aligned square sizes */                                                 \
    TEST_F(F, Square_Small)         { run_test(32,  32,  32);  }                    \
    TEST_F(F, Square_Medium)        { run_test(512, 512, 512); }                    \
    /* all dims tile-aligned, non-square */                                         \
    TEST_F(F, Rectangular)          { run_test(128, 256, 64);  }                    \
    TEST_F(F, TallSkinny)           { run_test(256, 16,  64);  }                    \
    TEST_F(F, WideFat)              { run_test(16,  256, 64);  }                    \
    /* single-element inner product (K=1): only one tile phase, partial tile */     \
    TEST_F(F, KEqualsOne)           { run_test(32,  32,  1);   }                    \
    /* K smaller than tile width: single partial tile along K */                    \
    TEST_F(F, KSmallerThanTile)     { run_test(32,  32,  8);   }                    \
    /* only K crosses a tile boundary (32 full + 1 leftover) */                     \
    TEST_F(F, KBoundary)            { run_test(32,  32,  33);  }                    \
    /* only M crosses a tile boundary */                                             \
    TEST_F(F, MBoundary)            { run_test(33,  32,  32);  }                    \
    /* only N crosses a tile boundary */                                             \
    TEST_F(F, NBoundary)            { run_test(32,  33,  32);  }                    \
    /* all three dims have remainder tiles, different remainders */                  \
    TEST_F(F, AllDimsBoundary)      { run_test(33,  35,  37);  }                    \
    /* dims just above tile size: 1 full + 1 element partial tile */                \
    TEST_F(F, JustAboveTile)        { run_test(17,  17,  17);  }                    \
    /* dims smaller than tile size: single partial tile in every direction */       \
    TEST_F(F, SmallerThanTile)      { run_test(7,   11,  5);   }                    \
    /* non-multiple-of-32, non-power-of-2 */                                        \
    TEST_F(F, NonMultiple32)        { run_test(100, 100, 100); }                    \
    /* large K to stress many tile phases */                                        \
    TEST_F(F, LargeK)               { run_test(32,  32,  512); }                    \
    /* 1×1×1 degenerate */                                                          \
    TEST_F(F, Tiny)                 { run_test(1,   1,   1);   }                    \
    /* single output row/col */                                                     \
    TEST_F(F, SingleRow)            { run_test(1,   64,  64);  }                    \
    TEST_F(F, SingleCol)            { run_test(64,  1,   64);  }

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

class MatmulTestTiled : public MatmulTest {
protected:
    void run_kernel(const float* d_A, const float* d_B, float* d_C,
                              int M, int N, int K) override {
        matmulTiled(d_A, d_B, d_C, M, N, K);
    }
};

MATMUL_TEST_CASES(MatmulTestOneElem)
MATMUL_TEST_CASES(MatmulTestOneRow)
MATMUL_TEST_CASES(MatmulTestOneCol)
MATMUL_TEST_CASES(MatmulTestTiled)
