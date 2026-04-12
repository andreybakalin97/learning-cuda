#include "vecadd.cuh"
#include "cuda_utils.cuh"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

class VecAddTest : public ::testing::Test {
  protected:
    void run_test(int n) {
        std::vector<float> h_A(n), h_B(n), h_C(n);

        // Fill with deterministic data
        for (int i = 0; i < n; i++) {
            h_A[i] = static_cast<float>(i) * 0.001f;
            h_B[i] = static_cast<float>(n - i) * 0.001f;
        }

        float* d_A = to_device(h_A.data(), n);
        float* d_B = to_device(h_B.data(), n);
        float* d_C = nullptr;
        CHECK_CUDA(cudaMalloc(&d_C, n * sizeof(float)));

        vecadd(d_A, d_B, d_C, n);
        CHECK_LAST_KERNEL();

        to_host(h_C.data(), d_C, n);

        for (int i = 0; i < n; i++) {
            float expected = h_A[i] + h_B[i];
            ASSERT_NEAR(h_C[i], expected, 1e-5f)
                << "Mismatch at index " << i;
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }
};

TEST_F(VecAddTest, Small)     { run_test(256); }
TEST_F(VecAddTest, Medium)    { run_test(100'000); }
TEST_F(VecAddTest, Large)     { run_test(10'000'000); }
TEST_F(VecAddTest, NonPow2)   { run_test(999'999); }
