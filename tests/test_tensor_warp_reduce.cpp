/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_backend_test.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

void launch_consecutive_tensor_warp_reductions(float* output, int blocks);
template <typename T>
void launch_tensor_block_reductions(const T* input, T* output, int blocks, int block_size);
template <typename T>
void launch_tensor_warp_reductions(const T* input, T* output, int threads);

namespace {

    template <typename T>
    std::vector<T> reduction_inputs(const size_t count) {
        std::vector<T> values(count);
        for (size_t i = 0; i < count; ++i) {
            // Integer data has both signs so the result does not depend on the padding value.
            const int pattern = static_cast<int>(i * 7 % 97) - 48;
            values[i] = std::is_integral_v<T> ? static_cast<T>(pattern) : static_cast<T>(pattern) / static_cast<T>(8);
        }
        return values;
    }

    template <typename T>
    std::vector<T> run_reduction(const std::vector<T>& input, const size_t outputs, const int blocks, const int threads) {
        T* device_input = nullptr;
        T* device_output = nullptr;
        EXPECT_EQ(cudaMalloc(&device_input, input.size() * sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&device_output, outputs * sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(device_input, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        if (blocks > 0)
            launch_tensor_block_reductions(device_input, device_output, blocks, threads);
        else
            launch_tensor_warp_reductions(device_input, device_output, threads);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<T> results(outputs);
        EXPECT_EQ(cudaMemcpy(results.data(), device_output, outputs * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaFree(device_input), cudaSuccess);
        EXPECT_EQ(cudaFree(device_output), cudaSuccess);
        return results;
    }

    template <typename T>
    void expect_block_reductions(const int block_size) {
        constexpr int kBlocks = 3;
        const auto input = reduction_inputs<T>(static_cast<size_t>(block_size) * kBlocks);
        const auto results = run_reduction(input, kBlocks * 4, kBlocks, block_size);
        for (int block = 0; block < kBlocks; ++block) {
            const auto first = input.begin() + static_cast<std::ptrdiff_t>(block) * block_size;
            double sum = 0.0;
            for (auto it = first; it != first + block_size; ++it)
                sum += static_cast<double>(*it);
            double prod = 1.0;
            for (int t = 0; t < block_size; ++t)
                prod *= t % 37 == 0 ? 2.0 : 1.0;
            EXPECT_NEAR(static_cast<double>(results[block * 4 + 0]), sum, 1e-4) << "block " << block;
            EXPECT_EQ(results[block * 4 + 1], *std::max_element(first, first + block_size)) << "block " << block;
            EXPECT_EQ(results[block * 4 + 2], *std::min_element(first, first + block_size)) << "block " << block;
            EXPECT_EQ(static_cast<double>(results[block * 4 + 3]), prod) << "block " << block;
        }
    }

    // Warp reductions combine fixed 32-lane groups.
    template <typename T>
    void expect_warp_reductions() {
        constexpr int kThreads = 128;
        const auto input = reduction_inputs<T>(kThreads);
        const auto results = run_reduction(input, kThreads * 4, 0, kThreads);
        for (int lane = 0; lane < kThreads; ++lane) {
            const auto first = input.begin() + lane / 32 * 32;
            double sum = 0.0, prod = 1.0;
            for (int u = lane / 32 * 32; u < lane / 32 * 32 + 32; ++u) {
                sum += static_cast<double>(input[u]);
                prod *= u % 7 == 0 ? 2.0 : 1.0;
            }
            EXPECT_NEAR(static_cast<double>(results[lane * 4 + 0]), sum, 1e-4) << "lane " << lane;
            EXPECT_EQ(results[lane * 4 + 1], *std::max_element(first, first + 32)) << "lane " << lane;
            EXPECT_EQ(results[lane * 4 + 2], *std::min_element(first, first + 32)) << "lane " << lane;
            EXPECT_EQ(static_cast<double>(results[lane * 4 + 3]), prod) << "lane " << lane;
        }
    }

} // namespace

class TensorWarpReduceTest : public lfs::test::CudaBackendTest {};

TEST_F(TensorWarpReduceTest, ConsecutiveBlockReductionsKeepIndependentResults) {
    constexpr int kBlocks = 128;
    constexpr int kValuesPerBlock = 8;
    float* device_output = nullptr;
    ASSERT_EQ(cudaMalloc(&device_output, kBlocks * kValuesPerBlock * sizeof(float)), cudaSuccess);

    launch_consecutive_tensor_warp_reductions(device_output, kBlocks);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> results(kBlocks * kValuesPerBlock);
    ASSERT_EQ(cudaMemcpy(results.data(), device_output, results.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(device_output), cudaSuccess);

    for (int block = 0; block < kBlocks; ++block) {
        const auto* values = results.data() + block * kValuesPerBlock;
        EXPECT_FLOAT_EQ(values[0], 32896.0f);
        EXPECT_FLOAT_EQ(values[1], -65792.0f);
        EXPECT_FLOAT_EQ(values[2], 256.0f);
        EXPECT_FLOAT_EQ(values[3], -1.0f);
        EXPECT_FLOAT_EQ(values[4], 1.0f);
        EXPECT_FLOAT_EQ(values[5], -256.0f);
        EXPECT_FLOAT_EQ(values[6], 1.0f);
        EXPECT_NEAR(values[7], std::pow(1.0001f, 256.0f), 1e-5f);
    }
}

TEST_F(TensorWarpReduceTest, BlockReductionsMatchHostForEveryWarpMultiple) {
    for (int block_size = 32; block_size <= 1024; block_size += 32) {
        SCOPED_TRACE(block_size);
        expect_block_reductions<float>(block_size);
        expect_block_reductions<double>(block_size);
        expect_block_reductions<int>(block_size);
    }
}

TEST_F(TensorWarpReduceTest, WarpReductionsCombineThirtyTwoLaneGroups) {
    expect_warp_reductions<float>();
    expect_warp_reductions<double>();
    expect_warp_reductions<int>();
}
