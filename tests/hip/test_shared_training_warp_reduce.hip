/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Runs the trainer's warp_reduce.cuh block reductions on the selected GPU. It is
// a separate executable because the trainer and Tensor headers define the same
// functions in the same namespace.

#include "device_kernel_checks.hpp"
#include "training/kernels/warp_reduce.cuh"

#include <algorithm>
#include <type_traits>

namespace {
    using namespace lfs::hip_tests;
    namespace warp_ops = lfs::core::warp_ops;

    template <typename T>
    __global__ void block_reductions(const T* input, T* output) {
        const T x = input[blockIdx.x * blockDim.x + threadIdx.x];
        const T sum = warp_ops::block_reduce_sum(x);
        const T max = warp_ops::block_reduce_max(x);
        const T min = warp_ops::block_reduce_min(x);
        const T prod = warp_ops::block_reduce_prod(static_cast<T>(threadIdx.x % 37 == 0 ? 2 : 1));
        if (threadIdx.x == 0) {
            output[blockIdx.x * 4 + 0] = sum;
            output[blockIdx.x * 4 + 1] = max;
            output[blockIdx.x * 4 + 2] = min;
            output[blockIdx.x * 4 + 3] = prod;
        }
    }

    __global__ void fused_minimums(const float* input, float* output) {
        const float x = input[blockIdx.x * blockDim.x + threadIdx.x];
        const float4 m4 = warp_ops::block_reduce_min4(make_float4(x, -x, x + 1.0f, -x - 1.0f));
        const float2 m2 = warp_ops::block_reduce_min2(make_float2(2.0f * x, -2.0f * x));
        if (threadIdx.x == 0) {
            float* out = output + blockIdx.x * 6;
            out[0] = m4.x;
            out[1] = m4.y;
            out[2] = m4.z;
            out[3] = m4.w;
            out[4] = m2.x;
            out[5] = m2.y;
        }
    }

    unsigned rng = 54321u;
    float random_value() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<float>((rng >> 8) & 0xffff) / 65536.0f - 0.5f;
    }

    template <typename T>
    const char* type_name() {
        if constexpr (std::is_same_v<T, float>)
            return "float";
        else if constexpr (std::is_same_v<T, double>)
            return "double";
        else
            return "int";
    }

    // Integer data has both signs so the result does not depend on the padding value.
    template <typename T>
    void check_block_reductions(const int block) {
        constexpr int blocks = 3;
        std::vector<T> host(static_cast<size_t>(block) * blocks);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = std::is_integral_v<T> ? static_cast<T>(static_cast<int>(i * 7 % 97) - 48)
                                            : static_cast<T>(random_value() * 8.0f);
        DeviceBuffer<T> input(host.size());
        DeviceBuffer<T> output(blocks * 4);
        input.upload(host);
        block_reductions<T><<<blocks, block>>>(input.data, output.data);
        require(hipGetLastError(), "block_reductions");
        const auto result = output.download(blocks * 4);
        for (int b = 0; b < blocks; ++b) {
            double sum = 0.0, max = -INFINITY, min = INFINITY, prod = 1.0;
            for (int t = 0; t < block; ++t) {
                const double v = static_cast<double>(host[static_cast<size_t>(b) * block + t]);
                sum += v;
                max = std::fmax(max, v);
                min = std::fmin(min, v);
                prod *= t % 37 == 0 ? 2.0 : 1.0;
            }
            const std::string where = std::string(type_name<T>()) + " block " + std::to_string(block) + " #" + std::to_string(b);
            expect_near(static_cast<double>(result[b * 4 + 0]), sum, 1e-5, where + " sum");
            expect_near(static_cast<double>(result[b * 4 + 1]), max, 0.0, where + " max");
            expect_near(static_cast<double>(result[b * 4 + 2]), min, 0.0, where + " min");
            expect_near(static_cast<double>(result[b * 4 + 3]), prod, 1e-6, where + " prod");
        }
    }

    // Integer max of negative values and min of positive values: unused
    // partials must not contribute 0.
    void check_one_signed_integer_reductions(const int block) {
        for (const int sign : {-1, 1}) {
            std::vector<int> host(block);
            for (int i = 0; i < block; ++i)
                host[i] = sign * (1000 + i * 13 % 500);
            DeviceBuffer<int> input(host.size());
            DeviceBuffer<int> output(4);
            input.upload(host);
            block_reductions<int><<<1, block>>>(input.data, output.data);
            require(hipGetLastError(), "block_reductions");
            const auto result = output.download(4);
            int max = host[0], min = host[0];
            for (const int v : host) {
                max = std::max(max, v);
                min = std::min(min, v);
            }
            const std::string where = (sign < 0 ? "negative" : "positive") + std::string(" int block ") + std::to_string(block);
            expect(result[1] == max, where + " max: got " + std::to_string(result[1]) + ", want " + std::to_string(max));
            expect(result[2] == min, where + " min: got " + std::to_string(result[2]) + ", want " + std::to_string(min));
        }
    }

    void check_fused_minimums(const int block) {
        constexpr int blocks = 2;
        std::vector<float> host(static_cast<size_t>(block) * blocks);
        for (auto& v : host)
            v = random_value() * 8.0f;
        DeviceBuffer<float> input(host.size());
        DeviceBuffer<float> output(blocks * 6);
        input.upload(host);
        fused_minimums<<<blocks, block>>>(input.data, output.data);
        require(hipGetLastError(), "fused_minimums");
        const auto result = output.download(blocks * 6);
        for (int b = 0; b < blocks; ++b) {
            double max = -INFINITY, min = INFINITY;
            for (int t = 0; t < block; ++t) {
                max = std::fmax(max, host[static_cast<size_t>(b) * block + t]);
                min = std::fmin(min, host[static_cast<size_t>(b) * block + t]);
            }
            const std::string where = "block " + std::to_string(block) + " #" + std::to_string(b);
            expect_near(result[b * 6 + 0], min, 0.0, where + " min4.x");
            expect_near(result[b * 6 + 1], -max, 0.0, where + " min4.y");
            expect_near(result[b * 6 + 2], min + 1.0, 1e-6, where + " min4.z");
            expect_near(result[b * 6 + 3], -max - 1.0, 1e-6, where + " min4.w");
            expect_near(result[b * 6 + 4], 2.0 * min, 0.0, where + " min2.x");
            expect_near(result[b * 6 + 5], -2.0 * max, 0.0, where + " min2.y");
        }
    }

} // namespace

int main(int argc, char** argv) {
    select_device(argc, argv);
    for (int block = 32; block <= 1024; block += 32) {
        check_block_reductions<float>(block);
        check_block_reductions<double>(block);
        check_block_reductions<int>(block);
        check_one_signed_integer_reductions(block);
        check_fused_minimums(block);
    }
    return finish();
}
