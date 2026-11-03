/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Runs the shared CUDA/HIP warp-group kernels on the selected GPU. Block sizes
// cover a partial wave (32 threads on a 64-lane GPU) and odd numbers of 32-lane
// groups, which is where a 32-lane assumption meets a 64-lane wavefront.

#include "core/device/block_ballot.cuh"
#include "core/tensor/backend/cuda/kernels/broadcast_channel_coalesced.cuh"
#include "core/tensor/backend/cuda/kernels/warp_reduce.cuh"
#include "core/tensor/backend/cuda/kernels/warp_segment_reduce.cuh"
#include "device_kernel_checks.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {
    using namespace lfs::hip_tests;
    namespace warp_ops = lfs::core::warp_ops;
    namespace tensor_ops = lfs::core::tensor_ops;

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

    template <typename T>
    __global__ void warp_reductions(const T* input, T* output) {
        const T x = input[threadIdx.x];
        output[threadIdx.x * 4 + 0] = warp_ops::warp_reduce_sum(x);
        output[threadIdx.x * 4 + 1] = warp_ops::warp_reduce_max(x);
        output[threadIdx.x * 4 + 2] = warp_ops::warp_reduce_min(x);
        output[threadIdx.x * 4 + 3] = warp_ops::warp_reduce_prod(static_cast<T>(threadIdx.x % 7 == 0 ? 2 : 1));
    }

    __global__ void ballot64(const uint64_t* patterns, uint64_t* output) {
        const bool bit = ((patterns[blockIdx.x] >> threadIdx.x) & 1u) != 0;
        const uint64_t mask = lfs::core::device::block_ballot64(bit);
        if (threadIdx.x == 63)
            output[blockIdx.x] = mask;
    }

    struct ScaledAdd {
        __device__ float operator()(const float a, const float b) const { return a + 2.0f * b; }
    };

    unsigned rng = 12345u;
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

    // Warp reductions combine 32-lane groups on every GPU.
    template <typename T>
    void check_warp_reductions() {
        constexpr int block = 128;
        std::vector<T> host(block);
        for (int i = 0; i < block; ++i)
            host[i] = std::is_integral_v<T> ? static_cast<T>(i * 3 - 100) : static_cast<T>(random_value() * 8.0f);
        DeviceBuffer<T> input(block);
        DeviceBuffer<T> output(block * 4);
        input.upload(host);
        warp_reductions<T><<<1, block>>>(input.data, output.data);
        require(hipGetLastError(), "warp_reductions");
        const auto result = output.download(block * 4);
        for (int t = 0; t < block; ++t) {
            const int first = t / 32 * 32;
            double sum = 0.0, max = -INFINITY, min = INFINITY, prod = 1.0;
            for (int u = first; u < first + 32; ++u) {
                sum += static_cast<double>(host[u]);
                max = std::fmax(max, static_cast<double>(host[u]));
                min = std::fmin(min, static_cast<double>(host[u]));
                prod *= u % 7 == 0 ? 2.0 : 1.0;
            }
            const std::string where = std::string(type_name<T>()) + " warp lane " + std::to_string(t);
            expect_near(static_cast<double>(result[t * 4 + 0]), sum, 1e-5, where + " sum");
            expect_near(static_cast<double>(result[t * 4 + 1]), max, 0.0, where + " max");
            expect_near(static_cast<double>(result[t * 4 + 2]), min, 0.0, where + " min");
            expect_near(static_cast<double>(result[t * 4 + 3]), prod, 1e-6, where + " prod");
        }
    }

    // Launched exactly as tensor_warp_reduce.cu launches the medium-segment kernels.
    void check_medium_segments(const size_t segments, const size_t segment_size, const bool misaligned) {
        const size_t offset = misaligned ? 1 : 0;
        std::vector<float> host(segments * segment_size);
        for (auto& v : host)
            v = 0.5f + random_value() * 0.25f;
        DeviceBuffer<float> input(host.size() + 1);
        DeviceBuffer<float> output(segments);
        input.upload(host, offset);
        constexpr int block_size = 256;
        constexpr int warps_per_block = block_size / 32;
        const int grid = static_cast<int>((segments + warps_per_block - 1) / warps_per_block);
        static const char* names[] = {"sum", "mean", "max", "min", "prod"};
        for (int op = 0; op < 5; ++op) {
            const float* in = input.data + offset;
            switch (op) {
            case 0: tensor_ops::warp_medium_segment_reduce_sum_kernel<<<grid, block_size>>>(in, output.data, segments, segment_size); break;
            case 1: tensor_ops::warp_medium_segment_reduce_mean_kernel<<<grid, block_size>>>(in, output.data, segments, segment_size); break;
            case 2: tensor_ops::warp_medium_segment_reduce_max_kernel<<<grid, block_size>>>(in, output.data, segments, segment_size); break;
            case 3: tensor_ops::warp_medium_segment_reduce_min_kernel<<<grid, block_size>>>(in, output.data, segments, segment_size); break;
            default: tensor_ops::warp_medium_segment_reduce_prod_kernel<<<grid, block_size>>>(in, output.data, segments, segment_size); break;
            }
            require(hipGetLastError(), "medium segment reduction");
            const auto result = output.download(segments);
            for (size_t s = 0; s < segments; ++s) {
                double want = op == 2 ? -INFINITY : op == 3 ? INFINITY
                                                : op == 4   ? 1.0
                                                            : 0.0;
                for (size_t i = 0; i < segment_size; ++i) {
                    const double v = host[s * segment_size + i];
                    want = op <= 1 ? want + v : op == 2 ? std::fmax(want, v)
                                            : op == 3   ? std::fmin(want, v)
                                                        : want * v;
                }
                if (op == 1)
                    want /= static_cast<double>(segment_size);
                expect_near(result[s], want, op == 4 ? 1e-4 : 1e-5,
                            std::string("medium ") + names[op] + " segments " + std::to_string(segments) + " size " +
                                std::to_string(segment_size) + (misaligned ? " misaligned" : "") + " #" + std::to_string(s));
            }
        }
    }

    void check_channel_broadcast(const size_t pixels, const size_t channels) {
        std::vector<float> pixel_data(pixels * channels), channel_data(channels);
        for (auto& v : pixel_data)
            v = random_value();
        for (auto& v : channel_data)
            v = random_value();
        DeviceBuffer<float> a(pixel_data.size());
        DeviceBuffer<float> b(channels);
        DeviceBuffer<float> c(pixel_data.size());
        a.upload(pixel_data);
        b.upload(channel_data);
        constexpr int block_size = 256;
        constexpr int warps_per_block = block_size / 32;
        const int grid = static_cast<int>((pixels + warps_per_block - 1) / warps_per_block);
        tensor_ops::broadcast_channel3d_coalesced_kernel_float<<<grid, block_size, channels * sizeof(float)>>>(
            a.data, b.data, c.data, 1, pixels, channels, false, ScaledAdd{});
        require(hipGetLastError(), "channel broadcast");
        const auto result = c.download(pixel_data.size());
        size_t wrong = 0;
        for (size_t p = 0; p < pixels; ++p)
            for (size_t ch = 0; ch < channels; ++ch)
                wrong += result[p * channels + ch] != pixel_data[p * channels + ch] + 2.0f * channel_data[ch];
        expect(wrong == 0, "channel broadcast pixels " + std::to_string(pixels) + " channels " + std::to_string(channels) +
                               ": " + std::to_string(wrong) + " wrong values");
    }

    // The collision voxelizer stores one bit per thread of a 64-thread block;
    // the upper patterns are carried by lanes 32-63 of a 64-lane wavefront.
    void check_block_ballot64() {
        const std::vector<uint64_t> patterns = {
            0, ~uint64_t(0), 0x00000000ffffffffull, 0xffffffff00000000ull, uint64_t(1) << 32,
            uint64_t(1) << 63, 1, 0x5555555555555555ull, 0x0123456789abcdefull, 0x8000000180000001ull};
        DeviceBuffer<uint64_t> input(patterns.size());
        DeviceBuffer<uint64_t> output(patterns.size());
        input.upload(patterns);
        ballot64<<<static_cast<unsigned>(patterns.size()), 64>>>(input.data, output.data);
        require(hipGetLastError(), "ballot64");
        const auto result = output.download(patterns.size());
        for (size_t i = 0; i < patterns.size(); ++i) {
            char text[96];
            std::snprintf(text, sizeof(text), "block_ballot64 pattern %016llx: got %016llx",
                          static_cast<unsigned long long>(patterns[i]), static_cast<unsigned long long>(result[i]));
            expect(result[i] == patterns[i], text);
        }
    }

} // namespace

int main(int argc, char** argv) {
    select_device(argc, argv);
    for (int block = 32; block <= 1024; block += 32) {
        check_block_reductions<float>(block);
        check_block_reductions<double>(block);
        check_block_reductions<int>(block);
    }
    check_warp_reductions<float>();
    check_warp_reductions<double>();
    check_warp_reductions<int>();
    for (const size_t segments : {1, 3, 7, 8, 9, 33, 65, 1001})
        for (const size_t size : {31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 1023, 1024, 2048}) {
            check_medium_segments(segments, size, false);
            check_medium_segments(segments, size, true);
        }
    for (const size_t pixels : {1, 7, 8, 9, 33, 65})
        for (const size_t channels : {3073, 3104, 3135, 4096})
            check_channel_broadcast(pixels, channels);
    check_block_ballot64();
    return finish();
}
