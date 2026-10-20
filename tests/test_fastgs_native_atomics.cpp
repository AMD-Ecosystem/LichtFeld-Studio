/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

cudaError_t launch_fastgs_atomic_probe(float* sums, int count, float value,
                                       bool alternate_sign, cudaStream_t stream);
cudaError_t launch_fastgs_blend_backward_probe(
    const uint2* tile_instance_ranges, const unsigned int* instance_primitive_indices,
    const float2* primitive_mean2d, const float4* primitive_conic_opacity,
    const float3* primitive_color, const float* grad_image, const float* grad_alpha_map,
    const float* image, const float* alpha_map, const unsigned int* tile_n_contributions,
    const float* tile_final_transmittance, float2* safe_grad_mean2d,
    float* safe_grad_conic, float* safe_grad_opacity, float3* safe_grad_color,
    float2* native_grad_mean2d, float* native_grad_conic, float* native_grad_opacity,
    float3* native_grad_color, unsigned int n_instances, unsigned int n_primitives,
    unsigned int width, unsigned int height);

namespace {
    using lfs::core::RasterizerMemoryArena;

    class ArenaFrame {
    public:
        explicit ArenaFrame(RasterizerMemoryArena& arena)
            : arena_(arena), id_(arena.begin_frame()) {}
        ~ArenaFrame() { arena_.end_frame(id_); }
        auto allocator() { return arena_.get_allocator(id_); }

    private:
        RasterizerMemoryArena& arena_;
        uint64_t id_;
    };

    class FastGSNativeAtomics : public ::testing::Test {
    protected:
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
                GTEST_SKIP() << "GPU unavailable";
            }
        }
    };

    template <typename T>
    T* arena_array(std::function<char*(size_t)>& allocate, size_t count) {
        return reinterpret_cast<T*>(allocate(count * sizeof(T)));
    }

    template <typename T>
    bool upload(T* destination, const std::vector<T>& source) {
        return cudaMemcpy(destination, source.data(), source.size() * sizeof(T),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    }

    TEST_F(FastGSNativeAtomics, ArenaOwnershipRejectsInvalidAndExternalRanges) {
        RasterizerMemoryArena arena;
        ArenaFrame frame(arena);
        auto allocate = frame.allocator();
        auto* memory = allocate(4096);
        ASSERT_NE(memory, nullptr);
        EXPECT_TRUE(arena.owns_device_allocation(memory, 4096));
        EXPECT_TRUE(arena.owns_device_allocation(memory + 128, sizeof(float)));
        EXPECT_FALSE(arena.owns_device_allocation(nullptr, sizeof(float)));
        EXPECT_FALSE(arena.owns_device_allocation(memory, 0));
        EXPECT_FALSE(arena.owns_device_allocation(memory, std::numeric_limits<size_t>::max()));
        const auto capacity = arena.get_statistics().capacity;
        const auto outside = reinterpret_cast<const void*>(
            reinterpret_cast<std::uintptr_t>(memory) + capacity);
        EXPECT_FALSE(arena.owns_device_allocation(outside, 1));

        void* external = nullptr;
        ASSERT_EQ(cudaMalloc(&external, 4096), cudaSuccess);
        const std::shared_ptr<void> owner(external, [](void* ptr) { cudaFree(ptr); });
        RasterizerMemoryArena external_arena;
        int device = -1;
        ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
        RasterizerMemoryArena::ExternalBacking backing{
            .device_ptr = external,
            .size = 4096,
            .device = device,
            .owner = owner,
            .label = "fastgs-native-atomic-test.external"};
        ASSERT_TRUE(external_arena.install_external_backing(std::move(backing)));
        ArenaFrame external_frame(external_arena);
        auto external_allocate = external_frame.allocator();
        auto* imported = external_allocate(256);
        ASSERT_NE(imported, nullptr);
        EXPECT_FALSE(external_arena.owns_device_allocation(imported, sizeof(float)));
    }

    TEST_F(FastGSNativeAtomics, NativeAndSafeAddsMatchOnArenaStorage) {
        constexpr int count = 8192;
        RasterizerMemoryArena arena;
        ArenaFrame frame(arena);
        auto allocate = frame.allocator();
        auto* sums = reinterpret_cast<float*>(allocate(2 * sizeof(float)));
        ASSERT_NE(sums, nullptr);
        ASSERT_TRUE(arena.owns_device_allocation(sums, 2 * sizeof(float)));

        for (const float value : {0.125f, -0.125f, 1.0e-30f, -1.0e-30f}) {
            for (const bool alternate : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "value=" << value << " alternate=" << alternate);
                ASSERT_EQ(cudaMemset(sums, 0, 2 * sizeof(float)), cudaSuccess);
                ASSERT_EQ(launch_fastgs_atomic_probe(sums, count, value, alternate, nullptr), cudaSuccess);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                std::array<float, 2> actual{};
                ASSERT_EQ(cudaMemcpy(actual.data(), sums, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
                const float expected = alternate ? 0.0f : count * value;
                const float tolerance = std::max(std::fabs(count * value) * 3.0e-4f, 1.0e-35f);
                EXPECT_TRUE(std::isfinite(actual[0]));
                EXPECT_TRUE(std::isfinite(actual[1]));
                EXPECT_NEAR(actual[0], expected, tolerance);
                EXPECT_NEAR(actual[1], expected, tolerance);
                EXPECT_NEAR(actual[1], actual[0], tolerance);
            }
        }
    }

    TEST_F(FastGSNativeAtomics, BlendBackwardSafeAndNativeKernelsMatchWithOverlappingInstances) {
        constexpr unsigned int width = 16;
        constexpr unsigned int height = 16;
        constexpr size_t pixels = width * height;
        RasterizerMemoryArena arena;
        ArenaFrame frame(arena);
        auto allocate = frame.allocator();

        auto* tile_instance_ranges = arena_array<uint2>(allocate, 1);
        auto* instance_primitive_indices = arena_array<unsigned int>(allocate, 2);
        auto* primitive_mean2d = arena_array<float2>(allocate, 1);
        auto* primitive_conic_opacity = arena_array<float4>(allocate, 1);
        auto* primitive_color = arena_array<float3>(allocate, 1);
        auto* grad_image = arena_array<float>(allocate, pixels * 3);
        auto* grad_alpha = arena_array<float>(allocate, pixels);
        auto* image = arena_array<float>(allocate, pixels * 3);
        auto* alpha_map = arena_array<float>(allocate, pixels);
        auto* contributions = arena_array<unsigned int>(allocate, pixels);
        auto* transmittance = arena_array<float>(allocate, pixels);
        auto* safe_mean = arena_array<float2>(allocate, 1);
        auto* safe_conic = arena_array<float>(allocate, 3);
        auto* safe_opacity = arena_array<float>(allocate, 1);
        auto* safe_color = arena_array<float3>(allocate, 1);
        auto* native_mean = arena_array<float2>(allocate, 1);
        auto* native_conic = arena_array<float>(allocate, 3);
        auto* native_opacity = arena_array<float>(allocate, 1);
        auto* native_color = arena_array<float3>(allocate, 1);
        ASSERT_NE(tile_instance_ranges, nullptr);
        ASSERT_NE(instance_primitive_indices, nullptr);
        ASSERT_NE(primitive_mean2d, nullptr);
        ASSERT_NE(primitive_conic_opacity, nullptr);
        ASSERT_NE(primitive_color, nullptr);
        ASSERT_NE(grad_image, nullptr);
        ASSERT_NE(grad_alpha, nullptr);
        ASSERT_NE(image, nullptr);
        ASSERT_NE(alpha_map, nullptr);
        ASSERT_NE(contributions, nullptr);
        ASSERT_NE(transmittance, nullptr);
        ASSERT_NE(safe_mean, nullptr);
        ASSERT_NE(safe_conic, nullptr);
        ASSERT_NE(safe_opacity, nullptr);
        ASSERT_NE(safe_color, nullptr);
        ASSERT_NE(native_mean, nullptr);
        ASSERT_NE(native_conic, nullptr);
        ASSERT_NE(native_opacity, nullptr);
        ASSERT_NE(native_color, nullptr);

        ASSERT_TRUE(arena.owns_device_allocation(safe_mean, 2 * sizeof(float)));
        ASSERT_TRUE(arena.owns_device_allocation(safe_conic, 3 * sizeof(float)));
        ASSERT_TRUE(arena.owns_device_allocation(safe_opacity, sizeof(float)));
        ASSERT_TRUE(arena.owns_device_allocation(safe_color, sizeof(float3)));
        ASSERT_TRUE(arena.owns_device_allocation(native_mean, 2 * sizeof(float)));
        ASSERT_TRUE(arena.owns_device_allocation(native_conic, 3 * sizeof(float)));
        ASSERT_TRUE(arena.owns_device_allocation(native_opacity, sizeof(float)));
        ASSERT_TRUE(arena.owns_device_allocation(native_color, sizeof(float3)));

        std::vector<float> image_gradient(pixels * 3);
        std::vector<float> alpha_gradient(pixels);
        std::vector<float> image_values(pixels * 3, 0.2f);
        std::vector<float> alpha_values(pixels, 0.3f);
        std::vector<unsigned int> contribution_counts(pixels, 2);
        std::vector<float> final_transmittance(pixels, 0.49f);
        for (size_t i = 0; i < pixels; ++i) {
            image_gradient[i] = 0.01f * static_cast<float>(i + 1);
            image_gradient[pixels + i] = 0.02f * static_cast<float>(pixels - i);
            image_gradient[2 * pixels + i] = 0.015f * static_cast<float>((i % 11) + 1);
            alpha_gradient[i] = 0.005f * static_cast<float>((i % 7) + 1);
        }

        ASSERT_TRUE(upload(tile_instance_ranges, std::vector<uint2>{{0, 2}}));
        ASSERT_TRUE(upload(instance_primitive_indices, std::vector<unsigned int>{0, 0}));
        ASSERT_TRUE(upload(primitive_mean2d, std::vector<float2>{make_float2(7.5f, 7.5f)}));
        ASSERT_TRUE(upload(primitive_conic_opacity, std::vector<float4>{make_float4(1.0f, 0.0f, 1.0f, 0.3f)}));
        ASSERT_TRUE(upload(primitive_color, std::vector<float3>{make_float3(0.5f, 0.25f, 0.75f)}));
        ASSERT_TRUE(upload(grad_image, image_gradient));
        ASSERT_TRUE(upload(grad_alpha, alpha_gradient));
        ASSERT_TRUE(upload(image, image_values));
        ASSERT_TRUE(upload(alpha_map, alpha_values));
        ASSERT_TRUE(upload(contributions, contribution_counts));
        ASSERT_TRUE(upload(transmittance, final_transmittance));
        ASSERT_EQ(cudaMemset(safe_mean, 0, 2 * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(safe_conic, 0, 3 * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(safe_opacity, 0, sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(safe_color, 0, sizeof(float3)), cudaSuccess);
        ASSERT_EQ(cudaMemset(native_mean, 0, 2 * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(native_conic, 0, 3 * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(native_opacity, 0, sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemset(native_color, 0, sizeof(float3)), cudaSuccess);

        ASSERT_EQ(launch_fastgs_blend_backward_probe(
                      tile_instance_ranges, instance_primitive_indices, primitive_mean2d,
                      primitive_conic_opacity, primitive_color, grad_image, grad_alpha,
                      image, alpha_map, contributions, transmittance, safe_mean, safe_conic,
                      safe_opacity, safe_color, native_mean, native_conic, native_opacity,
                      native_color, 2, 1, width, height),
                  cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        std::array<float, 24> actual{};
        ASSERT_EQ(cudaMemcpy(actual.data(), safe_mean, 2 * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual.data() + 2, native_mean, 2 * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual.data() + 4, safe_conic, 3 * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual.data() + 7, native_conic, 3 * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual.data() + 10, safe_opacity, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual.data() + 11, native_opacity, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual.data() + 12, safe_color, sizeof(float3), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual.data() + 15, native_color, sizeof(float3), cudaMemcpyDeviceToHost), cudaSuccess);
        for (const auto [safe_index, native_index, count] : {
                 std::array<size_t, 3>{0, 2, 2}, {4, 7, 3}, {10, 11, 1}, {12, 15, 3}}) {
            float safe_max_abs = 0.0f;
            float native_max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                safe_max_abs = std::max(safe_max_abs, std::fabs(actual[safe_index + i]));
                native_max_abs = std::max(native_max_abs, std::fabs(actual[native_index + i]));
            }
            EXPECT_GT(safe_max_abs, 1.0e-7f);
            EXPECT_GT(native_max_abs, 1.0e-7f);
        }
        for (const auto [safe_index, native_index, count] : {
                 std::array<size_t, 3>{0, 2, 2}, {4, 7, 3}, {10, 11, 1}, {12, 15, 3}}) {
            for (size_t i = 0; i < count; ++i) {
                const float safe = actual[safe_index + i];
                const float native = actual[native_index + i];
                EXPECT_TRUE(std::isfinite(safe));
                EXPECT_TRUE(std::isfinite(native));
                EXPECT_NEAR(native, safe, 1.0e-5f + 1.0e-5f * std::fabs(safe));
            }
        }
    }
} // namespace
