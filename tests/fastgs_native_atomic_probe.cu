/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/rasterization/fastgs/rasterization/include/kernels_backward.cuh"

namespace {
    template <bool Native>
    __global__ void add_gradient_probe(float* sum, int count, float value, bool alternate_sign) {
        const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (index < count) {
            const float increment = alternate_sign && (index & 1) ? -value : value;
            fast_lfs::rasterization::kernels::atomic_add_device_gradient<Native>(sum, increment);
        }
    }
} // namespace

cudaError_t launch_fastgs_atomic_probe(float* sums, int count, float value,
                                       bool alternate_sign, cudaStream_t stream) {
    const int blocks = (count + 255) / 256;
    add_gradient_probe<false><<<blocks, 256, 0, stream>>>(sums, count, value, alternate_sign);
    const auto first = cudaGetLastError();
    if (first != cudaSuccess) {
        return first;
    }
    add_gradient_probe<true><<<blocks, 256, 0, stream>>>(sums + 1, count, value, alternate_sign);
    return cudaGetLastError();
}

cudaError_t launch_fastgs_blend_backward_probe(
    const uint2* tile_instance_ranges,
    const uint* instance_primitive_indices,
    const float2* primitive_mean2d,
    const float4* primitive_conic_opacity,
    const float3* primitive_color,
    const float* grad_image,
    const float* grad_alpha_map,
    const float* image,
    const float* alpha_map,
    const uint* tile_n_contributions,
    const float* tile_final_transmittance,
    float2* safe_grad_mean2d,
    float* safe_grad_conic,
    float* safe_grad_opacity,
    float3* safe_grad_color,
    float2* native_grad_mean2d,
    float* native_grad_conic,
    float* native_grad_opacity,
    float3* native_grad_color,
    uint n_instances,
    uint n_primitives,
    uint width,
    uint height) {
    using fast_lfs::rasterization::config::block_size_blend_backward;
    using fast_lfs::rasterization::kernels::backward::blend_backward_cu;
    blend_backward_cu<::DensificationType::None, false><<<1, block_size_blend_backward>>>(
        tile_instance_ranges, instance_primitive_indices, primitive_mean2d,
        primitive_conic_opacity, primitive_color, grad_image, grad_alpha_map,
        image, alpha_map, tile_n_contributions, tile_final_transmittance,
        safe_grad_mean2d, safe_grad_conic, safe_grad_opacity, safe_grad_color,
        nullptr, nullptr, nullptr, n_instances, n_primitives, width, height, 1);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    blend_backward_cu<::DensificationType::None, true><<<1, block_size_blend_backward>>>(
        tile_instance_ranges, instance_primitive_indices, primitive_mean2d,
        primitive_conic_opacity, primitive_color, grad_image, grad_alpha_map,
        image, alpha_map, tile_n_contributions, tile_final_transmittance,
        native_grad_mean2d, native_grad_conic, native_grad_opacity, native_grad_color,
        nullptr, nullptr, nullptr, n_instances, n_primitives, width, height, 1);
    return cudaGetLastError();
}
