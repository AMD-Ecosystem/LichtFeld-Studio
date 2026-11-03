/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/device/portability.cuh"

#include <cstddef>

namespace lfs::core::tensor_ops {
    namespace cg = cooperative_groups;

    // Channel3D broadcast kernel WITH COALESCED MEMORY ACCESS
    // Pattern: (H×W×C) op (1×1×C) - OPTIMIZED for memory coalescing
    // Each WARP processes one pixel, threads cooperatively handle channels
    // This achieves COALESCED access: threads in warp load CONSECUTIVE channels!
    template <typename BinaryOp>
    __global__ void broadcast_channel3d_coalesced_kernel_float(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ c,
        size_t H, size_t W, size_t C,
        bool a_is_broadcast,
        BinaryOp op) {
        // Shared memory for broadcast channels
        extern __shared__ float s_channels[];

        const float* channel_vals = a_is_broadcast ? a : b;
        const float* pixel_data = a_is_broadcast ? b : a;

        // Load channels into shared memory with vectorization
        for (size_t ch = threadIdx.x * 4; ch + 3 < C; ch += blockDim.x * 4) {
            float4 vals = reinterpret_cast<const float4*>(channel_vals)[ch / 4];
            reinterpret_cast<float4*>(s_channels)[ch / 4] = vals;
        }
        // Handle remainder
        for (size_t ch = (C / 4) * 4 + threadIdx.x; ch < C; ch += blockDim.x) {
            s_channels[ch] = channel_vals[ch];
        }
        __syncthreads();

        // Each warp processes different pixels
        const auto warp32 = cg::tiled_partition<32>(cg::this_thread_block());
        const int warp_id = warp32.meta_group_rank();
        const int lane_id = warp32.thread_rank();
        const int num_warps = warp32.meta_group_size();
        const size_t total_pixels = H * W;

        // Assign pixels to warps
        for (size_t pixel = blockIdx.x * num_warps + warp_id;
             pixel < total_pixels;
             pixel += gridDim.x * num_warps) {
            const size_t pixel_offset = pixel * C;

            // Process channels in groups of 32 (one per lane)
            for (size_t ch_base = 0; ch_base < C; ch_base += warp32.num_threads()) {
                const size_t ch = ch_base + lane_id;

                if (ch < C) {
                    // COALESCED LOAD: All 32 threads load consecutive channels
                    const float pix_val = pixel_data[pixel_offset + ch];
                    const float chan_val = s_channels[ch];

                    // Compute result
                    const float result = a_is_broadcast ? op(chan_val, pix_val) : op(pix_val, chan_val);

                    // COALESCED STORE: All 32 threads write consecutive channels
                    c[pixel_offset + ch] = result;
                }
            }
        }
    }

} // namespace lfs::core::tensor_ops
