/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor/backend/cuda/kernels/warp_reduce.cuh"

#include <cmath>
#include <cstddef>

namespace lfs::core::tensor_ops {
    namespace cg = cooperative_groups;

    /**
     * @brief FAST kernel for MEDIUM segments (32-2048 elements)
     *
     * Each WARP processes one segment entirely using warp shuffle reductions.
     * NO block synchronization needed!
     *
     * OPTIMIZATION: COALESCED float4 loads - all lanes read consecutive memory.
     * For segment_size=1024: 32 lanes * 4 floats = 128 floats per iteration, 8 iterations.
     */
    __global__ void warp_medium_segment_reduce_sum_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const auto warp32 = cg::tiled_partition<32>(cg::this_thread_block());
        const int warp_id = warp32.meta_group_rank();
        const int lane = warp32.thread_rank();
        const int warps_per_block = warp32.meta_group_size();

        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float sum = 0.0f;

            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += warp32.num_threads()) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        sum += v.x + v.y + v.z + v.w;
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += warp32.num_threads()) {
                    sum += segment_start[i];
                }
            }

            sum = warp_ops::warp_reduce_sum(sum);

            if (lane == 0) {
                output[seg_idx] = sum;
            }
        }
    }

    /**
     * @brief FAST kernel for MEDIUM segments - MEAN variant with coalesced loads
     */
    __global__ void warp_medium_segment_reduce_mean_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const auto warp32 = cg::tiled_partition<32>(cg::this_thread_block());
        const int warp_id = warp32.meta_group_rank();
        const int lane = warp32.thread_rank();
        const int warps_per_block = warp32.meta_group_size();
        const float inv_size = 1.0f / static_cast<float>(segment_size);
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float sum = 0.0f;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += warp32.num_threads()) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        sum += v.x + v.y + v.z + v.w;
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += warp32.num_threads()) {
                    sum += segment_start[i];
                }
            }

            sum = warp_ops::warp_reduce_sum(sum);

            if (lane == 0) {
                output[seg_idx] = sum * inv_size;
            }
        }
    }

    /**
     * @brief FAST kernel for MEDIUM segments - MAX variant with coalesced loads
     */
    __global__ void warp_medium_segment_reduce_max_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const auto warp32 = cg::tiled_partition<32>(cg::this_thread_block());
        const int warp_id = warp32.meta_group_rank();
        const int lane = warp32.thread_rank();
        const int warps_per_block = warp32.meta_group_size();
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float val = -INFINITY;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += warp32.num_threads()) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        val = ops::max_reduce_op{}(val, ops::max_reduce_op{}(ops::max_reduce_op{}(v.x, v.y), ops::max_reduce_op{}(v.z, v.w)));
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += warp32.num_threads()) {
                    val = ops::max_reduce_op{}(val, segment_start[i]);
                }
            }

            val = warp_ops::warp_reduce_max(val);

            if (lane == 0) {
                output[seg_idx] = val;
            }
        }
    }

    /**
     * @brief FAST kernel for MEDIUM segments - MIN variant with coalesced loads
     */
    __global__ void warp_medium_segment_reduce_min_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const auto warp32 = cg::tiled_partition<32>(cg::this_thread_block());
        const int warp_id = warp32.meta_group_rank();
        const int lane = warp32.thread_rank();
        const int warps_per_block = warp32.meta_group_size();
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float val = INFINITY;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += warp32.num_threads()) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        val = ops::min_reduce_op{}(val, ops::min_reduce_op{}(ops::min_reduce_op{}(v.x, v.y), ops::min_reduce_op{}(v.z, v.w)));
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += warp32.num_threads()) {
                    val = ops::min_reduce_op{}(val, segment_start[i]);
                }
            }

            val = warp_ops::warp_reduce_min(val);

            if (lane == 0) {
                output[seg_idx] = val;
            }
        }
    }

    __global__ void warp_medium_segment_reduce_prod_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        size_t num_segments,
        size_t segment_size) {
        const auto warp32 = cg::tiled_partition<32>(cg::this_thread_block());
        const int warp_id = warp32.meta_group_rank();
        const int lane = warp32.thread_rank();
        const int warps_per_block = warp32.meta_group_size();
        const bool can_vectorize = (segment_size % 4) == 0;

        for (size_t seg_idx = blockIdx.x * warps_per_block + warp_id;
             seg_idx < num_segments;
             seg_idx += gridDim.x * warps_per_block) {
            const float* segment_start = input + seg_idx * segment_size;

            float val = 1.0f;
            if (can_vectorize) {
                const size_t num_float4s = segment_size / 4;
                for (size_t base = 0; base < num_float4s; base += warp32.num_threads()) {
                    size_t idx = base + lane;
                    if (idx < num_float4s) {
                        float4 v = reinterpret_cast<const float4*>(segment_start)[idx];
                        val *= v.x * v.y * v.z * v.w;
                    }
                }
            } else {
                for (size_t i = lane; i < segment_size; i += warp32.num_threads()) {
                    val *= segment_start[i];
                }
            }

            val = warp_ops::warp_reduce_prod(val);

            if (lane == 0) {
                output[seg_idx] = val;
            }
        }
    }

} // namespace lfs::core::tensor_ops
