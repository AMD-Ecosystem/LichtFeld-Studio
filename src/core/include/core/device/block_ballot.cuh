/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/device/portability.cuh"

#include <cstdint>

namespace lfs::core::device {
    // One bit per thread of a 64-thread block, in thread order, for any
    // hardware warp width. Every thread of the block must call it.
    __device__ inline uint64_t block_ballot64(const bool predicate) {
        namespace cg = cooperative_groups;
        const auto warp32 = cg::tiled_partition<32>(cg::this_thread_block());
        __shared__ uint32_t masks[2];
        const auto bits = static_cast<uint32_t>(warp32.ballot(predicate));
        if (warp32.thread_rank() == 0)
            masks[warp32.meta_group_rank()] = bits;
        __syncthreads();
        return uint64_t(masks[0]) | (uint64_t(masks[1]) << 32);
    }
} // namespace lfs::core::device
