/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda/cuda_to_hip.h"

namespace fast_lfs::rasterization::kernels {

    // Native HIP atomics require coarse-grained device storage. Callers must
    // prove that the destination belongs to an arena allocation before using
    // Native=true; imported and caller-owned buffers stay on atomicAdd.
    template <bool Native>
    __device__ __forceinline__ void atomic_add_device_gradient(float* destination, float value) {
#if (defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)) && \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
     defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__))
        if constexpr (Native) {
            unsafeAtomicAdd(destination, value);
            return;
        }
#endif
        atomicAdd(destination, value);
    }

} // namespace fast_lfs::rasterization::kernels
