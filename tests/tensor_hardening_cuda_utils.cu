/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cstdint>

#include <cuda_runtime.h>

namespace tensor_hardening {
    namespace {
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
        // clock64() is a 20-bit wrapping shader-cycle counter on RDNA3, so a
        // cycle-bounded wait can end at its first wrap. wall_clock64() is a
        // 64-bit constant-rate counter on every AMD GPU.
        __global__ void delay_kernel(const uint64_t ticks) {
            const uint64_t start = wall_clock64();
            while (wall_clock64() - start < ticks) {
                __nanosleep(100);
            }
        }
#else
        __global__ void delay_kernel(const uint64_t cycles) {
            const uint64_t start = clock64();
            while (clock64() - start < cycles) {
                __nanosleep(100);
            }
        }
#endif
    } // namespace

    cudaError_t launch_delay_kernel(cudaStream_t stream, uint64_t cycles) {
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
        // Wait as long as `cycles` takes at 1 GHz (100 ms for the stream gate),
        // at least as long as on a CUDA GPU clocked at 1 GHz or more.
        int device = 0;
        int wall_clock_khz = 0;
        hipError_t err = hipGetDevice(&device);
        if (err == hipSuccess) {
            err = hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, device);
        }
        if (err != hipSuccess) {
            return err;
        }
        const uint64_t ticks = cycles * static_cast<uint64_t>(wall_clock_khz) / 1'000'000u;
        delay_kernel<<<1, 1, 0, stream>>>(ticks);
#else
        delay_kernel<<<1, 1, 0, stream>>>(cycles);
#endif
        return cudaGetLastError();
    }
} // namespace tensor_hardening
