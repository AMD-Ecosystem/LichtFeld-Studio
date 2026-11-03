/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Minimal checks for SDK-only GPU executables that compile shared CUDA/HIP kernels.
namespace lfs::hip_tests {

    inline int failures = 0;
    inline int checks = 0;

    inline void require(const hipError_t status, const char* what) {
        if (status != hipSuccess) {
            std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(status));
            std::exit(2);
        }
    }

    inline void expect(const bool ok, const std::string& what) {
        ++checks;
        if (!ok && ++failures <= 50)
            std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }

    inline void expect_near(const double got, const double want, const double tolerance, const std::string& what) {
        const bool ok = std::isinf(want) ? got == want : std::fabs(got - want) <= tolerance * std::fmax(1.0, std::fabs(want));
        expect(ok, what + ": got " + std::to_string(got) + ", want " + std::to_string(want));
    }

    // `--device N` selects the HIP ordinal, matching the other native GPU tests.
    inline void select_device(const int argc, char** argv) {
        int device = 0;
        for (int i = 1; i + 1 < argc; ++i)
            if (std::strcmp(argv[i], "--device") == 0)
                device = std::atoi(argv[i + 1]);
        require(hipSetDevice(device), "hipSetDevice");
        hipDeviceProp_t properties{};
        require(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
        std::printf("device %d: %s (%s), warp size %d\n", device, properties.name, properties.gcnArchName, properties.warpSize);
    }

    template <typename T>
    struct DeviceBuffer {
        T* data = nullptr;
        explicit DeviceBuffer(const size_t count) { require(hipMalloc(&data, count * sizeof(T) + 16), "hipMalloc"); }
        ~DeviceBuffer() { (void)hipFree(data); }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        void upload(const std::vector<T>& host, const size_t offset = 0) {
            require(hipMemcpy(data + offset, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
        }
        std::vector<T> download(const size_t count, const size_t offset = 0) const {
            std::vector<T> host(count);
            require(hipMemcpy(host.data(), data + offset, count * sizeof(T), hipMemcpyDeviceToHost), "download");
            return host;
        }
    };

    inline int finish() {
        require(hipDeviceSynchronize(), "hipDeviceSynchronize");
        std::printf("%d checks, %d failures\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }

} // namespace lfs::hip_tests
