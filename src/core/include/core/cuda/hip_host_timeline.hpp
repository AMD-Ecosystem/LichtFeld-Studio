/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP on Linux cannot import a Vulkan semaphore: hipImportExternalSemaphore
 * rejects the timeline handle types, and the runtime does not implement opaque
 * fd semaphores. The compatibility header routes the CUDA external-semaphore
 * calls here, where a Vulkan timeline semaphore is ordered against HIP streams
 * on the host: a wait blocks the caller until Vulkan reaches the value, and a
 * signal synchronizes the stream and then signals the value from the host.
 * The Vulkan side registers the operations once its device exists.
 */
#pragma once

#include "core/export.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace lfs::core::hip_host_timeline {

    struct VulkanOps {
        void* context = nullptr;
        // Imports a timeline semaphore fd, taking ownership of it on success.
        // Returns the imported semaphore or nullptr.
        void* (*import_fd)(void* context, int fd) = nullptr;
        // Blocks until the semaphore reaches value.
        bool (*wait)(void* context, void* semaphore, std::uint64_t value) = nullptr;
        bool (*signal)(void* context, void* semaphore, std::uint64_t value) = nullptr;
        void (*destroy)(void* context, void* semaphore) = nullptr;
    };

    // Registers the Vulkan operations; a default-constructed VulkanOps unregisters.
    LFS_CORE_API void set_vulkan_ops(const VulkanOps& ops);

    LFS_CORE_API hipError_t import_semaphore(hipExternalSemaphore_t* semaphore,
                                             const hipExternalSemaphoreHandleDesc* desc);
    LFS_CORE_API hipError_t signal_semaphores(const hipExternalSemaphore_t* semaphores,
                                              const hipExternalSemaphoreSignalParams* params,
                                              unsigned int count, hipStream_t stream);
    LFS_CORE_API hipError_t wait_semaphores(const hipExternalSemaphore_t* semaphores,
                                            const hipExternalSemaphoreWaitParams* params,
                                            unsigned int count, hipStream_t stream);
    LFS_CORE_API hipError_t destroy_semaphore(hipExternalSemaphore_t semaphore);

} // namespace lfs::core::hip_host_timeline
