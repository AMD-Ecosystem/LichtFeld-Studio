/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/hip_host_timeline.hpp"

#include <mutex>

namespace lfs::core::hip_host_timeline {

    namespace {
        std::mutex g_ops_mutex;
        VulkanOps g_ops;

        VulkanOps current_ops() {
            std::lock_guard lock(g_ops_mutex);
            return g_ops;
        }

        struct HostSemaphore {
            void* vulkan_semaphore = nullptr;
        };

        HostSemaphore* host_semaphore(const hipExternalSemaphore_t semaphore) {
            return reinterpret_cast<HostSemaphore*>(semaphore);
        }
    } // namespace

    void set_vulkan_ops(const VulkanOps& ops) {
        std::lock_guard lock(g_ops_mutex);
        g_ops = ops;
    }

    hipError_t import_semaphore(hipExternalSemaphore_t* const semaphore,
                                const hipExternalSemaphoreHandleDesc* const desc) {
        if (semaphore == nullptr || desc == nullptr) {
            return hipErrorInvalidValue;
        }
        if (desc->type != hipExternalSemaphoreHandleTypeTimelineSemaphoreFd) {
            return hipErrorNotSupported;
        }
        const VulkanOps ops = current_ops();
        if (ops.import_fd == nullptr) {
            return hipErrorNotSupported;
        }
        void* const imported = ops.import_fd(ops.context, desc->handle.fd);
        if (imported == nullptr) {
            return hipErrorInvalidValue;
        }
        *semaphore = reinterpret_cast<hipExternalSemaphore_t>(new HostSemaphore{imported});
        return hipSuccess;
    }

    hipError_t signal_semaphores(const hipExternalSemaphore_t* const semaphores,
                                 const hipExternalSemaphoreSignalParams* const params,
                                 const unsigned int count, const hipStream_t stream) {
        if (count == 0) {
            return hipSuccess;
        }
        if (semaphores == nullptr || params == nullptr) {
            return hipErrorInvalidValue;
        }
        const VulkanOps ops = current_ops();
        if (ops.signal == nullptr) {
            return hipErrorNotSupported;
        }
        if (const hipError_t status = hipStreamSynchronize(stream); status != hipSuccess) {
            return status;
        }
        for (unsigned int i = 0; i < count; ++i) {
            const HostSemaphore* const sem = host_semaphore(semaphores[i]);
            if (sem == nullptr ||
                !ops.signal(ops.context, sem->vulkan_semaphore, params[i].params.fence.value)) {
                return hipErrorInvalidValue;
            }
        }
        return hipSuccess;
    }

    hipError_t wait_semaphores(const hipExternalSemaphore_t* const semaphores,
                               const hipExternalSemaphoreWaitParams* const params,
                               const unsigned int count, hipStream_t) {
        if (count == 0) {
            return hipSuccess;
        }
        if (semaphores == nullptr || params == nullptr) {
            return hipErrorInvalidValue;
        }
        const VulkanOps ops = current_ops();
        if (ops.wait == nullptr) {
            return hipErrorNotSupported;
        }
        // Work the caller enqueues after this returns starts after Vulkan reached the value.
        for (unsigned int i = 0; i < count; ++i) {
            const HostSemaphore* const sem = host_semaphore(semaphores[i]);
            if (sem == nullptr ||
                !ops.wait(ops.context, sem->vulkan_semaphore, params[i].params.fence.value)) {
                return hipErrorLaunchTimeOut;
            }
        }
        return hipSuccess;
    }

    hipError_t destroy_semaphore(const hipExternalSemaphore_t semaphore) {
        HostSemaphore* const sem = host_semaphore(semaphore);
        if (sem == nullptr) {
            return hipErrorInvalidValue;
        }
        const VulkanOps ops = current_ops();
        if (ops.destroy != nullptr) {
            ops.destroy(ops.context, sem->vulkan_semaphore);
        }
        delete sem;
        return hipSuccess;
    }

} // namespace lfs::core::hip_host_timeline
