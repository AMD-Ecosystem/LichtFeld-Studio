/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/hip_stream_reuse.hpp"

#include <iterator>
#include <mutex>
#include <vector>

namespace lfs::core::hip_stream_reuse {

    namespace {
        struct ParkedStream {
            hipStream_t stream;
            int device;
            unsigned int flags;
            int priority;
        };

        // Never destroyed: a parked stream must outlive every event recorded on
        // it, including events queried from static destructors.
        std::mutex& parked_mutex() {
            static auto* const mutex = new std::mutex;
            return *mutex;
        }

        std::vector<ParkedStream>& parked_streams() {
            static auto* const streams = new std::vector<ParkedStream>;
            return *streams;
        }

        constexpr int kDefaultPriority = 0;
    } // namespace

    hipError_t create(hipStream_t* const stream) {
        return create_with_flags(stream, hipStreamDefault);
    }

    hipError_t create_with_flags(hipStream_t* const stream, const unsigned int flags) {
        int device = 0;
        if (stream != nullptr && hipGetDevice(&device) == hipSuccess) {
            std::lock_guard lock(parked_mutex());
            auto& parked = parked_streams();
            for (auto it = parked.rbegin(); it != parked.rend(); ++it) {
                if (it->device == device && it->flags == flags && it->priority == kDefaultPriority) {
                    *stream = it->stream;
                    parked.erase(std::next(it).base());
                    return hipSuccess;
                }
            }
        }
        return hipStreamCreateWithFlags(stream, flags);
    }

    hipError_t destroy(const hipStream_t stream) {
        if (stream == nullptr || stream == hipStreamPerThread || stream == hipStreamLegacy) {
            return hipStreamDestroy(stream);
        }
        // Destroying a capturing stream invalidates the capture; leave that to HIP.
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        ParkedStream entry{stream, 0, 0, 0};
        if (hipStreamIsCapturing(stream, &capture) != hipSuccess ||
            capture != hipStreamCaptureStatusNone ||
            hipStreamGetDevice(stream, &entry.device) != hipSuccess ||
            hipStreamGetFlags(stream, &entry.flags) != hipSuccess ||
            hipStreamGetPriority(stream, &entry.priority) != hipSuccess ||
            hipStreamSynchronize(stream) != hipSuccess) {
            return hipStreamDestroy(stream);
        }
        std::lock_guard lock(parked_mutex());
        parked_streams().push_back(entry);
        return hipSuccess;
    }

} // namespace lfs::core::hip_stream_reuse
