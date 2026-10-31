/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CUDA lets an event be queried or synchronized after the stream it was
 * recorded on is destroyed. HIP does not: hipEventRecord keeps the recording
 * stream in the event, and hipEventQuery and hipEventSynchronize read that
 * stream's capture state without checking that it still exists, so once the
 * heap reuses the freed stream they hang or crash. The compatibility header
 * routes stream creation and destruction here. destroy() synchronizes the
 * stream and parks it instead of freeing it, and create() hands a parked
 * stream with the same device, flags and priority back out, so every stream
 * an event was ever recorded on stays alive. The parked streams never exceed
 * the number that were alive at once.
 */
#pragma once

#include "core/export.hpp"

#include <hip/hip_runtime_api.h>

namespace lfs::core::hip_stream_reuse {

    LFS_CORE_API hipError_t create(hipStream_t* stream);
    LFS_CORE_API hipError_t create_with_flags(hipStream_t* stream, unsigned int flags);
    LFS_CORE_API hipError_t destroy(hipStream_t stream);

} // namespace lfs::core::hip_stream_reuse
