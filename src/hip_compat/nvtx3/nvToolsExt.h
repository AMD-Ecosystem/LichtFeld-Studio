/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <nvtx3/nvToolsExt.h>. The NVTX profiling ranges are
 * not part of GPU correctness; the compat header maps nvtxRangePush/Pop to
 * no-ops, so this shim only needs to make the include resolve.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"

#include <cstdint>

/* The fields the viewport's colored ranges fill in; nvtxRangePushEx is a no-op. */
typedef struct nvtxEventAttributes_v2 {
    std::uint16_t version;
    std::uint16_t size;
    std::int32_t colorType;
    std::uint32_t color;
    std::int32_t messageType;
    union {
        const char* ascii;
    } message;
} nvtxEventAttributes_t;

#define NVTX_VERSION 3
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE static_cast<std::uint16_t>(sizeof(nvtxEventAttributes_t))
#define NVTX_COLOR_ARGB 1
#define NVTX_MESSAGE_TYPE_ASCII 1
#define nvtxRangePushEx(attribs) ((void)(attribs))
