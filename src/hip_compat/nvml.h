/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP stand-in for <nvml.h>. The VRAM profiler loads NVML at run time with
 * dlopen, which finds nothing on an AMD system; it only needs these
 * declarations (layout as in NVML's nvmlProcessInfo_t) to compile.
 */
#pragma once

typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_INSUFFICIENT_SIZE = 7
} nvmlReturn_t;

#define NVML_VALUE_NOT_AVAILABLE (-1)

typedef struct nvmlProcessInfo_st {
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
} nvmlProcessInfo_t;

/* Layout as in NVML's nvmlMemory_v2_t; nvmlMemory_v2 is its version tag. */
typedef struct nvmlMemory_v2_st {
    unsigned int version;
    unsigned long long total;
    unsigned long long reserved;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_v2_t;

#define nvmlMemory_v2 (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U))
