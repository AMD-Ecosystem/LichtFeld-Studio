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
