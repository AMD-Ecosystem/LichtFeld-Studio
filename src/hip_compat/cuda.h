/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cuda.h>. gsplat's Common.h includes <cuda.h> solely
 * to obtain CUDA_VERSION for GLM's compiler detection. The compat header already
 * defines CUDA_VERSION; this shim just routes the include to it. FFmpeg's
 * hwcontext_cuda.h includes <cuda.h> inside extern "C", so restore C++ linkage
 * for the C++ headers the compat header pulls in.
 */
#pragma once
#ifdef __cplusplus
extern "C++" {
#include "core/cuda/cuda_to_hip.h"
}
#else
#include "core/cuda/cuda_to_hip.h"
#endif
