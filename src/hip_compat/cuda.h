/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cuda.h>. gsplat's Common.h includes <cuda.h> solely
 * to obtain CUDA_VERSION for GLM's compiler detection. The compat header already
 * defines CUDA_VERSION; this shim just routes the include to it.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
