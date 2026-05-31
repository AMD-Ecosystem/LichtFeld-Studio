/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <curand_kernel.h> (device-side RNG). Routes to
 * hiprand_kernel.h, which mirrors the curand device API
 * (curandState/curand_init/curand_uniform/curand_normal -> hiprand* via the
 * compat-header aliases).
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <hiprand/hiprand_kernel.h>
