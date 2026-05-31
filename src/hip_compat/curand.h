/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <curand.h> (host-side RNG). Routes to hiprand.h, which
 * mirrors the curand host API (curandCreateGenerator/curandGenerateNormal/... ->
 * hiprand* via the compat-header aliases).
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <hiprand/hiprand.h>
