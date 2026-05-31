/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <device_launch_parameters.h>. HIP exposes threadIdx/
 * blockIdx/blockDim/gridDim through <hip/hip_runtime.h>, so this just pulls the
 * compat header.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
