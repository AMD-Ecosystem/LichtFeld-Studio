/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cuda_bf16.h>. hip_bf16.h provides the bfloat16 types
 * and intrinsics as drop-ins. The compat header already pulls it in.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <hip/hip_bf16.h>
