/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cuda_fp16.h>. hip_fp16.h provides __half/__half2 and
 * the same intrinsics as drop-ins. The compat header already pulls it in.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <hip/hip_fp16.h>
