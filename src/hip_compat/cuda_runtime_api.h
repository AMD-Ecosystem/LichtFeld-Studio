/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cuda_runtime_api.h> (pulled by ROCm libtorch's
 * c10/cuda headers in the test binary). Routes to the compat header / HIP
 * runtime.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
