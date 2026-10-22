/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <thrust/system/cuda/execution_policy.h>. rocThrust
 * keeps the device execution policy in thrust::hip; alias thrust::cuda to it.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <thrust/execution_policy.h>
namespace thrust { namespace cuda = hip; }
