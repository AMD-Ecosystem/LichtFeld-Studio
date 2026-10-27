/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cuda_profiler_api.h>; the compat header maps
 * cudaProfilerStart/Stop to the HIP calls.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
