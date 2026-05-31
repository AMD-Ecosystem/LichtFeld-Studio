/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <nvtx3/nvToolsExt.h>. The NVTX profiling ranges are
 * not part of GPU correctness; the compat header maps nvtxRangePush/Pop to
 * no-ops, so this shim only needs to make the include resolve.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
