/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cooperative_groups.h>. Routes to HIP's cooperative
 * groups header. HIP CG provides this_grid/this_thread_block/tiled_partition/
 * thread_block_tile and member shuffles, but NOT cg::reduce -- the gsplat
 * warpSum/warpMax sites supply a USE_HIP shfl_xor reduction instead (see
 * Utils.cuh and <cooperative_groups/reduce.h> shim).
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <hip/hip_cooperative_groups.h>
