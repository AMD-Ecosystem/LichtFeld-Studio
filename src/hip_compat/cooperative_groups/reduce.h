/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cooperative_groups/reduce.h>. ROCm 7.2.1 cooperative
 * groups does NOT provide cg::reduce / this header. Sites that called
 * cg::reduce (gsplat Utils.cuh warpSum/warpMax, RasterizeToPixelsFromWorld3DGSBwd
 * warp_bin_final) are rewritten under USE_HIP to a butterfly __shfl_xor reduction
 * over the 32-lane tile (wave-size-agnostic). This shim only needs to make the
 * include resolve and pull the base cooperative groups header.
 */
#pragma once
#include <hip/hip_cooperative_groups.h>
