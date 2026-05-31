/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cub/cub.cuh>. Routes to hipCUB and aliases the `cub`
 * namespace to `hipcub` so cub::DoubleBuffer / cub::DeviceRadixSort /
 * cub::DeviceScan / cub::DeviceReduce in the sources resolve unchanged (the
 * cudaKDTree `namespace cub = hipcub` lesson; hipify maps the header but not the
 * namespace). hipCUB's DeviceRadixSort::SortPairs with begin_bit=0 over a 64-bit
 * key sorts correctly on ROCm (the nonzero-begin_bit hipCUB bug does not apply --
 * IntersectTile uses begin_bit=0).
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
