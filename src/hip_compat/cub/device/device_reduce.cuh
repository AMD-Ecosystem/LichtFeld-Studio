/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <cub/device/device_reduce.cuh>. hipCUB's umbrella
 * header provides DeviceReduce; alias cub -> hipcub.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
