/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim. On the ROCm build this directory is on the include path
 * (only under USE_HIP) so `#include <cuda_runtime.h>` resolves here instead of
 * the absent CUDA toolkit header. It pulls the compat header, which includes
 * the HIP runtime and aliases the cuda* symbols. On a CUDA build this directory
 * is not on the include path, so the real toolkit header is used unchanged.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
