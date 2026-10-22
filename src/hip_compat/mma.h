/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP forwarding shim for <mma.h>. The nvcuda::wmma fragments are only used
 * under __CUDA_ARCH__ >= 700, which the HIP device pass never satisfies, so the
 * kernels take their scalar fallbacks and nothing from this header is needed.
 */
#pragma once
#include "core/cuda/cuda_to_hip.h"
