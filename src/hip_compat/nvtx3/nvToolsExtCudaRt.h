/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HIP stand-in for <nvtx3/nvToolsExtCudaRt.h>: stream naming is a profiler
 * annotation only, so it is a no-op.
 */
#pragma once
#include "nvtx3/nvToolsExt.h"

#define nvtxNameCudaStreamA(stream, name) ((void)(stream), (void)(name))
