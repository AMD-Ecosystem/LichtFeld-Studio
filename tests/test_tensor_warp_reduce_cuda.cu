/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor/backend/cuda/kernels/warp_reduce.cuh"

namespace {

    __global__ void consecutive_reductions_kernel(float* output) {
        const float x = static_cast<float>(threadIdx.x + 1);
        const int base = static_cast<int>(blockIdx.x) * 8;

        const float sum_a = lfs::core::warp_ops::block_reduce_sum(x);
        const float sum_b = lfs::core::warp_ops::block_reduce_sum(-2.0f * x);
        const float max_a = lfs::core::warp_ops::block_reduce_max(x);
        const float max_b = lfs::core::warp_ops::block_reduce_max(-x);
        const float min_a = lfs::core::warp_ops::block_reduce_min(x);
        const float min_b = lfs::core::warp_ops::block_reduce_min(-x);
        const float prod_a = lfs::core::warp_ops::block_reduce_prod(1.0f);
        const float prod_b = lfs::core::warp_ops::block_reduce_prod(1.0001f);

        if (threadIdx.x == 0) {
            output[base + 0] = sum_a;
            output[base + 1] = sum_b;
            output[base + 2] = max_a;
            output[base + 3] = max_b;
            output[base + 4] = min_a;
            output[base + 5] = min_b;
            output[base + 6] = prod_a;
            output[base + 7] = prod_b;
        }
    }

    template <typename T>
    __global__ void block_reductions_kernel(const T* input, T* output) {
        const T x = input[blockIdx.x * blockDim.x + threadIdx.x];
        const T sum = lfs::core::warp_ops::block_reduce_sum(x);
        const T max = lfs::core::warp_ops::block_reduce_max(x);
        const T min = lfs::core::warp_ops::block_reduce_min(x);
        const T prod = lfs::core::warp_ops::block_reduce_prod(static_cast<T>(threadIdx.x % 37 == 0 ? 2 : 1));
        if (threadIdx.x == 0) {
            output[blockIdx.x * 4 + 0] = sum;
            output[blockIdx.x * 4 + 1] = max;
            output[blockIdx.x * 4 + 2] = min;
            output[blockIdx.x * 4 + 3] = prod;
        }
    }

    template <typename T>
    __global__ void warp_reductions_kernel(const T* input, T* output) {
        const T x = input[threadIdx.x];
        output[threadIdx.x * 4 + 0] = lfs::core::warp_ops::warp_reduce_sum(x);
        output[threadIdx.x * 4 + 1] = lfs::core::warp_ops::warp_reduce_max(x);
        output[threadIdx.x * 4 + 2] = lfs::core::warp_ops::warp_reduce_min(x);
        output[threadIdx.x * 4 + 3] = lfs::core::warp_ops::warp_reduce_prod(static_cast<T>(threadIdx.x % 7 == 0 ? 2 : 1));
    }

} // namespace

void launch_consecutive_tensor_warp_reductions(float* output, int blocks) {
    consecutive_reductions_kernel<<<blocks, 256>>>(output);
}

template <typename T>
void launch_tensor_block_reductions(const T* input, T* output, int blocks, int block_size) {
    block_reductions_kernel<T><<<blocks, block_size>>>(input, output);
}

template <typename T>
void launch_tensor_warp_reductions(const T* input, T* output, int threads) {
    warp_reductions_kernel<T><<<1, threads>>>(input, output);
}

template void launch_tensor_block_reductions<float>(const float*, float*, int, int);
template void launch_tensor_block_reductions<double>(const double*, double*, int, int);
template void launch_tensor_block_reductions<int>(const int*, int*, int, int);
template void launch_tensor_warp_reductions<float>(const float*, float*, int);
template void launch_tensor_warp_reductions<double>(const double*, double*, int);
template void launch_tensor_warp_reductions<int>(const int*, int*, int);
