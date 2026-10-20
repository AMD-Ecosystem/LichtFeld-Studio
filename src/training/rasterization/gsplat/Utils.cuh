#pragma once

#include "Common.h"
#include "core/cuda_safe_format.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    ///////////////////////////////
    // Reduce
    ///////////////////////////////

    // ROCm 7.2.1 cooperative groups provides no cg::reduce. The warp argument
    // here is always a cg::thread_block_tile<32>; a butterfly __shfl_xor over the
    // tile's 32 lanes is an all-reduce that is wave-size-agnostic (a 32-lane tile
    // on a 64-lane CDNA wavefront keeps its member-masked shuffles in-tile, so it
    // reduces per 32-lane group exactly like a 32-lane NVIDIA warp -- the
    // a prior gsplat port confirmed this). On CUDA we keep cg::reduce.
    // (PORTING_GUIDE: gsplat fault 4.)
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    template <class WarpT>
    inline __device__ float warpReduceSum(float val, WarpT& warp) {
#pragma unroll
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            val += warp.shfl_xor(val, offset);
        }
        return val;
    }
    template <class WarpT>
    inline __device__ float warpReduceMax(float val, WarpT& warp) {
#pragma unroll
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, warp.shfl_xor(val, offset));
        }
        return val;
    }
#define GSPLAT_WARP_SUM(val, warp) warpReduceSum((val), (warp))
// Raw lane intrinsics over a 32-lane tile: a 64-lane wavefront holds two
// tiles, so a ballot returns this tile's half of the wave mask and a shuffle
// addresses a lane inside the tile. The two tiles can diverge, so the mask
// names only this tile.
#define GSPLAT_TILE32_MASK LFS_GROUP32_LANE_MASK
#define GSPLAT_TILE32_BALLOT(pred) \
    static_cast<unsigned>(__ballot_sync(LFS_GROUP32_LANE_MASK, (pred)) >> (__lane_id() & 32u))
#define GSPLAT_TILE32_SHFL(var, src) __shfl_sync(LFS_GROUP32_LANE_MASK, (var), (src), 32)
#define GSPLAT_WARP_MAX(val, warp) warpReduceMax((val), (warp))
#else
#define GSPLAT_WARP_SUM(val, warp) cg::reduce((warp), (val), cg::plus<float>())
#define GSPLAT_TILE32_MASK 0xffffffffu
#define GSPLAT_TILE32_BALLOT(pred) __ballot_sync(0xffffffffu, (pred))
#define GSPLAT_TILE32_SHFL(var, src) __shfl_sync(0xffffffffu, (var), (src))
#define GSPLAT_WARP_MAX(val, warp) cg::reduce((warp), (val), cg::greater<float>())
#endif

    template <uint32_t DIM, class WarpT>
    inline __device__ void warpSum(float* val, WarpT& warp) {
#pragma unroll
        for (uint32_t i = 0; i < DIM; i++) {
            val[i] = GSPLAT_WARP_SUM(val[i], warp);
        }
    }

    template <class WarpT>
    inline __device__ void warpSum(float& val, WarpT& warp) {
        val = GSPLAT_WARP_SUM(val, warp);
    }

    template <class WarpT>
    inline __device__ void warpSum(vec4& val, WarpT& warp) {
        val.x = GSPLAT_WARP_SUM(val.x, warp);
        val.y = GSPLAT_WARP_SUM(val.y, warp);
        val.z = GSPLAT_WARP_SUM(val.z, warp);
        val.w = GSPLAT_WARP_SUM(val.w, warp);
    }

    template <class WarpT>
    inline __device__ void warpSum(vec3& val, WarpT& warp) {
        val.x = GSPLAT_WARP_SUM(val.x, warp);
        val.y = GSPLAT_WARP_SUM(val.y, warp);
        val.z = GSPLAT_WARP_SUM(val.z, warp);
    }

    template <class WarpT>
    inline __device__ void warpSum(vec2& val, WarpT& warp) {
        val.x = GSPLAT_WARP_SUM(val.x, warp);
        val.y = GSPLAT_WARP_SUM(val.y, warp);
    }

    template <class WarpT>
    inline __device__ void warpSum(mat4& val, WarpT& warp) {
        warpSum(val[0], warp);
        warpSum(val[1], warp);
        warpSum(val[2], warp);
        warpSum(val[3], warp);
    }

    template <class WarpT>
    inline __device__ void warpSum(mat3& val, WarpT& warp) {
        warpSum(val[0], warp);
        warpSum(val[1], warp);
        warpSum(val[2], warp);
    }

    template <class WarpT>
    inline __device__ void warpSum(mat2& val, WarpT& warp) {
        warpSum(val[0], warp);
        warpSum(val[1], warp);
    }

    template <class WarpT>
    inline __device__ void warpMax(float& val, WarpT& warp) {
        val = GSPLAT_WARP_MAX(val, warp);
    }

    inline __device__ float activated_opacity(const float raw) {
        return 1.0f / (1.0f + expf(-raw));
    }

    inline __device__ vec3 activated_scale(const vec3 raw) {
        return vec3(expf(raw.x), expf(raw.y), expf(raw.z));
    }

    inline __device__ int32_t chw_pix(const uint32_t channel,
                                      const int32_t pix_id,
                                      const uint32_t image_height,
                                      const uint32_t image_width) {
        return static_cast<int32_t>(channel * image_height * image_width) + pix_id;
    }

    ///////////////////////////////
    // Quaternion
    ///////////////////////////////

    inline __device__ mat3 quat_to_rotmat(const vec4 quat) {
        float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
        // normalize
        float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
        w *= inv_norm;
        float x2 = x * x, y2 = y * y, z2 = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;
        return mat3(
            (1.f - 2.f * (y2 + z2)),
            (2.f * (xy + wz)),
            (2.f * (xz - wy)), // 1st col
            (2.f * (xy - wz)),
            (1.f - 2.f * (x2 + z2)),
            (2.f * (yz + wx)), // 2nd col
            (2.f * (xz + wy)),
            (2.f * (yz - wx)),
            (1.f - 2.f * (x2 + y2)) // 3rd col
        );
    }

    inline __device__ void
    quat_to_rotmat_vjp(const vec4 quat, const mat3 v_R, vec4& v_quat) {
        float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
        // normalize
        float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
        w *= inv_norm;
        vec4 v_quat_n = vec4(
            2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                   z * (v_R[0][1] - v_R[1][0])),
            2.f *
                (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
                 z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
            2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
                   z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
            2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
                   2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])));

        vec4 quat_n = vec4(w, x, y, z);
        v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
    }

    inline __device__ void quat_scale_to_preci_half_vjp(
        // fwd inputs
        const vec4 quat,
        const vec3 scale,
        // precompute
        const mat3 R,
        // grad outputs
        const mat3 v_M, // M is perci_half
        // grad inputs
        vec4& v_quat,
        vec3& v_scale) {
        float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
        float sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

        // M = R * S
        mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
        mat3 v_R = v_M * S;

        // grad for (quat, scale) from preci
        quat_to_rotmat_vjp(quat, v_R, v_quat);

        v_scale[0] +=
            -sx * sx *
            (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
        v_scale[1] +=
            -sy * sy *
            (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
        v_scale[2] +=
            -sz * sz *
            (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
    }

    ///////////////////////////////
    // Misc
    ///////////////////////////////

    inline __device__ void
    inverse_vjp(const mat2 Minv, const mat2 v_Minv, mat2& v_M) {
        // P = M^-1
        // df/dM = -P * df/dP * P
        v_M += -Minv * v_Minv * Minv;
    }

    inline __device__ float
    add_blur(const float eps2d, mat2& covar, float& compensation) {
        float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
        covar[0][0] += eps2d;
        covar[1][1] += eps2d;
        float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
        compensation = sqrt(max(0.f, det_orig / det_blur));
        return det_blur;
    }

    inline __device__ vec3 safe_normalize(vec3 v) {
        const float l = v.x * v.x + v.y * v.y + v.z * v.z;
        return l > 0.0f ? (v * rsqrtf(l)) : v;
    }

    inline __device__ vec3 safe_normalize_bw(const vec3& v, const vec3& d_out) {
        const float l = v.x * v.x + v.y * v.y + v.z * v.z;
        if (l > 0.0f) {
            const float il = rsqrtf(l);
            const float il3 = (il * il * il);
            return il * d_out - il3 * glm::dot(d_out, v) * v;
        }
        return d_out;
    }

    inline __device__ float3 load_float3(const float* p) {
        return *reinterpret_cast<const float3*>(p);
    }

    inline __device__ float4 load_float4(const float* p) {
        return *reinterpret_cast<const float4*>(p);
    }

    // MaxDynamicSharedMemorySize is a ceiling: set it once per kernel
    // instantiation when a launch needs more than the last recorded size.
    // Failure throws so the caller cannot silently skip rendering.
    template <typename Kernel>
    inline void set_kernel_max_dynamic_smem(Kernel kernel, const int smem, const char* what) {
        static int max_set = 0;
        if (smem <= max_set) {
            return;
        }
        // hipFuncSetAttribute takes the kernel as const void* only; CUDA has the
        // same overload next to its templated one, so the cast is portable.
        const auto err = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        lfs::core::ensure_cuda_success(
            err,
            "cudaFuncSetAttribute(dynamic shared memory)",
            lfs::core::detail::format_cuda_safe("{} requested_bytes={}", what, smem),
            LFS_SOURCE_SITE_CURRENT());
        max_set = smem;
    }

} // namespace gsplat_lfs