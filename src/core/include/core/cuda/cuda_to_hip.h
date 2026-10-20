/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CUDA -> HIP compatibility shim for the ROCm/AMD build of the LichtFeld-Studio
 * GPU COMPUTE tranche (tensor library, training kernels, gsplat/fastgs
 * rasterizer backends). This is the ONLY file in the compute tree that knows
 * about HIP: it is force-included on every HIP translation unit (see the
 * top-level CMakeLists `USE_HIP` block, CMAKE_HIP_FLAGS -include ...). On a
 * CUDA build this header is never reached and the NVIDIA path is unchanged.
 *
 * Authored with the assistance of Claude (Anthropic).
 */
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

/* Pull libc host declarations in BEFORE the HIP runtime: inside a .cu compiled
 * as HIP, host-side memcpy/memset/abs can otherwise resolve to HIP's __device__
 * overloads once <hip/hip_runtime.h> is in scope, breaking host helpers. */
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

/* ---- CUDA runtime / cuRAND symbol aliases (only what the tranche uses) ----
 * Names taken verbatim from torch/utils/hipify/cuda_to_hip_mappings.py. */
#define cudaDevAttrComputeCapabilityMajor           hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor           hipDeviceAttributeComputeCapabilityMinor
#define cudaDeviceGetAttribute                      hipDeviceGetAttribute
#define cudaDeviceGetDefaultMemPool                 hipDeviceGetDefaultMemPool
#define cudaDeviceGetPCIBusId                       hipDeviceGetPCIBusId
#define cudaDeviceProp                              hipDeviceProp_t
#define cudaDeviceReset                             hipDeviceReset
#define cudaDeviceSynchronize                       hipDeviceSynchronize
#define cudaDriverGetVersion                        hipDriverGetVersion
#define cudaErrorAssert                             hipErrorAssert
#define cudaErrorContextIsDestroyed                 hipErrorContextIsDestroyed
#define cudaErrorCudartUnloading                    hipErrorDeinitialized
#define cudaErrorECCUncorrectable                   hipErrorECCNotCorrectable
#define cudaErrorIllegalAddress                     hipErrorIllegalAddress
#define cudaErrorInitializationError                hipErrorInitializationError
#define cudaErrorInsufficientDriver                 hipErrorInsufficientDriver
#define cudaErrorInvalidConfiguration               hipErrorInvalidConfiguration
#define cudaErrorInvalidDevice                      hipErrorInvalidDevice
#define cudaErrorInvalidMemcpyDirection             hipErrorInvalidMemcpyDirection
#define cudaErrorInvalidResourceHandle              hipErrorInvalidResourceHandle
#define cudaErrorInvalidValue                       hipErrorInvalidValue
#define cudaErrorLaunchFailure                      hipErrorLaunchFailure
#define cudaErrorLaunchTimeout                      hipErrorLaunchTimeOut
#define cudaErrorMemoryAllocation                   hipErrorMemoryAllocation
#define cudaErrorNoDevice                           hipErrorNoDevice
#define cudaErrorNotReady                           hipErrorNotReady
#define cudaErrorStreamCaptureUnsupported           hipErrorStreamCaptureUnsupported
#define cudaErrorUnknown                            hipErrorUnknown
#define cudaError_t                                 hipError_t
#define cudaEventCreate                             hipEventCreate
#define cudaEventCreateWithFlags                    hipEventCreateWithFlags
#define cudaEventDefault                            hipEventDefault
#define cudaEventElapsedTime                        hipEventElapsedTime
#define cudaEventDestroy                            hipEventDestroy
#define cudaEventDisableTiming                      hipEventDisableTiming
#define cudaEventQuery                              hipEventQuery
#define cudaEventRecord                             hipEventRecord
#define cudaEventSynchronize                        hipEventSynchronize
#define cudaEvent_t                                 hipEvent_t
#define cudaExternalSemaphoreWaitParams             hipExternalSemaphoreWaitParams
#define cudaExternalSemaphore_t                     hipExternalSemaphore_t
#define cudaFree                                    hipFree
#define cudaFreeAsync                               hipFreeAsync
#define cudaFreeHost                                hipHostFree
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncSetAttribute                        hipFuncSetAttribute
#define cudaGetDevice                               hipGetDevice
#define cudaGetDeviceCount                          hipGetDeviceCount
#define cudaGetDeviceProperties                     hipGetDeviceProperties
#define cudaGetErrorName                            hipGetErrorName
#define cudaGetErrorString                          hipGetErrorString
#define cudaGetLastError                            hipGetLastError
#define cudaHostAlloc                               hipHostMalloc
#define cudaHostAllocDefault                        hipHostMallocDefault
#define cudaLaunchHostFunc                          hipLaunchHostFunc
#define cudaMalloc                                  hipMalloc
#define cudaMallocAsync                             hipMallocAsync
#define cudaMallocHost                              hipHostMalloc
#define cudaMallocManaged                           hipMallocManaged
#define cudaMemGetInfo                              hipMemGetInfo
#define cudaMemPoolAttrReleaseThreshold             hipMemPoolAttrReleaseThreshold
#define cudaMemPoolAttrReservedMemCurrent           hipMemPoolAttrReservedMemCurrent
#define cudaMemPoolAttrUsedMemCurrent               hipMemPoolAttrUsedMemCurrent
#define cudaMemPoolGetAttribute                     hipMemPoolGetAttribute
#define cudaMemPoolSetAttribute                     hipMemPoolSetAttribute
#define cudaMemPoolTrimTo                           hipMemPoolTrimTo
#define cudaMemPool_t                               hipMemPool_t
#define cudaMemcpy                                  hipMemcpy
#define cudaMemcpy2DAsync                           hipMemcpy2DAsync
#define cudaMemcpyAsync                             hipMemcpyAsync
#define cudaMemcpyDeviceToDevice                    hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost                      hipMemcpyDeviceToHost
#define cudaMemcpyFromSymbol                        hipMemcpyFromSymbol
#define cudaMemcpyHostToDevice                      hipMemcpyHostToDevice
#define cudaMemcpyKind                              hipMemcpyKind
#define cudaMemcpyToSymbol                          hipMemcpyToSymbol
#define cudaMemoryType                              hipMemoryType
#define cudaMemoryTypeDevice                        hipMemoryTypeDevice
#define cudaMemoryTypeHost                          hipMemoryTypeHost
#define cudaMemoryTypeManaged                       hipMemoryTypeManaged
#define cudaMemoryTypeUnregistered                  hipMemoryTypeUnregistered
#define cudaMemset                                  hipMemset
#define cudaMemsetAsync                             hipMemsetAsync
#define cudaPeekAtLastError                         hipPeekAtLastError
#define cudaPointerAttributes                       hipPointerAttribute_t
#define cudaPointerGetAttributes                    hipPointerGetAttributes
#define cudaRuntimeGetVersion                       hipRuntimeGetVersion
#define cudaSetDevice                               hipSetDevice
#define cudaStreamCreate                            hipStreamCreate
#define cudaStreamCreateWithFlags                   hipStreamCreateWithFlags
#define cudaStreamDestroy                           hipStreamDestroy
#define cudaStreamNonBlocking                       hipStreamNonBlocking
#define cudaStreamQuery                             hipStreamQuery
#define cudaStreamSynchronize                       hipStreamSynchronize
#define cudaStreamWaitEvent                         hipStreamWaitEvent
#define cudaStream_t                                hipStream_t
#define cudaSuccess                                 hipSuccess
#define cudaWaitExternalSemaphoresAsync             hipWaitExternalSemaphoresAsync
#define curand                                      hiprand
#define curandCreateGenerator                       hiprandCreateGenerator
#define curandDestroyGenerator                      hiprandDestroyGenerator
#define curandGenerateNormal                        hiprandGenerateNormal
#define curandGenerator_t                           hiprandGenerator_t
#define curandSetGeneratorOffset                    hiprandSetGeneratorOffset
#define curandSetPseudoRandomGeneratorSeed          hiprandSetPseudoRandomGeneratorSeed
#define curandSetStream                             hiprandSetStream
#define curandState                                 hiprandState_t
#define curandStatePhilox4_32_10_t                  hiprandStatePhilox4_32_10_t
#define curandStatus_t                              hiprandStatus_t
#define curand_init                                 hiprand_init
#define curand_normal                               hiprand_normal
#define curand_normal4                              hiprand_normal4
#define curand_uniform                              hiprand_uniform
#define CURAND_RNG_PSEUDO_PHILOX4_32_10             HIPRAND_RNG_PSEUDO_PHILOX4_32_10
#define CURAND_STATUS_SUCCESS                       HIPRAND_STATUS_SUCCESS

/* ---- libtorch (ROCm) interop ----
 * The ROCm libtorch the compute gtests link against ships c10/cuda headers that
 * reference these stream-capture / CUDA-graph / IPC-event symbols literally
 * (torch hipifies them at CUDAExtension build time, which a plain CMake link
 * does not do). Alias them so c10/cuda/CUDAStream.h etc. parse under hipcc. */
#define cudaDeviceGetStreamPriorityRange            hipDeviceGetStreamPriorityRange
#define cudaEventRecordDefault                      hipEventRecordDefault
#define cudaEventRecordExternal                     hipEventRecordExternal
#define cudaEventRecordWithFlags                    hipEventRecordWithFlags
#define cudaEventWaitDefault                        hipEventWaitDefault
#define cudaEventWaitExternal                       hipEventWaitExternal
#define cudaGraphRetainUserObject                   hipGraphRetainUserObject
#define cudaGraphUserObjectMove                     hipGraphUserObjectMove
#define cudaGraph_t                                 hipGraph_t
#define cudaHostFn_t                                hipHostFn_t
#define cudaIpcEventHandle_t                        hipIpcEventHandle_t
#define cudaIpcGetEventHandle                       hipIpcGetEventHandle
#define cudaIpcOpenEventHandle                      hipIpcOpenEventHandle
#define cudaStreamBeginCapture                      hipStreamBeginCapture
#define cudaStreamCaptureMode                       hipStreamCaptureMode
#define cudaStreamCaptureStatus                     hipStreamCaptureStatus
#define cudaStreamCaptureStatusActive               hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated          hipStreamCaptureStatusInvalidated
#define cudaStreamCaptureStatusNone                 hipStreamCaptureStatusNone
#define cudaStreamGetCaptureInfo                    hipStreamGetCaptureInfo
#define cudaStreamGetCaptureInfo_v2                 hipStreamGetCaptureInfo_v2
#define cudaStreamGetPriority                       hipStreamGetPriority
#define cudaStreamIsCapturing                       hipStreamIsCapturing
#define cudaThreadExchangeStreamCaptureMode         hipThreadExchangeStreamCaptureMode
#define cudaUserObjectCreate                        hipUserObjectCreate
#define cudaUserObjectNoDestructorSync              hipUserObjectNoDestructorSync
#define cudaUserObjectRelease                       hipUserObjectRelease
#define cudaUserObject_t                            hipUserObject_t

/* ---- CUDA Driver API (VMM) -> HIP ----
 * memory_arena.cu uses the low-level virtual-memory-management driver API
 * (cuMemCreate/Map/SetAccess/AddressReserve, CUmemGenericAllocationHandle,
 * CUdeviceptr, ...) which all have hip* driver equivalents. */
#define CUDA_ERROR_OUT_OF_MEMORY                         hipErrorOutOfMemory
#define CUDA_SUCCESS                                     hipSuccess
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE               hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOCATION_TYPE_PINNED                    hipMemAllocationTypePinned
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM                 hipMemAllocationGranularityMinimum
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR         hipMemHandleTypePosixFileDescriptor
#define CU_MEM_LOCATION_TYPE_DEVICE                      hipMemLocationTypeDevice
#define CUevent_st                                       ihipEvent_t
#define CUstream_st                                      ihipStream_t
#define CUdevice_attribute                               hipDeviceAttribute_t
#define CUdeviceptr                                      hipDeviceptr_t
#define CUmemAccessDesc                                  hipMemAccessDesc
#define CUmemAllocationHandleType                        hipMemAllocationHandleType
#define CUmemAllocationProp                              hipMemAllocationProp
#define CUmemGenericAllocationHandle                     hipMemGenericAllocationHandle_t
#define CUresult                                         hipError_t
#define cuDeviceGetAttribute                             hipDeviceGetAttribute
#define cuGetErrorName                                   hipDrvGetErrorName
#define cuGetErrorString                                 hipDrvGetErrorString
#define cuMemAddressFree                                 hipMemAddressFree
#define cuMemAddressReserve                              hipMemAddressReserve
#define cuMemCreate                                      hipMemCreate
#define cuMemExportToShareableHandle                     hipMemExportToShareableHandle
#define cuMemGetAllocationGranularity                    hipMemGetAllocationGranularity
#define cuMemMap                                         hipMemMap
#define cuMemRelease                                     hipMemRelease
#define cuMemSetAccess                                   hipMemSetAccess
#define cuMemUnmap                                       hipMemUnmap

/* CUDA error codes with no HIP counterpart. The error classifiers name them in
 * switch labels; HIP never returns them, so give each its CUDA value, which no
 * hipError_t enumerator uses, to keep the labels distinct and unreachable. */
#define cudaErrorCompatNotSupportedOnDevice static_cast<hipError_t>(804)
#define cudaErrorDevicesUnavailable         static_cast<hipError_t>(46)
#define cudaErrorHardwareStackError         static_cast<hipError_t>(714)
#define cudaErrorIllegalInstruction         static_cast<hipError_t>(715)
#define cudaErrorInvalidAddressSpace        static_cast<hipError_t>(717)
#define cudaErrorInvalidPc                  static_cast<hipError_t>(718)
#define cudaErrorMisalignedAddress          static_cast<hipError_t>(716)
#define cudaErrorStartupFailure             static_cast<hipError_t>(127)
#define cudaErrorSystemDriverMismatch       static_cast<hipError_t>(803)
#define cudaErrorSystemNotReady             static_cast<hipError_t>(802)

/* __grid_constant__ only marks a const kernel parameter as read in place; HIP
 * passes kernel arguments the same way without it. */
#define __grid_constant__
#define __trap() __builtin_trap()

/* fp16/bf16: HIP provides __half/__half2 (hip_fp16.h) as drop-ins. The CUDA
 * bfloat16 scalar __nv_bfloat16 maps to HIP's __hip_bfloat16. */
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162

/* ---- __CUDA_ARCH__ during the HIP device pass ----
 * The tensor code uses `#ifdef __CUDA_ARCH__` to pick device intrinsics (powf,
 * fmodf, __ldg, ...) over the host std fallbacks. HIP uses __HIP_DEVICE_COMPILE__
 * for that, not __CUDA_ARCH__, so without this every such guard takes the HOST
 * branch on the HIP device pass (e.g. std::pow in device code -> "ambiguous").
 * Define __CUDA_ARCH__ ONLY in the HIP device pass; it is 0/undefined on the
 * host pass, so rocThrust's host config (which breaks under a global __CUDACC__)
 * is unaffected. (PORTING_GUIDE: cudaKDTree/MPPI __CUDA_ARCH__ lesson.) */
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 1
#endif
#endif

/* ---- Warp size abstraction (NEVER hardcode 32) ----
 * CDNA (gfx90a/gfx94x, __GFX9__) wavefront = 64; RDNA (gfx10xx/gfx11xx) = 32;
 * CUDA = 32. Mirrors PyTorch's C10_WARP_SIZE. __GFX9__ is defined only during
 * device compilation; host code that needs the value queries prop.warpSize. */
#if defined(__GFX9__)
#define LFS_WARP_SIZE 64
#else
#define LFS_WARP_SIZE 32
#endif
namespace lfs {
#if defined(__GFX9__)
    static constexpr int kWarpSize = 64;
#else
    static constexpr int kWarpSize = 32;
#endif
}

/* Full-warp lane mask for __shfl_*_sync / __ballot_sync. ROCm static_asserts a
 * 64-bit mask (sizeof(MaskT)==8) regardless of active wave width, so the CUDA
 * literal 0xffffffff fails to COMPILE on HIP. */
#define LFS_FULL_WARP_MASK 0xffffffffffffffffULL

/* ---- GLM in device code ----
 * The build uses GLM 1.0.1 (vendored), which has a NATIVE GLM_COMPILER_HIP path
 * (glm/detail/setup.hpp keys on __HIPCC__) and emits __device__ __host__ on its
 * math under hipcc with no extra steering. So, unlike GLM 0.9.9.x (which needs
 * __CUDACC__), we deliberately do NOT define __CUDACC__ / CUDA_VERSION /
 * GLM_FORCE_CUDA here: defining __CUDACC__ would make rocThrust take its broken
 * CUDA-system path (<cub/detail/detect_cuda_runtime.cuh> not found). glm::dot in
 * a __device__ fn compiles clean on hipcc with 1.0.1 and no macros.
 * (PORTING_GUIDE: 3DUNDERWORLD GLM lesson -- newer GLM needs no __CUDACC__.) */

/* ---- NVTX profiling ranges -> no-op on ROCm ----
 * The profiling ranges are not part of GPU correctness; map to no-ops so the
 * <nvtx3/nvToolsExt.h> shim's call sites compile without roctracer linkage. */
#define nvtxRangePush(msg) ((void)0)
#define nvtxRangePushA(msg) ((void)0)
#define nvtxRangePop()     ((void)0)

#else /* CUDA build: real toolkit headers, unchanged. */
#include <cuda_runtime.h>
#endif
