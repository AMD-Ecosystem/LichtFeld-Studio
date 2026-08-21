/* Compatibility shim: on Windows ROCm, TheRock's PyTorch build ships a
 * non-hipified c10/cuda/CUDACachingAllocator.h whose headers pull in the real
 * <cuda_runtime_api.h> toolkit chain, so it cannot be included directly here.
 * The HIP build instead needs the hipified counterpart, c10/hip/HIPCachingAllocator.h
 * (an auto-generated-by-hipify file); on a current TheRock build that header
 * already opens `namespace c10::cuda::CUDACachingAllocator` itself with the
 * hipified content, so simply including it is enough -- no separate namespace
 * alias is declared (declaring one here would redefine that namespace).
 * On Linux ROCm, torch is already hipified and c10::cuda::CUDACachingAllocator
 * exists; include the real header unchanged. */
#pragma once

#if defined(_WIN32)
// Windows: the hipified header defines c10::cuda::CUDACachingAllocator itself.
#include <c10/hip/HIPCachingAllocator.h>
#else
// Linux: ROCm torch is already hipified; include the real header unchanged.
#include_next <c10/cuda/CUDACachingAllocator.h>
#endif
