/* MOAT shim: on Windows ROCm, the TheRock PyTorch build ships a non-hipified
 * c10/cuda/CUDACachingAllocator.h whose extern symbols live only in the CUDA
 * c10 build. The HIP equivalents are in c10/hip/HIPCachingAllocator.h.
 * On Linux ROCm, torch is already hipified and c10::cuda::CUDACachingAllocator
 * exists. Apply the redirect and namespace alias ONLY on Windows. */
#pragma once

#if defined(_WIN32)
// Windows: redirect to HIP header and create the cuda namespace alias.
#include <c10/hip/HIPCachingAllocator.h>

namespace c10 {
namespace cuda {
// Expose the HIP caching allocator under the CUDA namespace alias so existing
// call-sites that use c10::cuda::CUDACachingAllocator:: work on ROCm without edits.
namespace CUDACachingAllocator = c10::hip::HIPCachingAllocator;
} // namespace cuda
} // namespace c10
#else
// Linux: ROCm torch is already hipified; include the real header unchanged.
#include_next <c10/cuda/CUDACachingAllocator.h>
#endif
