# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# HIP compiler and CUDA-compatibility setup shared by the ROCm builds. Included
# by the top-level CMakeLists after project(... LANGUAGES HIP CXX C), in place
# of the CUDA toolkit setup, and by cmake/HipCompute.cmake.

message(STATUS "[ROCm] HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Force-include the single CUDA->HIP compat header on every HIP translation unit
# so its symbol aliases / kWarpSize / GLM steering precede any source include,
# regardless of each file's include order. USE_HIP reaches device + host passes.
set(_LFS_COMPAT_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include/core/cuda/cuda_to_hip.h")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -DUSE_HIP=1 -include ${_LFS_COMPAT_HEADER}")
# --extended-lambda has no hipcc equivalent (lambdas in device code just work);
# the relaxed-constexpr behaviour is the clang default. Quiet the GLM 0.9.9.x
# "CUDA compiler detected" pragma-message spam from the forced __CUDACC__ path.
# The flag is added via CMAKE_HIP_FLAGS (a string) rather than add_compile_options
# so the embedded '#' is not parsed as a CMake comment.
string(APPEND CMAKE_HIP_FLAGS " -Wno-#pragma-messages -Wno-unused-result")
# The project's logger CUDA path uses printf-style snprintf with the project's
# own format strings; clang's -Wnon-pod-varargs / -Wformat-security are errors by
# default on the HIP toolchain where nvcc was lenient. The behaviour matches the
# CUDA build; demote to non-fatal.
string(APPEND CMAKE_HIP_FLAGS " -Wno-non-pod-varargs -Wno-format-security")
# Without this, clang's HIP headers declare host min/max for int only, so min or
# max of floats on the host side of a __host__ __device__ function (the camera
# model constructors) silently converts to int. CUDA declares float, double and
# wider integer overloads for host and device; this enables clang's. Its math
# header is pre-included, so the define cannot come from a header.
string(APPEND CMAKE_HIP_FLAGS " -D__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__=1")
# __HIP_PLATFORM_AMD__ makes <hip/hip_runtime.h> expose the AMD runtime
# declarations on plain C++ TUs too (the HIP toolchain auto-defines it only for
# .cu / -x hip). Many host .cpp in lfs_core/lfs_diagnostics include
# <cuda_runtime.h> -> the compat header -> the HIP runtime, and need the symbols.
add_compile_definitions(USE_HIP=1 __HIP_PLATFORM_AMD__=1)
if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
    # GNU-style rules for clang++, as in LibTorch's LoadHIP.cmake.
    set(CMAKE_HIP_COMPILE_OBJECT
        "<CMAKE_HIP_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -x hip -c <SOURCE>")
    set(CMAKE_INCLUDE_SYSTEM_FLAG_HIP "-isystem ")
    set(CMAKE_HIP_DEPFILE_FORMAT gcc)
    set(CMAKE_DEPFILE_FLAGS_HIP "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
    # Without -fgpu-rdc HIP objects are plain COFF, so link them like C++.
    set(CMAKE_HIP_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")
    set(CMAKE_HIP_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_LIBRARY}")
    set(CMAKE_HIP_CREATE_SHARED_MODULE "${CMAKE_CXX_CREATE_SHARED_MODULE}")
    set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreaded -fms-runtime-lib=static)
    set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDLL -fms-runtime-lib=dll)
    set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebug -fms-runtime-lib=static_dbg)
    set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_MultiThreadedDebugDLL -fms-runtime-lib=dll_dbg)
    # The MSVC debug-information options (-Zi, -Z7) are clang-cl only; the GNU
    # per-config flags below already carry -g.
    set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_Embedded "")
    set(CMAKE_HIP_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_ProgramDatabase "")
    unset(CMAKE_HIP_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT_EditAndContinue)
    # With C/CXX on clang-cl, CMake 4.3+ seeds the HIP flags with MSVC-style
    # defaults (/DWIN32 /O2 ...) that clang++ rejects; use GNU-style ones.
    string(REGEX REPLACE "(^| )/[A-Za-z][A-Za-z0-9_:=.]*" "" CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS}")
    set(_lfs_hip_flags_DEBUG "-O0 -g")
    set(_lfs_hip_flags_RELEASE "-O3 -DNDEBUG")
    set(_lfs_hip_flags_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(_lfs_hip_flags_MINSIZEREL "-Os -DNDEBUG")
    foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        if(CMAKE_HIP_FLAGS_${_cfg} MATCHES "(^| )/[A-Za-z]")
            set(CMAKE_HIP_FLAGS_${_cfg} "${_lfs_hip_flags_${_cfg}}")
        endif()
    endforeach()
endif()

# ---- Toolkit shim include dir (HIP build only) -------------------------------
# Forwarding headers named <cuda.h>/<cuda_runtime.h>/<cooperative_groups*>/
# <curand*>/<cub/*>/<device_launch_parameters.h>/<nvtx3/nvToolsExt.h> live here
# and resolve the toolkit angle-includes to their HIP equivalents. On a CUDA
# build this dir is never on the include path.
set(LFS_HIP_COMPAT_INC "${CMAKE_CURRENT_SOURCE_DIR}/src/hip_compat")
# The shims resolve the compat header by its in-tree path core/cuda/cuda_to_hip.h,
# so src/core/include must be globally reachable too (every TU that includes a
# toolkit angle-header, even plain .cpp, transitively pulls the compat header).
# Plain .cpp host TUs that include <cuda_runtime.h> also need the HIP runtime
# headers (/opt/rocm/include) -- the HIP toolchain only auto-adds them for
# .cu/-x hip TUs, not for the C++ compiler.
if(NOT ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()
include_directories(BEFORE
    "${LFS_HIP_COMPAT_INC}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include")
# After the package include directories, so ROCm's bundled copies of
# third-party headers (nlohmann, ...) do not shadow the project's own.
include_directories(AFTER SYSTEM "${ROCM_PATH}/include")

# ---- Retag every .cu as LANGUAGE HIP without editing leaf CMakeLists ----------
# Call once after every subdirectory is added. A post-pass over the targets,
# rather than an add_library/add_executable override, because the vcpkg
# toolchain already overrides both and a second override recurses.
function(lfs_hip_retag_cuda_sources dir)
    get_property(_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_tgt IN LISTS _targets)
        get_target_property(_type ${_tgt} TYPE)
        if(_type STREQUAL "INTERFACE_LIBRARY" OR _type STREQUAL "UTILITY")
            continue()
        endif()
        get_target_property(_srcs ${_tgt} SOURCES)
        get_target_property(_srcdir ${_tgt} SOURCE_DIR)
        set(_cu "")
        foreach(_s IN LISTS _srcs)
            if(_s MATCHES "\\.cu$" AND NOT _s MATCHES "^\\$<")
                cmake_path(ABSOLUTE_PATH _s BASE_DIRECTORY "${_srcdir}" OUTPUT_VARIABLE _abs)
                list(APPEND _cu "${_abs}")
            endif()
        endforeach()
        if(_cu)
            set_source_files_properties(${_cu} TARGET_DIRECTORY ${_tgt} PROPERTIES LANGUAGE HIP)
            set_target_properties(${_tgt} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
        endif()
    endforeach()
    get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(_sub IN LISTS _subdirs)
        lfs_hip_retag_cuda_sources("${_sub}")
    endforeach()
endfunction()

# ---- Stub the CUDA toolkit imported targets with HIP equivalents -------------
# The compute libraries reference CUDA::cudart / CUDA::curand and
# ${CUDAToolkit_INCLUDE_DIRS} / ${LichtFeld-Studio_CUDA_ARCH}. Provide INTERFACE
# stand-ins so those references resolve without editing every leaf: HIP's runtime
# is implicit for HIP TUs, and hipRAND is linked for the RNG users.
find_library(HIPRAND_LIB hiprand HINTS "${ROCM_PATH}/lib" /opt/rocm/lib REQUIRED)
# On Linux, the HIP runtime is linked implicitly for HIP TUs. On Windows, plain
# C++ .cpp files that include <cuda_runtime.h> -> the compat header -> HIP runtime
# must explicitly link amdhip64; satisfy that by making CUDA::cudart carry it.
find_library(AMDHIP64_LIB amdhip64 HINTS "${ROCM_PATH}/lib" /opt/rocm/lib REQUIRED)
if(NOT TARGET CUDA::cudart)
    add_library(CUDA::cudart INTERFACE IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES INTERFACE_LINK_LIBRARIES "${AMDHIP64_LIB}")
endif()
# On Windows with --nostdlib in the HIP DLL link command, compiler-rt builtins
# (float16 helpers: __extendhfsf2/__truncsfhf2/__truncdfhf2) are not pulled in
# automatically. Link the builtins archive explicitly.
if(WIN32)
    find_library(CLANG_RT_BUILTINS clang_rt.builtins-x86_64
        HINTS "${ROCM_PATH}/lib/llvm/lib/clang/23/lib/windows"
              "${ROCM_PATH}/lib/llvm/lib/clang/22/lib/windows"
              "${ROCM_PATH}/lib/llvm/lib/clang/21/lib/windows")
    if(CLANG_RT_BUILTINS)
        link_libraries("${CLANG_RT_BUILTINS}")
        message(STATUS "[ROCm] Windows: linking clang_rt.builtins: ${CLANG_RT_BUILTINS}")
    else()
        message(WARNING "[ROCm] Windows: clang_rt.builtins-x86_64.lib not found -- float16 ops may fail to link")
    endif()
endif()
if(NOT TARGET CUDA::curand)
    add_library(CUDA::curand INTERFACE IMPORTED)
    set_target_properties(CUDA::curand PROPERTIES INTERFACE_LINK_LIBRARIES "${HIPRAND_LIB}")
endif()
if(NOT TARGET CUDA::cuda_driver)
    add_library(CUDA::cuda_driver INTERFACE IMPORTED)
endif()
if(NOT TARGET CUDA::cupti)
    add_library(CUDA::cupti INTERFACE IMPORTED)
endif()
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()
set(CUDAToolkit_INCLUDE_DIRS "" CACHE STRING "" FORCE)
# Some leaf targets set CUDA_ARCHITECTURES "${LichtFeld-Studio_CUDA_ARCH}" on
# their (now HIP) sources; give it a harmless value (HIP_ARCHITECTURES is what
# actually drives codegen and is set by lfs_hip_retag_cuda_sources).
set(LichtFeld-Studio_CUDA_ARCH "${CMAKE_HIP_ARCHITECTURES}")
if(NOT WIN32)
    # Find the HIP runtime libraries without LD_LIBRARY_PATH when ROCm is not in
    # the loader's default path (for example the ROCm Python SDK). The
    # application's RPATH takes the CUDA toolkit library directory.
    set(CUDAToolkit_LIBRARY_DIR "${ROCM_PATH}/lib")
    list(APPEND CMAKE_BUILD_RPATH "${ROCM_PATH}/lib")
endif()
