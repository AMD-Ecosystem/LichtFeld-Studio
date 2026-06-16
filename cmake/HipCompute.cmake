# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ROCm/HIP compute-only build (linux-gfx90a). Builds the
# libtorch-free GPU COMPUTE tranche -- tensor library, training kernels, and the
# gsplat/fastgs/edge rasterizer backends -- plus a focused GPU gtest binary
# (lfs_compute_tests). The full desktop app (GUI/Vulkan-interop/Python/MCP/USD/
# nvImageCodec) is DEFERRED and never configured here. The top-level CMakeLists
# returns immediately after including this file, so the NVIDIA path is untouched.
#
# Authored with the assistance of Claude (Anthropic).

# ---- HIP language + target architecture --------------------------------------
# HIP is enabled at project() time in the top-level CMakeLists (LANGUAGES HIP).
# Never hardcode the default arch as a literal: a literal would override
# -DCMAKE_HIP_ARCHITECTURES and force every other arch (gfx1100, gfx1151) to edit
# this file. Default the arch only when the cache var is unset.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
endif()
message(STATUS "[ROCm] Building GPU-compute tranche for HIP arch: ${CMAKE_HIP_ARCHITECTURES}")

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
# __HIP_PLATFORM_AMD__ makes <hip/hip_runtime.h> expose the AMD runtime
# declarations on plain C++ TUs too (the HIP toolchain auto-defines it only for
# .cu / -x hip). Many host .cpp in lfs_core/lfs_diagnostics include
# <cuda_runtime.h> -> the compat header -> the HIP runtime, and need the symbols.
add_compile_definitions(USE_HIP=1 __HIP_PLATFORM_AMD__=1)
if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/include"
    "${ROCM_PATH}/include")

# ---- Retag every .cu as LANGUAGE HIP without editing leaf CMakeLists ----------
# Each compute subdir globs/lists *.cu into add_library; override add_library and
# add_executable at this scope to forward to the real command, then retag the .cu
# sources LANGUAGE HIP and set HIP_ARCHITECTURES (skip ALIAS/IMPORTED/INTERFACE).
# (PORTING_GUIDE: MPPI top-scope override lesson.)
function(_lfs_retag_hip_sources target)
    get_target_property(_t ${target} TYPE)
    if(_t STREQUAL "INTERFACE_LIBRARY")
        return()
    endif()
    get_target_property(_srcs ${target} SOURCES)
    if(NOT _srcs)
        return()
    endif()
    foreach(_s IN LISTS _srcs)
        if(_s MATCHES "\\.cu$")
            set_source_files_properties(${_s} PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
    set_target_properties(${target} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endfunction()

function(add_library target)
    _add_library(${target} ${ARGN})
    if(NOT "${target}" MATCHES "::" )
        list(FIND ARGN "IMPORTED" _imp)
        list(FIND ARGN "ALIAS" _ali)
        list(FIND ARGN "INTERFACE" _int)
        if(_imp EQUAL -1 AND _ali EQUAL -1 AND _int EQUAL -1)
            _lfs_retag_hip_sources(${target})
        endif()
    endif()
endfunction()

function(add_executable target)
    _add_executable(${target} ${ARGN})
    list(FIND ARGN "IMPORTED" _imp)
    list(FIND ARGN "ALIAS" _ali)
    if(_imp EQUAL -1 AND _ali EQUAL -1)
        _lfs_retag_hip_sources(${target})
    endif()
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
# actually drives codegen and is set by the override above).
set(LichtFeld-Studio_CUDA_ARCH "${CMAKE_HIP_ARCHITECTURES}")

# ---- Project-wide definitions / debug options the leaf CMakeLists expect -----
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
add_definitions(-DPROJECT_ROOT_PATH="${PROJ_ROOT_DIR}")
set(LFS_CUDA_DEVICE_DEBUG_FLAGS "" CACHE INTERNAL "")
option(ENABLE_ALLOCATION_PROFILING "Enable tensor allocation profiling" OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# git_version.h + config.h that the sources include (generated at configure).
execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH_SHORT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT GIT_COMMIT_HASH_SHORT)
    set(GIT_COMMIT_HASH_SHORT "unknown")
endif()
set(GIT_TAGGED_VERSION "${GIT_COMMIT_HASH_SHORT}")
file(WRITE "${CMAKE_BINARY_DIR}/include/git_version.h"
"#pragma once\n#define GIT_COMMIT_HASH_SHORT \"${GIT_COMMIT_HASH_SHORT}\"\n#define GIT_TAGGED_VERSION \"${GIT_TAGGED_VERSION}\"\n")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/config.h" @ONLY)

# ---- Dependencies (no vcpkg: glm vendored, gtest conda, libtorch ROCm) -------
# Use GLM 1.0.1 (vendored via -DLFS_GLM_INCLUDE_DIR), NOT the system 0.9.9.8:
# 1.0.1 has a native GLM_COMPILER_HIP path so glm:: math compiles in __device__
# code under hipcc with no __CUDACC__ hack (which would break rocThrust). Build
# the glm::glm target by hand pointing at the vendored headers.
if(NOT LFS_GLM_INCLUDE_DIR OR NOT EXISTS "${LFS_GLM_INCLUDE_DIR}/glm/glm.hpp")
    message(FATAL_ERROR "Pass -DLFS_GLM_INCLUDE_DIR=<dir containing glm/glm.hpp> (GLM >= 1.0.1)")
endif()
if(NOT TARGET glm::glm)
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LFS_GLM_INCLUDE_DIR}")
endif()
find_package(Threads REQUIRED)
find_package(OpenMP)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(TBB REQUIRED)
# lfs_core's Camera/Scene/image_io retain their CPU image + mesh paths; supply
# the libs (installed via apt; vcpkg is not bootstrapped). The shipped binary's
# GUI/io codec stack (FFmpeg/WebP/nvjpeg/USD/assimp) stays DEFERRED -- only the
# compute libraries and their CPU data-structure deps are built here.
# Construct the OpenImageIO target by hand: the apt -dev package's CMake config
# references CLI tool binaries (iconvert/...) that the package omits, which trips
# find_package. We only need the lib + headers for image_io.cpp.
# On Windows, image_io.cpp and mesh_data.cpp are replaced by stubs that do not
# include OIIO/OpenMesh headers, so both libraries are skipped on that platform.
if(NOT WIN32)
    # OpenImageIO: on Linux use the apt-installed shared lib.
    if(LFS_OIIO_LIB AND LFS_OIIO_INC)
        set(OIIO_LIB "${LFS_OIIO_LIB}")
        set(OIIO_INC "${LFS_OIIO_INC}")
    else()
        find_library(OIIO_LIB NAMES OpenImageIO HINTS /usr/lib/x86_64-linux-gnu REQUIRED)
        find_path(OIIO_INC NAMES OpenImageIO/imageio.h HINTS /usr/include REQUIRED)
    endif()
    if(NOT TARGET OpenImageIO::OpenImageIO)
        add_library(OpenImageIO::OpenImageIO INTERFACE IMPORTED)
        set_target_properties(OpenImageIO::OpenImageIO PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OIIO_INC}"
            INTERFACE_LINK_LIBRARIES "${OIIO_LIB}")
    endif()
    # OpenMesh: on Linux the apt package ships OpenMeshCore (shared).
    if(LFS_OPENMESH_LIB AND LFS_OPENMESH_INC)
        set(OPENMESH_CORE_LIB "${LFS_OPENMESH_LIB}")
        set(OPENMESH_INC "${LFS_OPENMESH_INC}")
    else()
        find_library(OPENMESH_CORE_LIB NAMES OpenMeshCore HINTS /usr/lib/x86_64-linux-gnu REQUIRED)
        find_path(OPENMESH_INC NAMES OpenMesh/Core/Mesh/ArrayKernel.hh HINTS /usr/include REQUIRED)
    endif()
    foreach(_omt OpenMeshCore OpenMeshCoreStatic)
        if(NOT TARGET ${_omt})
            add_library(${_omt} INTERFACE IMPORTED)
            set_target_properties(${_omt} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${OPENMESH_INC}"
                INTERFACE_LINK_LIBRARIES "${OPENMESH_CORE_LIB}")
        endif()
    endforeach()
endif()
# Taywee/args (header-only CLI parser, used only by argument_parser.cpp). No apt
# package; the header is supplied via -DLFS_ARGS_INCLUDE_DIR (vendored here).
if(NOT TARGET taywee::args)
    find_path(ARGS_INCLUDE_DIR args.hxx HINTS ${LFS_ARGS_INCLUDE_DIR} /usr/include /usr/local/include)
    if(NOT ARGS_INCLUDE_DIR)
        message(FATAL_ERROR "args.hxx not found; pass -DLFS_ARGS_INCLUDE_DIR=<dir containing args.hxx>")
    endif()
    add_library(taywee::args INTERFACE IMPORTED)
    set_target_properties(taywee::args PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ARGS_INCLUDE_DIR}")
endif()
# GLM device-code steering also needs GLM to NOT clamp z to [0,1] differently
# than the kernels expect; keep its default (the GUI's GLM_FORCE_DEPTH_ZERO_TO_ONE
# is a viewer convention, irrelevant to the compute math).

# ---- Compute libraries (the tranche) -----------------------------------------
add_subdirectory(src/diagnostics)
add_subdirectory(src/geometry)
add_subdirectory(src/core)
add_subdirectory(src/training)

# ---- Focused GPU gtest binary ------------------------------------------------
if(BUILD_TESTS)
    add_subdirectory(cmake/hip_tests)
endif()

message(STATUS "[ROCm] Compute-only configuration complete.")
