# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ROCm/HIP compute-only build (-DUSE_HIP=ON -DLFS_HIP_COMPUTE_ONLY=ON). Builds
# the GPU compute libraries -- tensor library, training kernels, and the
# gsplat/fastgs rasterizer backends -- plus a focused GPU gtest binary
# (lfs_compute_tests) from distribution packages, without vcpkg. The top-level
# CMakeLists returns right after including this file.
#
# Authored with the assistance of Claude (Anthropic).

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HipToolchain.cmake")

# Only the compile-only lfs_cuda_diagnostics_compile_check object library names
# Vulkan; it is not part of the compute tests.
if(NOT TARGET Vulkan::Vulkan)
    add_library(Vulkan::Vulkan INTERFACE IMPORTED)
endif()

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

# lfs_core ABI stamp header (lfs_core depends on the lfs_core_abi_stamp target),
# as in the top-level CMakeLists, with the HIP compiler in the device fields.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/InitializeCoreAbiStamp.cmake")
lfs_initialize_core_abi_stamps()
add_custom_target(lfs_core_abi_stamp ALL
    BYPRODUCTS "${LFS_CORE_ABI_HEADER}"
    COMMAND ${CMAKE_COMMAND}
        "-DLFS_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        "-DLFS_CORE_ABI_TEMPLATE=${CMAKE_SOURCE_DIR}/cmake/lfs_core_abi_stamp.h.in"
        "-DLFS_CORE_ABI_HEADER=${LFS_CORE_ABI_HEADER}"
        "-DLFS_PROJECT_VERSION=${PROJECT_VERSION}"
        "-DLFS_CONFIGURED_GIT_COMMIT_HASH_SHORT=${GIT_COMMIT_HASH_SHORT}"
        "-DLFS_BUILD_CONFIG=$<CONFIG>"
        "-DLFS_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
        "-DLFS_SIZEOF_VOID_P=${CMAKE_SIZEOF_VOID_P}"
        "-DLFS_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
        "-DLFS_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}"
        "-DLFS_CUDA_COMPILER_ID=${CMAKE_HIP_COMPILER_ID}"
        "-DLFS_CUDA_COMPILER_VERSION=${CMAKE_HIP_COMPILER_VERSION}"
        "-DLFS_CUDA_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}"
        "-DLFS_VCPKG_TARGET_TRIPLET="
        -P "${CMAKE_SOURCE_DIR}/cmake/GenerateCoreAbiStamp.cmake"
    COMMENT "Refreshing lfs_core ABI stamp"
    VERBATIM
)
add_compile_definitions(
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:RELEASE_BUILD>
)

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
# lfs_logger hashes with xxHash. Use it header-only (XXH_INLINE_ALL) when no
# CMake package is installed, so only xxhash.h is needed.
find_package(xxHash CONFIG QUIET)
if(NOT TARGET xxHash::xxhash)
    find_path(LFS_XXHASH_INCLUDE_DIR xxhash.h REQUIRED)
    add_library(xxHash::xxhash INTERFACE IMPORTED)
    set_target_properties(xxHash::xxhash PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LFS_XXHASH_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS XXH_INLINE_ALL)
endif()
# lfs_core keeps its CPU image codec and mesh paths. On Linux supply the codec
# libraries (distribution packages; vcpkg is not bootstrapped) and OpenMesh. On
# Windows image_codecs.cpp, image_exr.cpp and mesh_data.cpp are replaced by stubs
# (src/core/CMakeLists.txt), so the targets are empty stand-ins there.
if(NOT WIN32)
    find_package(JPEG REQUIRED)
    find_package(PNG REQUIRED)
    find_package(TIFF REQUIRED)
    find_package(WebP CONFIG QUIET)
    if(NOT TARGET WebP::webp)
        find_library(LFS_WEBP_LIB NAMES webp REQUIRED)
        find_path(LFS_WEBP_INC NAMES webp/decode.h REQUIRED)
        add_library(WebP::webp INTERFACE IMPORTED)
        set_target_properties(WebP::webp PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LFS_WEBP_INC}"
            INTERFACE_LINK_LIBRARIES "${LFS_WEBP_LIB}")
    endif()
    # src/core/image_exr.cpp calls exr_get_chunk_table_offset, new in OpenEXR 3.3.
    find_package(OpenEXR 3.3 CONFIG REQUIRED)
    # OpenMesh: on Linux the apt package ships OpenMeshCore (shared).
    if(LFS_OPENMESH_LIB AND LFS_OPENMESH_INC)
        set(OPENMESH_CORE_LIB "${LFS_OPENMESH_LIB}")
        set(OPENMESH_INC "${LFS_OPENMESH_INC}")
    else()
        find_library(OPENMESH_CORE_LIB NAMES OpenMeshCore HINTS /usr/lib/x86_64-linux-gnu REQUIRED)
        find_path(OPENMESH_INC NAMES OpenMesh/Core/Mesh/ArrayKernel.hh HINTS /usr/include REQUIRED)
    endif()
    if(NOT TARGET OpenMeshCore)
        add_library(OpenMeshCore INTERFACE IMPORTED)
        set_target_properties(OpenMeshCore PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OPENMESH_INC}"
            INTERFACE_LINK_LIBRARIES "${OPENMESH_CORE_LIB}")
    endif()
else()
    foreach(_lfs_stub JPEG::JPEG PNG::PNG TIFF::TIFF WebP::webp OpenEXR::OpenEXRCore OpenMeshCore)
        if(NOT TARGET ${_lfs_stub})
            add_library(${_lfs_stub} INTERFACE IMPORTED)
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

# The io library is not built here. A few compute sources (MRNF, the depth
# anchor cache) include its lightweight headers, so expose those only.
add_library(lfs_io INTERFACE)
target_include_directories(lfs_io INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src/io/include")

# ---- Compute libraries (the tranche) -----------------------------------------
add_subdirectory(src/diagnostics)
add_subdirectory(src/geometry)
add_subdirectory(src/core)
# std::stacktrace in libstdc++ lives in libstdc++exp. Upstream links it for GCC;
# the ROCm clang uses the same libstdc++ on Linux, so link it when it works.
if(UNIX AND NOT APPLE)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-std=c++23")
    set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
    check_cxx_source_compiles("#include <stacktrace>
int main() { return static_cast<int>(std::stacktrace::current().size()); }" LFS_HAVE_STDCXXEXP)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(LFS_HAVE_STDCXXEXP)
        target_link_libraries(lfs_core PRIVATE stdc++exp)
    endif()
endif()
add_subdirectory(src/training)

# ---- Focused GPU gtest binary ------------------------------------------------
if(BUILD_TESTS)
    add_subdirectory(cmake/hip_tests)
endif()

lfs_hip_retag_cuda_sources("${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "[ROCm] Compute-only configuration complete.")
