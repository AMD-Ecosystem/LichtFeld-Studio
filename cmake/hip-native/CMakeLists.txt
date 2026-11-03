# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(LichtFeldHipNative LANGUAGES NONE)
# SDK-only Tensor runtime check. No application, CUDA toolkit, or GPU is required.
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Defined before adding the runtime so an explicitly requested GPU test build
# can also compile its separate fault-injection module.
option(LFS_HIP_NATIVE_GPU_TESTS "Register SDK-only HIP GPU boundary tests" OFF)
add_subdirectory("${_root}/src/core/tensor/backend/hip" runtime)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME HipNative.CiFilter
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/python/test_hip_native_ci_filter.py")
set_tests_properties(HipNative.CiFilter PROPERTIES LABELS "cpu;hip" TIMEOUT 60)
add_test(NAME HipNative.Configuration
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/python/test_hip_runtime_configuration.py")
set_tests_properties(HipNative.Configuration PROPERTIES
    ENVIRONMENT "CMAKE_COMMAND=${CMAKE_COMMAND}" LABELS "cpu;hip" TIMEOUT 60)
add_test(NAME HipNative.PointwiseCatalog
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/python/test_hip_pointwise_catalog.py")
set_tests_properties(HipNative.PointwiseCatalog PROPERTIES LABELS "cpu;hip" TIMEOUT 60)
add_test(NAME HipNative.ModuleBoundary
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_native_runtime.py"
        "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin" --abi-only)
set_tests_properties(HipNative.ModuleBoundary PROPERTIES LABELS "cpu;hip" TIMEOUT 30)
add_test(NAME HipNative.TensorModuleBoundary
    COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_tensor_operations.py"
        "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin" --abi-only)
set_tests_properties(HipNative.TensorModuleBoundary PROPERTIES LABELS "cpu;hip" TIMEOUT 30)

# GPU execution is explicit; merely configuring or compiling needs no AMD GPU.
if(LFS_HIP_NATIVE_GPU_TESTS)
    set(LFS_HIP_NATIVE_TEST_DEVICE "0" CACHE STRING "HIP ordinal for explicit native GPU tests")
    add_test(NAME HipNative.RuntimeBoundary
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_native_runtime.py"
            "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}")
    set_tests_properties(HipNative.RuntimeBoundary PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
    add_test(NAME HipNative.FaultRecovery
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_native_runtime.py"
            "${LFS_HIP_FAULT_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}" --faults)
    set_tests_properties(HipNative.FaultRecovery PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
    add_test(NAME HipNative.TensorOperations
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_tensor_operations.py"
            "${LFS_HIP_RUNTIME_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}")
    set_tests_properties(HipNative.TensorOperations PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
    add_test(NAME HipNative.TensorWorkspaceRecovery
        COMMAND "${Python3_EXECUTABLE}" "${_root}/tests/hip/test_tensor_operations.py"
            "${LFS_HIP_FAULT_LIBRARY}" --runtime-dir "${LFS_HIP_ROOT}/bin"
            --device "${LFS_HIP_NATIVE_TEST_DEVICE}" --faults)
    set_tests_properties(HipNative.TensorWorkspaceRecovery PROPERTIES LABELS "hip;gpu" TIMEOUT 120)

endif()

# Shared CUDA/HIP warp-group kernels: always compiled, run only with the GPU tests.
if(WIN32 OR CMAKE_HOST_WIN32)
    set(_device_test_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++.exe")
    set(_device_test_link_flags)
else()
    set(_device_test_compiler "${LFS_HIP_ROOT}/lib/llvm/bin/clang++")
    set(_device_test_link_flags "-Wl,-rpath,${LFS_HIP_ROOT}/lib" "-Wl,-rpath,${LFS_HIP_ROOT}/lib64")
endif()
foreach(_device_test IN ITEMS SharedWarpKernels:shared_warp_kernels
                              SharedTrainingWarpReduce:shared_training_warp_reduce)
    string(REPLACE ":" ";" _device_test "${_device_test}")
    list(GET _device_test 0 _test_name)
    list(GET _device_test 1 _test_source)
    set(_test_executable "${CMAKE_BINARY_DIR}/hip_${_test_source}_tests${CMAKE_EXECUTABLE_SUFFIX}")
    add_custom_command(OUTPUT "${_test_executable}"
        COMMAND "${_device_test_compiler}" -x hip -std=c++20 -O3 ${LFS_HIP_OFFLOAD_FLAGS}
            "--hip-path=${LFS_HIP_ROOT}" -DLFS_TENSOR_PRIVATE_ACCESS
            "-I${_root}/src" "-I${_root}/src/core/include"
            -MD -MF "${_test_executable}.d"
            "${_root}/tests/hip/test_${_test_source}.hip" --hip-link ${_device_test_link_flags}
            -o "${_test_executable}"
        DEPENDS "${_root}/tests/hip/test_${_test_source}.hip" "${_device_test_compiler}"
        DEPFILE "${_test_executable}.d"
        VERBATIM)
    add_custom_target(${_test_source}_hip_tests ALL DEPENDS "${_test_executable}")
    if(LFS_HIP_NATIVE_GPU_TESTS)
        add_test(NAME HipNative.${_test_name}
            COMMAND "${_test_executable}" --device "${LFS_HIP_NATIVE_TEST_DEVICE}")
        set_tests_properties(HipNative.${_test_name} PROPERTIES LABELS "hip;gpu" TIMEOUT 120)
    endif()
endforeach()
