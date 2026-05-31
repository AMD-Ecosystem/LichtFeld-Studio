# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# MOAT focused GPU gtest binary for the ROCm compute tranche. Links only the
# compute libraries + GTest + the ROCm libtorch parity oracle and the subset of
# tests/test_*.cpp that exercise the tensor library, training kernels, losses,
# MCMC and the gsplat/fastgs/edge rasterizers. The monolithic upstream
# lichtfeld_tests target is NOT buildable on ROCm (it links the whole GUI/USD/
# Python graph); this is the decoupled compute-test exe (build-file addition).

find_package(GTest CONFIG REQUIRED)
# ROCm libtorch is the test-time parity oracle (tensor-vs-torch comparisons).
# Torch_DIR is supplied on the command line (the project's external/libtorch is a
# CUDA build; point at the installed ROCm python torch instead).
find_package(Torch REQUIRED)
message(STATUS "[MOAT/HIP] Tests: libtorch ${Torch_VERSION} (ROCm parity oracle)")

set(LFS_TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")

# Compute-only test set (pure tensor/kernel/rasterizer/loss/MCMC). Tests that
# pull lfs_io fixtures (PLY/exporter) or the GUI/MCP/Python/USD graph are omitted.
set(LFS_HIP_TEST_FILES
    # tensor library (wave64-sensitive core)
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_reduction.cpp
    test_tensor_reduction_alignment.cpp
    test_tensor_math.cpp
    test_tensor_matrix.cpp
    test_tensor_broadcast.cpp
    test_tensor_masking.cpp
    test_tensor_random.cpp
    test_tensor_random_advanced.cpp
    test_tensor_vs_torch.cpp
    test_torch_comparisons.cpp
    test_tensor_clamping.cpp
    test_sort_minimal.cpp
    test_tensor_bool.cpp
    test_bool_reduction.cpp
    test_bool_any_all.cpp
    test_nan_inf_gpu_check.cpp
    test_curand_buffer_overflow.cpp
    # losses / ssim
    test_fused_l1_ssim.cpp
    test_mask_loss.cpp
    test_loss_gradients.cpp
    test_activation_gradients.cpp
    test_gradient_accumulation.cpp
    test_image_kernels.cpp
    # mcmc / strategy
    test_mcmc.cpp
    test_mcmc_kernels.cpp
    test_mcmc_tensor_ops.cpp
    test_mcmc_memory_ops.cpp
    test_mcmc_dead_mask.cpp
    test_mcmc_nan_fix.cpp
    test_mcmc_logit_verification.cpp
    test_relocate_gs_edge_cases.cpp
    test_densification_tensor_ops.cpp
    # rasterizers
    test_gsplat_rasterizer.cpp
    test_fastgs_analytical_gradients.cpp
    # adam / scheduler / bilateral / ppisp / sparsity
    test_lfs_scheduler.cpp
    test_bilateral_grid.cpp
    test_ppisp_cuda_vs_torch.cpp
    test_ppisp_regularization.cpp
    test_sparsity_optimizer.cpp
    # mcmc compute regressions (no io dependency -- were silently dropped)
    test_mcmc_relocate_optimizer_state_bug.cpp
    # CPU tensor regressions (no io dependency -- were silently dropped)
    test_cpu_dtype_conversions.cpp
    test_cpu_large_tensor_bugs.cpp
)

set(LFS_HIP_TEST_SOURCES "${LFS_TESTS_DIR}/test_main.cpp")
foreach(_t IN LISTS LFS_HIP_TEST_FILES)
    if(EXISTS "${LFS_TESTS_DIR}/${_t}")
        list(APPEND LFS_HIP_TEST_SOURCES "${LFS_TESTS_DIR}/${_t}")
    else()
        message(WARNING "[MOAT/HIP] test file not found, skipping: ${_t}")
    endif()
endforeach()

add_executable(lfs_compute_tests ${LFS_HIP_TEST_SOURCES})

set_target_properties(lfs_compute_tests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
)

target_include_directories(lfs_compute_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src/training
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core/include
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_SOURCE_DIR}/src/training/rasterization
)

target_compile_definitions(lfs_compute_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data"
    PROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}"
    # ROCm libtorch ships c10/hip, not c10/cuda/impl/cuda_cmake_macros.h; torch's
    # own header documents this define as the escape hatch for the unported AMD
    # configure-file. USE_ROCM routes the c10/cuda compat includes to c10/hip.
    C10_CUDA_NO_CMAKE_CONFIGURE_FILE
    USE_ROCM=1
)

target_link_libraries(lfs_compute_tests PRIVATE
    lfs_training
    lfs_core
    lfs_geometry
    fastlfs_backend
    gsplat_backend_lfs
    edge_compute_backend
    GTest::gtest
    ${TORCH_LIBRARIES}
    OpenImageIO::OpenImageIO
    CUDA::cudart
    CUDA::curand
    spdlog::spdlog
)

include(GoogleTest)
gtest_discover_tests(lfs_compute_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 120
    DISCOVERY_MODE PRE_TEST
)

message(STATUS "[MOAT/HIP] lfs_compute_tests configured (compute gtest set + ROCm torch oracle)")
