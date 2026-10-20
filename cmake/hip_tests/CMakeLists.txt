# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Focused GPU gtest binary for the ROCm compute tranche. Links only the
# compute libraries + GTest + the ROCm libtorch parity oracle and the subset of
# tests/test_*.cpp that exercise the tensor library, training kernels, losses,
# MCMC and the gsplat/fastgs rasterizers. The monolithic upstream
# lichtfeld_tests target is NOT buildable on ROCm (it links the whole GUI/USD/
# Python graph); this is the decoupled compute-test exe (build-file addition).

find_package(GTest CONFIG REQUIRED)
# ROCm libtorch is the test-time parity oracle (tensor-vs-torch comparisons).
# Torch_DIR is supplied on the command line (the project's external/libtorch is a
# CUDA build; point at the installed ROCm python torch instead).
find_package(Torch REQUIRED)
message(STATUS "[ROCm] Tests: libtorch ${Torch_VERSION} (ROCm parity oracle)")

set(LFS_TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")

# Compute-only test set (pure tensor/kernel/rasterizer/loss/MCMC). Tests that
# pull lfs_io fixtures (PLY/exporter), drive the Trainer or a strategy that
# includes the dataset loader (test_mask_loss, test_mcmc), or need the
# GUI/MCP/Python/USD graph are omitted.
set(LFS_HIP_TEST_FILES
    # tensor library (wave64-sensitive core)
    test_tensor_basic.cpp
    test_tensor_ops.cpp
    test_tensor_reduction.cpp
    test_tensor_reduction_alignment.cpp
    test_tensor_matrix.cpp
    test_tensor_broadcast.cpp
    test_tensor_masking.cpp
    test_tensor_random.cpp
    test_tensor_random_advanced.cpp
    test_tensor_clamping.cpp
    test_bool_any_all.cpp
    test_nan_inf_gpu_check.cpp
    # losses / ssim
    test_fused_l1_ssim.cpp
    test_gradient_accumulation.cpp
    test_image_kernels.cpp
    # mcmc / strategy
    test_mcmc_tensor_ops.cpp
    test_mcmc_nan_fix.cpp
    test_densification_tensor_ops.cpp
    # rasterizers
    test_gsplat_rasterizer.cpp
    test_fastgs_analytical_gradients.cpp
    # adam / scheduler / bilateral / ppisp
    test_lfs_scheduler.cpp
    test_ppisp_cuda_vs_torch.cpp
    test_ppisp_regularization.cpp
    # mcmc compute regressions (no io dependency -- were silently dropped)
    # CPU tensor regressions (no io dependency -- were silently dropped)
    test_cpu_dtype_conversions.cpp
    # nn ops kernels (max_pool2d, adaptive_avg_pool2d, conv1x1, linear, relu)
    test_tensor_nn_ops.cpp
    # tensor library coverage (compute-only: include only core/tensor + torch oracle)
    test_add_new_gs_tensor_ops.cpp
    test_background_image.cpp
    test_expanded_tensor_ops.cpp
    test_expression_templates.cpp
    test_extended_unary_ops_vs_torch.cpp
    test_interleaved_slice_copy.cpp
    test_logger.cpp
    test_slice_mm_bug.cpp
    test_stride_bugs.cpp
    test_strided_fill.cpp
    test_tensor_accessor.cpp
    test_tensor_advanced.cpp
    test_tensor_bitwise.cpp
    test_tensor_cat_reduction_bug.cpp
    test_tensor_conversions_shape.cpp
    test_tensor_cumsum_convenience.cpp
    test_tensor_index_select_uint8.cpp
    test_tensor_indexing_advanced.cpp
    test_tensor_inplace_capacity.cpp
    test_tensor_lazy_ir.cpp
    test_tensor_lazy_stateful_ops.cpp
    test_tensor_masked_select_rows.cpp
    test_tensor_memory.cpp
    test_tensor_memory_planner.cpp
    test_tensor_reserve_inplace_cat.cpp
    test_tensor_serialization.cpp
    test_tensor_slicing_safety.cpp
    test_tensor_storage_version.cpp
    test_tensor_stream.cpp
    test_tensor_stress.cpp
    test_tensor_uint8_inversion.cpp
    test_tensor_uint8_masking.cpp
    test_tensor_views.cpp
    test_tensor_zero_dimension.cpp
    test_allocator_policy.cpp
    # diagnostics (vram profiler accounting -- lfs_diagnostics linked via lfs_core)
    test_vram_profiler_metrics.cpp
)

set(LFS_HIP_TEST_SOURCES "${LFS_TESTS_DIR}/test_main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nvcodec_image_loader_stub.cpp")
foreach(_t IN LISTS LFS_HIP_TEST_FILES)
    if(EXISTS "${LFS_TESTS_DIR}/${_t}")
        list(APPEND LFS_HIP_TEST_SOURCES "${LFS_TESTS_DIR}/${_t}")
    else()
        message(WARNING "[ROCm] test file not found, skipping: ${_t}")
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
    # Set by the skipped configure-file; keeps dllimport on c10_hip data exports.
    $<$<PLATFORM_ID:Windows>:C10_CUDA_BUILD_SHARED_LIBS>
    USE_ROCM=1
)

# Some upstream tests narrow size_t in braced lists, which GCC only warns about
# and clang rejects; keep GCC's behaviour for the test sources.
target_compile_options(lfs_compute_tests PRIVATE -Wno-c++11-narrowing)

target_link_libraries(lfs_compute_tests PRIVATE
    lfs_training
    lfs_depth_anchor
    lfs_io
    lfs_core
    lfs_geometry
    fastlfs_backend
    gsplat_backend_lfs
    GTest::gtest
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::curand
    spdlog::spdlog
)

if(WIN32)
    # find_package(Torch) restores clang++ HIP link rules in this directory.
    set_target_properties(lfs_compute_tests PROPERTIES LINKER_LANGUAGE CXX)
    # Keep the torch_hip.dll import, or torch::cuda::is_available() is false.
    target_link_options(lfs_compute_tests PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
    # The driver's amdhip64 in System32 wins over PATH and cannot run SDK code objects.
    file(GLOB _lfs_amdhip_dll "${ROCM_PATH}/bin/amdhip64_*.dll")
    if(_lfs_amdhip_dll)
        add_custom_command(TARGET lfs_compute_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_lfs_amdhip_dll} $<TARGET_FILE_DIR:lfs_compute_tests>)
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(lfs_compute_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES TIMEOUT 120
    DISCOVERY_MODE PRE_TEST
)

message(STATUS "[ROCm] lfs_compute_tests configured (compute gtest set + ROCm torch oracle)")
