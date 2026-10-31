# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

# The ROCm build decodes AV1 on rocDecode where the GPU has AV1 decode
# hardware, and with FFmpeg's software decoders elsewhere. FFmpeg's own AV1
# decoder only drives hardware accelerators, so software AV1 decoding needs
# libdav1d. The CUDA build decodes AV1 on NVDEC and keeps the manifest as it is:
# for the ROCm build, vcpkg installs a copy of the manifest in the build tree
# with FFmpeg's dav1d feature added. Call before project().
function(lfs_hip_vcpkg_manifest)
    set(_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    set(_manifest_dir "${CMAKE_BINARY_DIR}/vcpkg-manifest-rocm")
    if(DEFINED CACHE{VCPKG_MANIFEST_DIR} AND NOT VCPKG_MANIFEST_DIR STREQUAL "" AND
       NOT VCPKG_MANIFEST_DIR STREQUAL _source_dir AND NOT VCPKG_MANIFEST_DIR STREQUAL _manifest_dir)
        message(STATUS "vcpkg manifest: ${VCPKG_MANIFEST_DIR} (set by the user; FFmpeg's dav1d feature is not added)")
        return()
    endif()

    file(READ "${_source_dir}/vcpkg.json" _manifest)
    string(JSON _count LENGTH "${_manifest}" dependencies)
    math(EXPR _last "${_count} - 1")
    set(_ffmpeg_found FALSE)
    foreach(_index RANGE ${_last})
        string(JSON _type TYPE "${_manifest}" dependencies ${_index})
        if(NOT _type STREQUAL "OBJECT")
            continue()
        endif()
        string(JSON _name GET "${_manifest}" dependencies ${_index} name)
        if(_name STREQUAL "ffmpeg")
            string(JSON _features LENGTH "${_manifest}" dependencies ${_index} features)
            string(JSON _manifest SET "${_manifest}" dependencies ${_index} features ${_features} "\"dav1d\"")
            set(_ffmpeg_found TRUE)
        endif()
    endforeach()
    if(NOT _ffmpeg_found)
        message(FATAL_ERROR "vcpkg.json has no ffmpeg dependency with a feature list to add dav1d to")
    endif()

    # The configuration's overlay paths are relative to the source tree.
    file(READ "${_source_dir}/vcpkg-configuration.json" _configuration)
    string(JSON _overlays ERROR_VARIABLE _no_overlays LENGTH "${_configuration}" overlay-ports)
    if(NOT _no_overlays AND _overlays GREATER 0)
        math(EXPR _last "${_overlays} - 1")
        foreach(_index RANGE ${_last})
            string(JSON _overlay GET "${_configuration}" overlay-ports ${_index})
            cmake_path(ABSOLUTE_PATH _overlay BASE_DIRECTORY "${_source_dir}" NORMALIZE)
            string(JSON _configuration SET "${_configuration}" overlay-ports ${_index} "\"${_overlay}\"")
        endforeach()
    endif()
    string(JSON _manifest SET "${_manifest}" vcpkg-configuration "${_configuration}")

    # Rewrite only on change, so vcpkg does not rerun on every configure.
    set(_previous "")
    if(EXISTS "${_manifest_dir}/vcpkg.json")
        file(READ "${_manifest_dir}/vcpkg.json" _previous)
    endif()
    if(NOT _previous STREQUAL "${_manifest}\n")
        file(WRITE "${_manifest_dir}/vcpkg.json" "${_manifest}\n")
        # pkg_check_modules keeps its first answer in the cache. A changed
        # manifest can change the static link line (libavcodec now needs
        # -ldav1d), so query pkg-config again.
        get_cmake_property(_cache_variables CACHE_VARIABLES)
        foreach(_variable IN LISTS _cache_variables)
            if(_variable MATCHES "^__pkg_config_checked_")
                unset(${_variable} CACHE)
            endif()
        endforeach()
    endif()
    set_property(DIRECTORY "${_source_dir}" APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_source_dir}/vcpkg.json" "${_source_dir}/vcpkg-configuration.json")
    set(VCPKG_MANIFEST_DIR "${_manifest_dir}" PARENT_SCOPE)
    message(STATUS "vcpkg manifest: ${_manifest_dir}/vcpkg.json (vcpkg.json with FFmpeg's dav1d feature)")
endfunction()
