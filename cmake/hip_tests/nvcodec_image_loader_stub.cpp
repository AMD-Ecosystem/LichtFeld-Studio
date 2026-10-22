/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * nvImageCodec is NVIDIA-only and not part of the ROCm compute build. The
 * image-kernel tests construct the loader inside a try block and skip when it
 * throws, which is the path this stand-in takes.
 */

#include "io/nvcodec_image_loader.hpp"

#include <stdexcept>

namespace lfs::io {

    struct NvCodecImageLoader::Impl {};

    NvCodecImageLoader::NvCodecImageLoader(const Options&) {
        throw std::runtime_error("nvImageCodec is not available in the ROCm compute build");
    }

    NvCodecImageLoader::~NvCodecImageLoader() = default;

    std::vector<lfs::core::Tensor> NvCodecImageLoader::decode_jpeg_batch_from_spans(
        const std::vector<std::pair<const uint8_t*, size_t>>&, void*, bool, bool,
        std::vector<lfs::core::Tensor*>*, std::vector<lfs::core::Tensor*>*) {
        throw std::runtime_error("nvImageCodec is not available in the ROCm compute build");
    }

} // namespace lfs::io
