/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <cstdint>
#include <span>

extern "C" {
#include <libavcodec/codec_id.h>
}

namespace lfs::io::video {

    // A short valid clip and the MD5 of each frame FFmpeg's software decoder
    // produces for it, packed as NV12 (8-bit) or P010 (10-bit) with no row
    // padding, which is what FFmpeg's framemd5 muxer hashes.
    struct RocDecodeReferenceClip {
        AVCodecID codec;
        int bit_depth;
        const char* demuxer;
        std::span<const uint8_t> data;
        std::array<const char*, 2> frame_md5;
    };

    [[nodiscard]] std::span<const RocDecodeReferenceClip> rocDecodeReferenceClips();

} // namespace lfs::io::video
