/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Built on Windows when the ROCm SDK has no rocDecode (Windows support starts
// with rocDecode 1.12 in ROCm 10.2): every stream decodes on the CPU.

#include "rocdecode_decoder.hpp"

#include <cerrno>

namespace lfs::io::video {

    namespace {
        constexpr const char* kUnavailable = "rocDecode not available in this ROCm SDK on Windows";
    } // namespace

    struct RocDecodeDecoder::State {};

    std::expected<std::string, std::string>
    RocDecodeDecoder::unsupportedReason(const AVCodecParameters* /*parameters*/) {
        return std::string(kUnavailable);
    }

    std::expected<std::unique_ptr<RocDecodeDecoder>, std::string>
    RocDecodeDecoder::create(const AVCodecParameters* /*parameters*/, AVRational /*time_base*/) {
        return std::unexpected(std::string(kUnavailable));
    }

    RocDecodeDecoder::~RocDecodeDecoder() = default;

    int RocDecodeDecoder::sendPacket(const AVPacket* /*packet*/) { return AVERROR(ENOSYS); }

    int RocDecodeDecoder::receiveFrame(AVFrame* /*frame*/) { return AVERROR(ENOSYS); }

    void RocDecodeDecoder::flush() {}

    int RocDecodeDecoder::transferToHost(AVFrame* /*dst*/, const AVFrame* /*src*/) { return AVERROR(ENOSYS); }

} // namespace lfs::io::video
