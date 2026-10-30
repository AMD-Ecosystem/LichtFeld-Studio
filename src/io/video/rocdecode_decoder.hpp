/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <expected>
#include <memory>
#include <string>

namespace lfs::io::video {

    // Hardware video decode on ROCm. FFmpeg demuxes; rocDecode parses and decodes
    // on the GPU's video engine. Frames are returned in device memory: format is
    // the semi-planar layout (NV12, P010 or P012), data[0] and data[1] are device
    // pointers to the luma and interleaved chroma planes, and buf[0] owns them.
    // They are complete when returned, so no stream ordering is needed.
    class RocDecodeDecoder {
    public:
        // Why rocDecode cannot decode this stream on the current GPU, or an empty
        // string when it can; those streams decode on the CPU. A failed capability
        // query is an error.
        [[nodiscard]] static std::expected<std::string, std::string>
        unsupportedReason(const AVCodecParameters* parameters);

        // Opens a decoder for a stream unsupportedReason accepts.
        [[nodiscard]] static std::expected<std::unique_ptr<RocDecodeDecoder>, std::string>
        create(const AVCodecParameters* parameters, AVRational time_base);

        ~RocDecodeDecoder();
        RocDecodeDecoder(const RocDecodeDecoder&) = delete;
        RocDecodeDecoder& operator=(const RocDecodeDecoder&) = delete;

        // Same contract as avcodec_send_packet, avcodec_receive_frame and
        // avcodec_flush_buffers; a null packet starts draining. Errors are
        // negative AVERROR codes with the reason in lastError().
        [[nodiscard]] int sendPacket(const AVPacket* packet);
        [[nodiscard]] int receiveFrame(AVFrame* frame);
        void flush();

        [[nodiscard]] const std::string& lastError() const { return error_; }

        // Copies a frame from receiveFrame into host memory, as av_hwframe_transfer_data does.
        [[nodiscard]] static int transferToHost(AVFrame* dst, const AVFrame* src);

        struct State;

    private:
        RocDecodeDecoder() = default;
        [[nodiscard]] int parse(const uint8_t* data, int size, int64_t pts, bool end_of_stream, bool discard);

        std::unique_ptr<State> state_;
        std::string error_;
    };

} // namespace lfs::io::video
