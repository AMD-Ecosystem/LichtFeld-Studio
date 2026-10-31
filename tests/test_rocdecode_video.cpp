/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// rocDecode video decoding on ROCm: frames match FFmpeg's software decoder, and
// streams the GPU cannot decode are refused with a reason.

#include "io/video/rocdecode_decoder.hpp"
#include "io/video/rocdecode_reference_clips.hpp"

#include <gtest/gtest.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/md5.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <array>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using lfs::io::video::RocDecodeDecoder;
using lfs::io::video::RocDecodeReferenceClip;

namespace {

    std::filesystem::path fixture(const char* name) {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/video" / name;
    }

    struct FrameDeleter {
        void operator()(AVFrame* frame) const { av_frame_free(&frame); }
    };
    using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;

    class Input {
    public:
        explicit Input(const std::filesystem::path& path) {
            if (avformat_open_input(&format_, path.string().c_str(), nullptr, nullptr) < 0)
                return;
            if (avformat_find_stream_info(format_, nullptr) < 0)
                return;
            stream_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            packet_ = av_packet_alloc();
        }
        ~Input() {
            av_packet_free(&packet_);
            avformat_close_input(&format_);
        }
        [[nodiscard]] bool ok() const { return format_ && stream_ >= 0 && packet_; }
        [[nodiscard]] AVStream* stream() const { return format_->streams[stream_]; }
        // The next packet of the video stream, or null at the end.
        AVPacket* next() {
            av_packet_unref(packet_);
            while (av_read_frame(format_, packet_) >= 0) {
                if (packet_->stream_index == stream_)
                    return packet_;
                av_packet_unref(packet_);
            }
            return nullptr;
        }
        void rewind() { ASSERT_GE(av_seek_frame(format_, stream_, 0, AVSEEK_FLAG_BACKWARD), 0); }

    private:
        AVFormatContext* format_ = nullptr;
        AVPacket* packet_ = nullptr;
        int stream_ = -1;
    };

    // Host copies of every frame rocDecode returns, in display order.
    std::vector<FramePtr> decodeOnGpu(RocDecodeDecoder& decoder, Input& input, const std::size_t limit = SIZE_MAX) {
        std::vector<FramePtr> frames;
        FramePtr frame(av_frame_alloc());
        bool drained = false;
        while (frames.size() < limit) {
            const int received = decoder.receiveFrame(frame.get());
            if (received == 0) {
                FramePtr host(av_frame_alloc());
                EXPECT_EQ(RocDecodeDecoder::transferToHost(host.get(), frame.get()), 0);
                EXPECT_EQ(av_frame_copy_props(host.get(), frame.get()), 0);
                frames.push_back(std::move(host));
                continue;
            }
            if (received == AVERROR_EOF)
                break;
            EXPECT_EQ(received, AVERROR(EAGAIN)) << decoder.lastError();
            if (received != AVERROR(EAGAIN) || drained)
                break;
            AVPacket* const packet = input.next();
            drained = packet == nullptr;
            EXPECT_EQ(decoder.sendPacket(packet), 0) << decoder.lastError();
        }
        return frames;
    }

    std::vector<FramePtr> decodeOnCpu(const std::filesystem::path& path) {
        std::vector<FramePtr> frames;
        Input input(path);
        const AVCodec* const codec = avcodec_find_decoder(input.stream()->codecpar->codec_id);
        AVCodecContext* context = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(context, input.stream()->codecpar);
        context->pkt_timebase = input.stream()->time_base;
        if (avcodec_open2(context, codec, nullptr) < 0) {
            avcodec_free_context(&context);
            return frames;
        }
        FramePtr frame(av_frame_alloc());
        for (AVPacket* packet = input.next();; packet = input.next()) {
            avcodec_send_packet(context, packet);
            while (avcodec_receive_frame(context, frame.get()) == 0)
                frames.emplace_back(av_frame_clone(frame.get()));
            if (!packet)
                break;
        }
        avcodec_free_context(&context);
        return frames;
    }

    // A moving gradient encoded with FFmpeg's default H.264 encoder settings (High profile).
    bool writeH264High(const std::filesystem::path& path, const int width, const int height, const int frames) {
        const AVCodec* const codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        AVFormatContext* format = nullptr;
        if (!codec || avformat_alloc_output_context2(&format, nullptr, "mp4", path.string().c_str()) < 0)
            return false;
        AVStream* const stream = avformat_new_stream(format, nullptr);
        AVCodecContext* encoder = avcodec_alloc_context3(codec);
        encoder->width = width;
        encoder->height = height;
        encoder->pix_fmt = AV_PIX_FMT_YUV420P;
        encoder->time_base = AVRational{1, 25};
        encoder->gop_size = 5;
        encoder->max_b_frames = 2;
        encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        av_opt_set(encoder->priv_data, "preset", "fast", 0);
        bool ok = avcodec_open2(encoder, codec, nullptr) >= 0 &&
                  avcodec_parameters_from_context(stream->codecpar, encoder) >= 0;
        stream->time_base = encoder->time_base;
        ok = ok && avio_open(&format->pb, path.string().c_str(), AVIO_FLAG_WRITE) >= 0 &&
             avformat_write_header(format, nullptr) >= 0;
        FramePtr frame(av_frame_alloc());
        frame->format = AV_PIX_FMT_YUV420P;
        frame->width = width;
        frame->height = height;
        ok = ok && av_frame_get_buffer(frame.get(), 0) >= 0;
        AVPacket* packet = av_packet_alloc();
        for (int i = 0; ok && i <= frames; ++i) {
            if (i < frames) {
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x)
                        frame->data[0][y * frame->linesize[0] + x] = static_cast<uint8_t>(x + 3 * y + 7 * i);
                for (int y = 0; y < height / 2; ++y)
                    for (int x = 0; x < width / 2; ++x) {
                        frame->data[1][y * frame->linesize[1] + x] = static_cast<uint8_t>(64 + x + i);
                        frame->data[2][y * frame->linesize[2] + x] = static_cast<uint8_t>(192 - y);
                    }
                frame->pts = i;
            }
            ok = avcodec_send_frame(encoder, i < frames ? frame.get() : nullptr) >= 0;
            while (ok && avcodec_receive_packet(encoder, packet) == 0) {
                av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
                packet->stream_index = stream->index;
                ok = av_interleaved_write_frame(format, packet) >= 0;
            }
        }
        ok = ok && av_write_trailer(format) >= 0;
        av_packet_free(&packet);
        avcodec_free_context(&encoder);
        avio_closep(&format->pb);
        avformat_free_context(format);
        return ok;
    }

    std::unique_ptr<RocDecodeDecoder> openOrSkip(const AVStream* stream, std::string* reason = nullptr) {
        const auto refusal = RocDecodeDecoder::unsupportedReason(stream->codecpar);
        EXPECT_TRUE(refusal.has_value()) << refusal.error();
        if (reason)
            *reason = refusal ? *refusal : refusal.error();
        if (!refusal || !refusal->empty())
            return nullptr;
        auto decoder = RocDecodeDecoder::create(stream->codecpar, stream->time_base);
        EXPECT_TRUE(decoder.has_value()) << decoder.error();
        return decoder ? std::move(*decoder) : nullptr;
    }

    // Differing luma and chroma samples between rocDecode's NV12 frames and the
    // software decoder's planar frames, or -1 when the frame count, format or size differ.
    long long sampleDifferences(const std::vector<FramePtr>& gpu, const std::vector<FramePtr>& cpu) {
        if (gpu.size() != cpu.size() || cpu.empty())
            return -1;
        long long differences = 0;
        for (std::size_t i = 0; i < gpu.size(); ++i) {
            const AVFrame& g = *gpu[i];
            const AVFrame& c = *cpu[i];
            if (g.format != AV_PIX_FMT_NV12 || (c.format != AV_PIX_FMT_YUV420P && c.format != AV_PIX_FMT_YUVJ420P) ||
                g.width != c.width || g.height != c.height)
                return -1;
            for (int y = 0; y < g.height; ++y)
                for (int x = 0; x < g.width; ++x)
                    differences += g.data[0][y * g.linesize[0] + x] != c.data[0][y * c.linesize[0] + x];
            for (int y = 0; y < (g.height + 1) / 2; ++y)
                for (int x = 0; x < (g.width + 1) / 2; ++x) {
                    differences += g.data[1][y * g.linesize[1] + 2 * x] != c.data[1][y * c.linesize[1] + x];
                    differences += g.data[1][y * g.linesize[1] + 2 * x + 1] != c.data[2][y * c.linesize[2] + x];
                }
        }
        return differences;
    }

    // MD5 of a planar 4:2:0 software frame repacked as NV12 or P010, the layout the reference digests describe.
    std::string semiPlanarMd5(const AVFrame& frame) {
        const AVPixFmtDescriptor* const descriptor = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
        const int depth = descriptor->comp[0].depth;
        std::vector<uint8_t> packed;
        const auto put = [&](const uint8_t* plane, const int linesize, const int x, const int y) {
            if (depth == 8) {
                packed.push_back(plane[y * linesize + x]);
                return;
            }
            uint16_t value = 0;
            std::memcpy(&value, plane + y * linesize + 2 * x, sizeof(value));
            value = static_cast<uint16_t>(value << (16 - depth));
            packed.push_back(static_cast<uint8_t>(value & 0xff));
            packed.push_back(static_cast<uint8_t>(value >> 8));
        };
        for (int y = 0; y < frame.height; ++y)
            for (int x = 0; x < frame.width; ++x)
                put(frame.data[0], frame.linesize[0], x, y);
        for (int y = 0; y < (frame.height + 1) / 2; ++y)
            for (int x = 0; x < (frame.width + 1) / 2; ++x) {
                put(frame.data[1], frame.linesize[1], x, y);
                put(frame.data[2], frame.linesize[2], x, y);
            }
        uint8_t digest[16];
        av_md5_sum(digest, packed.data(), packed.size());
        std::string hex;
        for (const uint8_t byte : digest)
            std::format_to(std::back_inserter(hex), "{:02x}", byte);
        return hex;
    }

    std::filesystem::path writeClip(const RocDecodeReferenceClip& clip) {
        const auto path = std::filesystem::temp_directory_path() /
                          std::format("lfs_rocdecode_reference_{}_{}.{}", avcodec_get_name(clip.codec),
                                      clip.bit_depth, std::string_view(clip.demuxer) == "mp4" ? "mp4" : "webm");
        std::ofstream(path, std::ios::binary)
            .write(reinterpret_cast<const char*>(clip.data.data()), static_cast<std::streamsize>(clip.data.size()));
        return path;
    }

    // Each test starts and ends with no remembered reference clip results.
    class RocDecodeSelfCheck : public ::testing::Test {
    protected:
        void SetUp() override { RocDecodeDecoder::resetSelfCheckForTesting(); }
        void TearDown() override { RocDecodeDecoder::resetSelfCheckForTesting(); }
    };

} // namespace

TEST(RocDecodeVideo, StreamsTheVideoEngineCannotDecodeAreNamed) {
    // x264's lossless mode uses the High 4:4:4 Predictive profile even for 4:2:0.
    for (const char* name : {"h264_444.mp4", "h264_444_odd.mp4", "h264_full.mp4"}) {
        SCOPED_TRACE(name);
        Input input(fixture(name));
        ASSERT_TRUE(input.ok());
        const auto refusal = RocDecodeDecoder::unsupportedReason(input.stream()->codecpar);
        ASSERT_TRUE(refusal.has_value()) << refusal.error();
        EXPECT_EQ(*refusal, "rocDecode does not decode the H.264 High 4:4:4 Predictive profile");
    }

    AVCodecParameters* parameters = avcodec_parameters_alloc();
    parameters->codec_type = AVMEDIA_TYPE_VIDEO;
    parameters->codec_id = AV_CODEC_ID_VC1;
    parameters->width = 1920;
    parameters->height = 1080;
    const auto refusal = RocDecodeDecoder::unsupportedReason(parameters);
    ASSERT_TRUE(refusal.has_value()) << refusal.error();
    EXPECT_EQ(*refusal, "rocDecode has no vc1 decoder");
    avcodec_parameters_free(&parameters);
}

TEST(RocDecodeVideo, MinimumSizeAppliesToTheCodedSize) {
    // 60x60 is coded as 64x64 in both codecs, which rocDecode decodes. A GPU
    // and driver whose reference clip decodes wrongly refuse the codec instead.
    for (const char* name : {"h264_60x60.mp4", "hevc_60x60.mp4"}) {
        SCOPED_TRACE(name);
        Input input(fixture(name));
        ASSERT_TRUE(input.ok());
        const auto refusal = RocDecodeDecoder::unsupportedReason(input.stream()->codecpar);
        ASSERT_TRUE(refusal.has_value()) << refusal.error();
        EXPECT_TRUE(refusal->empty() || refusal->starts_with("rocDecode produced wrong frames")) << *refusal;
    }

    Input input(fixture("h264_420.mp4"));
    ASSERT_TRUE(input.ok());
    const auto refusal = RocDecodeDecoder::unsupportedReason(input.stream()->codecpar);
    ASSERT_TRUE(refusal.has_value()) << refusal.error();
    EXPECT_TRUE(refusal->starts_with("64x48 H.264 is outside rocDecode's ")) << *refusal;
}

class RocDecodeMatchesSoftware : public ::testing::TestWithParam<const char*> {};

TEST_P(RocDecodeMatchesSoftware, FramesAndTimestamps) {
    std::filesystem::path path = fixture(GetParam());
    const auto generated = std::filesystem::temp_directory_path() / "lfs_rocdecode_h264_high.mp4";
    if (std::string_view(GetParam()) == "generated H.264 High") {
        ASSERT_TRUE(writeH264High(generated, 320, 180, 12));
        path = generated;
    }
    Input input(path);
    ASSERT_TRUE(input.ok());
    std::string reason;
    const auto decoder = openOrSkip(input.stream(), &reason);
    if (!decoder)
        GTEST_SKIP() << "This GPU does not decode " << GetParam() << " on rocDecode: " << reason;
    const auto gpu = decodeOnGpu(*decoder, input);
    const auto cpu = decodeOnCpu(path);
    ASSERT_FALSE(cpu.empty());
    ASSERT_EQ(gpu.size(), cpu.size());
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        SCOPED_TRACE(i);
        const AVFrame& g = *gpu[i];
        const AVFrame& c = *cpu[i];
        ASSERT_EQ(g.format, AV_PIX_FMT_NV12);
        ASSERT_EQ(g.width, c.width);
        ASSERT_EQ(g.height, c.height);
        EXPECT_EQ(g.pts, c.pts);
        int luma = 0;
        int chroma = 0;
        for (int y = 0; y < g.height; ++y)
            for (int x = 0; x < g.width; ++x)
                luma += g.data[0][y * g.linesize[0] + x] != c.data[0][y * c.linesize[0] + x];
        for (int y = 0; y < (g.height + 1) / 2; ++y)
            for (int x = 0; x < (g.width + 1) / 2; ++x) {
                chroma += g.data[1][y * g.linesize[1] + 2 * x] != c.data[1][y * c.linesize[1] + x];
                chroma += g.data[1][y * g.linesize[1] + 2 * x + 1] != c.data[2][y * c.linesize[2] + x];
            }
        EXPECT_EQ(luma, 0);
        EXPECT_EQ(chroma, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(Fixtures, RocDecodeMatchesSoftware,
                         ::testing::Values("generated H.264 High", "h264_60x60.mp4", "hevc_full.mp4",
                                           "hevc_60x60.mp4", "vp9_full.webm", "vp9_timestamps.webm"));

TEST(RocDecodeVideo, FlushRestartsFromTheSeekTarget) {
    Input input(fixture("vp9_timestamps.webm"));
    ASSERT_TRUE(input.ok());
    std::string reason;
    const auto decoder = openOrSkip(input.stream(), &reason);
    if (!decoder)
        GTEST_SKIP() << "This GPU does not decode VP9 on rocDecode: " << reason;
    const auto first = decodeOnGpu(*decoder, input, 12);
    ASSERT_EQ(first.size(), 12u);
    input.rewind();
    decoder->flush();
    const auto again = decodeOnGpu(*decoder, input);
    ASSERT_EQ(again.size(), 185u);
    for (std::size_t i = 0; i < first.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(again[i]->pts, first[i]->pts);
        for (int y = 0; y < first[i]->height; ++y)
            ASSERT_EQ(0, std::memcmp(again[i]->data[0] + y * again[i]->linesize[0],
                                     first[i]->data[0] + y * first[i]->linesize[0], first[i]->width));
    }
}

TEST(RocDecodeVideo, ColourRangeFollowsTheStream) {
    for (const auto& [name, range] : {std::pair{"vp9_full.webm", AVCOL_RANGE_JPEG},
                                      std::pair{"vp9_limited.webm", AVCOL_RANGE_MPEG},
                                      std::pair{"av1_full.mkv", AVCOL_RANGE_JPEG},
                                      std::pair{"av1_limited.mkv", AVCOL_RANGE_MPEG},
                                      std::pair{"hevc_full.mp4", AVCOL_RANGE_JPEG}}) {
        SCOPED_TRACE(name);
        Input input(fixture(name));
        ASSERT_TRUE(input.ok());
        const auto decoder = openOrSkip(input.stream());
        if (!decoder)
            continue;
        const auto frames = decodeOnGpu(*decoder, input);
        ASSERT_EQ(frames.size(), 3u);
        for (const auto& frame : frames) {
            EXPECT_EQ(frame->color_range, range);
            EXPECT_EQ(frame->colorspace, AVCOL_SPC_BT470BG);
        }
    }
}

// The embedded digests are what FFmpeg's software decoders produce for the embedded clips.
TEST_F(RocDecodeSelfCheck, ReferenceDigestsMatchTheSoftwareDecode) {
    ASSERT_EQ(lfs::io::video::rocDecodeReferenceClips().size(), 7u);
    for (const RocDecodeReferenceClip& clip : lfs::io::video::rocDecodeReferenceClips()) {
        SCOPED_TRACE(std::format("{} {}-bit", avcodec_get_name(clip.codec), clip.bit_depth));
        const auto path = writeClip(clip);
        const auto frames = decodeOnCpu(path);
        std::filesystem::remove(path);
        ASSERT_EQ(frames.size(), clip.frame_md5.size());
        for (std::size_t i = 0; i < frames.size(); ++i) {
            EXPECT_EQ(frames[i]->width, 64);
            EXPECT_EQ(semiPlanarMd5(*frames[i]), clip.frame_md5[i]) << "frame " << i;
        }
    }
}

// A digest mismatch refuses the codec with a named reason, and the result is
// remembered: the reference clip is decoded once per process and codec.
TEST_F(RocDecodeSelfCheck, WrongFramesDecodeOnTheCpu) {
    Input input(fixture("h264_60x60.mp4"));
    ASSERT_TRUE(input.ok());
    const AVCodecParameters* const parameters = input.stream()->codecpar;
    RocDecodeDecoder::overrideSelfCheckDigestsForTesting(AV_CODEC_ID_H264, 8, {std::string(32, '0'), std::string(32, '0')});
    const auto refusal = RocDecodeDecoder::unsupportedReason(parameters);
    ASSERT_TRUE(refusal.has_value()) << refusal.error();
    if (RocDecodeDecoder::selfCheckRunsForTesting() == 0)
        GTEST_SKIP() << "This GPU does not decode H.264 on rocDecode: " << *refusal;
    EXPECT_EQ(*refusal, "rocDecode produced wrong frames for H.264 8-bit on this GPU and driver");
    EXPECT_EQ(RocDecodeDecoder::selfCheckRunsForTesting(), 1);

    const auto again = RocDecodeDecoder::unsupportedReason(parameters);
    ASSERT_TRUE(again.has_value()) << again.error();
    EXPECT_EQ(*again, *refusal);
    const auto decoder = RocDecodeDecoder::create(parameters, input.stream()->time_base);
    ASSERT_FALSE(decoder.has_value());
    EXPECT_EQ(decoder.error(), *refusal);
    EXPECT_EQ(RocDecodeDecoder::selfCheckRunsForTesting(), 1);

    // Another codec keeps its own result.
    Input hevc(fixture("hevc_60x60.mp4"));
    ASSERT_TRUE(hevc.ok());
    const auto hevc_refusal = RocDecodeDecoder::unsupportedReason(hevc.stream()->codecpar);
    ASSERT_TRUE(hevc_refusal.has_value()) << hevc_refusal.error();
    EXPECT_NE(*hevc_refusal, *refusal);
}

TEST_F(RocDecodeSelfCheck, ConcurrentFirstUseDecodesTheClipOnce) {
    Input input(fixture("h264_60x60.mp4"));
    ASSERT_TRUE(input.ok());
    const AVCodecParameters* const parameters = input.stream()->codecpar;
    std::array<std::expected<std::string, std::string>, 8> results;
    {
        std::vector<std::jthread> threads;
        for (auto& result : results)
            threads.emplace_back([&result, parameters] { result = RocDecodeDecoder::unsupportedReason(parameters); });
    }
    for (const auto& result : results) {
        ASSERT_TRUE(result.has_value()) << result.error();
        EXPECT_EQ(*result, *results[0]);
    }
    EXPECT_LE(RocDecodeDecoder::selfCheckRunsForTesting(), 1);
}

// The check's verdict on this GPU and driver predicts a full decode: a codec it
// accepts decodes the fixtures bit-exactly, and a codec it refuses does not.
TEST_F(RocDecodeSelfCheck, VerdictAgreesWithAFullDecode) {
    for (const char* name : {"h264_60x60.mp4", "hevc_60x60.mp4", "hevc_full.mp4", "vp9_full.webm"}) {
        SCOPED_TRACE(name);
        Input input(fixture(name));
        ASSERT_TRUE(input.ok());
        const auto refusal = RocDecodeDecoder::unsupportedReason(input.stream()->codecpar);
        ASSERT_TRUE(refusal.has_value()) << refusal.error();
        const bool wrong_frames = refusal->starts_with("rocDecode produced wrong frames");
        if (!refusal->empty() && !wrong_frames)
            continue;
        auto decoder = RocDecodeDecoder::createWithoutSelfCheckForTesting(input.stream()->codecpar,
                                                                          input.stream()->time_base);
        ASSERT_TRUE(decoder.has_value()) << decoder.error();
        const auto gpu = decodeOnGpu(**decoder, input);
        const long long differences = sampleDifferences(gpu, decodeOnCpu(fixture(name)));
        ASSERT_GE(differences, 0) << "frame count or layout differs from the software decode";
        if (wrong_frames)
            EXPECT_GT(differences, 0) << *refusal;
        else
            EXPECT_EQ(differences, 0);
    }
}

// Each reference clip either decodes correctly on this GPU, or its codec and
// bit depth are refused by name: caps, or wrong frames from the check.
TEST_F(RocDecodeSelfCheck, EveryReferenceClipIsAcceptedOrRefusedByName) {
    for (const RocDecodeReferenceClip& clip : lfs::io::video::rocDecodeReferenceClips()) {
        SCOPED_TRACE(std::format("{} {}-bit", avcodec_get_name(clip.codec), clip.bit_depth));
        const auto path = writeClip(clip);
        Input input(path);
        ASSERT_TRUE(input.ok());
        const auto refusal = RocDecodeDecoder::unsupportedReason(input.stream()->codecpar);
        std::filesystem::remove(path);
        ASSERT_TRUE(refusal.has_value()) << refusal.error();
        EXPECT_TRUE(refusal->empty() || refusal->starts_with("rocDecode produced wrong frames") ||
                    refusal->starts_with("rocDecode does not decode"))
            << *refusal;
    }
}
