/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rocdecode_decoder.hpp"
#include "rocdecode_reference_clips.hpp"

#include "core/logger.hpp"

extern "C" {
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/imgutils.h>
#include <libavutil/md5.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
}

#include <hip/hip_runtime_api.h>
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <format>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace lfs::io::video {

    namespace {

        constexpr int PITCH_ALIGNMENT = 256;

        struct CodecMapping {
            rocDecVideoCodec codec;
            const char* name;
            // Block size the coded size is a multiple of (H.264 macroblock, smallest HEVC coding block). The
            // caps minimum applies to the coded size.
            int coded_alignment = 1;
        };

        [[nodiscard]] std::optional<CodecMapping> rocDecodeCodec(const AVCodecID id) {
            switch (id) {
            case AV_CODEC_ID_H264: return CodecMapping{rocDecVideoCodec_AVC, "H.264", 16};
            case AV_CODEC_ID_HEVC: return CodecMapping{rocDecVideoCodec_HEVC, "HEVC", 8};
            case AV_CODEC_ID_AV1: return CodecMapping{rocDecVideoCodec_AV1, "AV1"};
            case AV_CODEC_ID_VP9: return CodecMapping{rocDecVideoCodec_VP9, "VP9"};
            case AV_CODEC_ID_VP8: return CodecMapping{rocDecVideoCodec_VP8, "VP8"};
            case AV_CODEC_ID_MPEG1VIDEO: return CodecMapping{rocDecVideoCodec_MPEG1, "MPEG-1"};
            case AV_CODEC_ID_MPEG2VIDEO: return CodecMapping{rocDecVideoCodec_MPEG2, "MPEG-2"};
            case AV_CODEC_ID_MPEG4: return CodecMapping{rocDecVideoCodec_MPEG4, "MPEG-4 Part 2"};
            default: return std::nullopt;
            }
        }

        [[nodiscard]] const char* chromaName(const rocDecVideoChromaFormat chroma) {
            switch (chroma) {
            case rocDecVideoChromaFormat_Monochrome: return "monochrome";
            case rocDecVideoChromaFormat_420: return "4:2:0";
            case rocDecVideoChromaFormat_422: return "4:2:2";
            case rocDecVideoChromaFormat_444: return "4:4:4";
            }
            return "unknown chroma";
        }

        struct StreamLayout {
            rocDecVideoChromaFormat chroma = rocDecVideoChromaFormat_420;
            int bit_depth = 8;
        };

        [[nodiscard]] std::expected<StreamLayout, std::string> streamLayout(const AVCodecParameters* const parameters) {
            StreamLayout layout;
            const AVPixFmtDescriptor* const descriptor =
                av_pix_fmt_desc_get(static_cast<AVPixelFormat>(parameters->format));
            if (!descriptor) {
                // The container did not name a pixel format; the sequence header is checked again when parsed.
                layout.bit_depth = std::max(8, parameters->bits_per_raw_sample);
                return layout;
            }
            layout.bit_depth = descriptor->comp[0].depth;
            if (descriptor->nb_components == 1)
                layout.chroma = rocDecVideoChromaFormat_Monochrome;
            else if (descriptor->log2_chroma_w == 1 && descriptor->log2_chroma_h == 1)
                layout.chroma = rocDecVideoChromaFormat_420;
            else if (descriptor->log2_chroma_w == 1 && descriptor->log2_chroma_h == 0)
                layout.chroma = rocDecVideoChromaFormat_422;
            else if (descriptor->log2_chroma_w == 0 && descriptor->log2_chroma_h == 0)
                layout.chroma = rocDecVideoChromaFormat_444;
            else
                return std::unexpected(std::format("pixel format {} has a chroma layout rocDecode does not define",
                                                   descriptor->name));
            return layout;
        }

        [[nodiscard]] std::expected<RocdecDecodeCaps, std::string> queryCaps(const rocDecVideoCodec codec,
                                                                             const rocDecVideoChromaFormat chroma,
                                                                             const int bit_depth) {
            int device = 0;
            if (const hipError_t result = hipGetDevice(&device); result != hipSuccess)
                return std::unexpected(std::format("hipGetDevice failed: {}", hipGetErrorString(result)));
            RocdecDecodeCaps caps{};
            caps.device_id = static_cast<uint8_t>(device);
            caps.codec_type = codec;
            caps.chroma_format = chroma;
            caps.bit_depth_minus_8 = static_cast<uint32_t>(std::max(0, bit_depth - 8));
            if (const rocDecStatus status = rocDecGetDecoderCaps(&caps); status != ROCDEC_SUCCESS)
                return std::unexpected(std::format("rocDecGetDecoderCaps failed: {}", rocDecGetErrorName(status)));
            return caps;
        }

        // Empty when the GPU decodes the codec, layout and size; otherwise the reason it does not.
        [[nodiscard]] std::expected<std::string, std::string> capsRefusal(const CodecMapping& codec,
                                                                          const rocDecVideoChromaFormat chroma,
                                                                          const int bit_depth,
                                                                          const int width,
                                                                          const int height) {
            const auto caps = queryCaps(codec.codec, chroma, bit_depth);
            if (!caps)
                return std::unexpected(caps.error());
            if (!caps->is_supported)
                return std::format("rocDecode does not decode {} {} {}-bit on this GPU", codec.name,
                                   chromaName(chroma), bit_depth);
            const auto coded = [&](const int size) {
                return (size + codec.coded_alignment - 1) / codec.coded_alignment * codec.coded_alignment;
            };
            if (width > 0 && height > 0 &&
                (static_cast<uint32_t>(width) > caps->max_width || static_cast<uint32_t>(height) > caps->max_height ||
                 coded(width) < caps->min_width || coded(height) < caps->min_height)) {
                return std::format("{}x{} {} is outside rocDecode's {}x{} to {}x{} on this GPU", width, height,
                                   codec.name, caps->min_width, caps->min_height, caps->max_width,
                                   caps->max_height);
            }
            return std::string{};
        }

        [[nodiscard]] AVPixelFormat surfacePixelFormat(const int bit_depth) {
            if (bit_depth <= 8)
                return AV_PIX_FMT_NV12;
            return bit_depth <= 10 ? AV_PIX_FMT_P010 : AV_PIX_FMT_P012;
        }

        AVBufferRef* allocateDeviceBuffer(void*, const size_t size) {
            void* pointer = nullptr;
            if (hipMalloc(&pointer, size) != hipSuccess)
                return nullptr;
            AVBufferRef* const buffer = av_buffer_create(
                static_cast<uint8_t*>(pointer), size,
                [](void*, uint8_t* data) { (void)hipFree(data); }, nullptr, 0);
            if (!buffer)
                (void)hipFree(pointer);
            return buffer;
        }

        [[nodiscard]] bool specified(const int code) { return code != 2; }

        // rocDecode's capability query takes no profile, and rocDecode 1.10 decodes
        // an H.264 High 4:4:4 Predictive stream with 4:2:0 chroma (x264's lossless
        // mode) without an error but with wrong pixels. Accept only the profiles
        // the video engine implements.
        [[nodiscard]] bool profileSupported(const AVCodecID codec, const int profile) {
            if (profile == AV_PROFILE_UNKNOWN)
                return true;
            switch (codec) {
            case AV_CODEC_ID_H264: {
                const int base = profile & ~(AV_PROFILE_H264_CONSTRAINED | AV_PROFILE_H264_INTRA);
                return base == AV_PROFILE_H264_BASELINE || base == AV_PROFILE_H264_MAIN ||
                       base == AV_PROFILE_H264_HIGH;
            }
            case AV_CODEC_ID_HEVC:
                return profile == AV_PROFILE_HEVC_MAIN || profile == AV_PROFILE_HEVC_MAIN_10 ||
                       profile == AV_PROFILE_HEVC_MAIN_STILL_PICTURE;
            default:
                return true;
            }
        }

        // An embedded clip read through FFmpeg's demuxers.
        struct MemoryInput {
            std::span<const uint8_t> data;
            int64_t position = 0;
        };

        int readMemory(void* const opaque, uint8_t* const buffer, const int size) {
            auto& input = *static_cast<MemoryInput*>(opaque);
            const int64_t left = static_cast<int64_t>(input.data.size()) - input.position;
            if (left <= 0)
                return AVERROR_EOF;
            const int count = static_cast<int>(std::min<int64_t>(size, left));
            std::memcpy(buffer, input.data.data() + input.position, count);
            input.position += count;
            return count;
        }

        int64_t seekMemory(void* const opaque, int64_t offset, const int whence) {
            auto& input = *static_cast<MemoryInput*>(opaque);
            const auto size = static_cast<int64_t>(input.data.size());
            switch (whence & ~AVSEEK_FORCE) {
            case AVSEEK_SIZE: return size;
            case SEEK_SET: break;
            case SEEK_CUR: offset += input.position; break;
            case SEEK_END: offset += size; break;
            default: return AVERROR(EINVAL);
            }
            if (offset < 0 || offset > size)
                return AVERROR(EINVAL);
            input.position = offset;
            return offset;
        }

        struct MemoryDemuxer {
            MemoryInput input;
            AVFormatContext* format = nullptr;
            AVIOContext* io = nullptr;
            AVPacket* packet = nullptr;

            ~MemoryDemuxer() {
                av_packet_free(&packet);
                avformat_close_input(&format);
                if (io)
                    av_freep(&io->buffer);
                avio_context_free(&io);
            }
        };

        // MD5 of a host frame packed without row padding, as FFmpeg's framemd5 muxer hashes it.
        [[nodiscard]] std::string frameMd5(const AVFrame* const frame) {
            const auto format = static_cast<AVPixelFormat>(frame->format);
            const int size = av_image_get_buffer_size(format, frame->width, frame->height, 1);
            if (size <= 0)
                return {};
            std::vector<uint8_t> packed(static_cast<size_t>(size));
            if (av_image_copy_to_buffer(packed.data(), size, frame->data, frame->linesize, format, frame->width,
                                        frame->height, 1) < 0)
                return {};
            uint8_t digest[16];
            av_md5_sum(digest, packed.data(), static_cast<size_t>(size));
            std::string hex;
            for (const uint8_t byte : digest)
                std::format_to(std::back_inserter(hex), "{:02x}", byte);
            return hex;
        }

        // The 12-bit AV1 profile shares the 16-bit surfaces of the 10-bit clip.
        [[nodiscard]] int referenceBitDepth(const int bit_depth) { return bit_depth > 8 ? 10 : 8; }

        [[nodiscard]] const RocDecodeReferenceClip* referenceClip(const AVCodecID codec, const int bit_depth) {
            for (const RocDecodeReferenceClip& clip : rocDecodeReferenceClips())
                if (clip.codec == codec && clip.bit_depth == referenceBitDepth(bit_depth))
                    return &clip;
            return nullptr;
        }

        // Results of the reference clip check, by device, codec and reference bit
        // depth: an empty string when the frames matched, else the CPU reason.
        struct SelfCheckRegistry {
            std::mutex mutex;
            std::map<std::tuple<int, AVCodecID, int>, std::string> results;
            std::map<std::pair<AVCodecID, int>, std::array<std::string, 2>> overrides;
            int runs = 0;
        };

        SelfCheckRegistry& selfCheckRegistry() {
            static auto* const registry = new SelfCheckRegistry();
            return *registry;
        }

    } // namespace

    struct RocDecodeDecoder::State {
        // rocDecode 1.10 reports limited range and matrix 0 for full-range VP9 and
        // AV1; FFmpeg's stream parameters carry the signalled values.
        bool stream_signal = false;
        AVColorRange color_range = AVCOL_RANGE_UNSPECIFIED;
        AVColorSpace colorspace = AVCOL_SPC_UNSPECIFIED;
        AVColorPrimaries color_primaries = AVCOL_PRI_UNSPECIFIED;
        AVColorTransferCharacteristic color_trc = AVCOL_TRC_UNSPECIFIED;
        AVChromaLocation chroma_location = AVCHROMA_LOC_UNSPECIFIED;
        CodecMapping codec{};
        int device = 0;
        hipStream_t stream = nullptr;
        AVBSFContext* bsf = nullptr;
        AVPacket* filtered = nullptr;
        std::vector<uint8_t> sequence_header;
        std::vector<uint8_t> scratch;
        bool send_sequence_header = false;

        RocdecVideoParser parser = nullptr;
        rocDecDecoderHandle decoder = nullptr;
        RocdecVideoFormat format{};
        uint32_t decode_surfaces = 0;

        AVBufferPool* pool = nullptr;
        AVPixelFormat pixel_format = AV_PIX_FMT_NONE;
        int width = 0;
        int height = 0;
        int bytes_per_sample = 1;
        int pitch = 0;
        std::deque<AVFrame*> ready;

        bool timestamps = false;
        // Frames of packets the demuxer marks for discard (outside an MP4 edit
        // list) are dropped, as FFmpeg's decoders do.
        std::unordered_set<int64_t> discard;
        bool draining = false;
        bool logged_concealment = false;
        std::string callback_error;

        ~State() {
            clearReady();
            if (parser)
                rocDecDestroyVideoParser(parser);
            if (decoder)
                rocDecDestroyDecoder(decoder);
            av_buffer_pool_uninit(&pool);
            av_bsf_free(&bsf);
            av_packet_free(&filtered);
            if (stream)
                (void)hipStreamDestroy(stream);
        }

        void clearReady() {
            for (AVFrame*& frame : ready)
                av_frame_free(&frame);
            ready.clear();
        }

        [[nodiscard]] bool createParser() {
            RocdecParserParams params{};
            params.codec_type = codec.codec;
            params.max_num_decode_surfaces = 1;
            params.user_data = this;
            params.pfn_sequence_callback = [](void* user, RocdecVideoFormat* format) {
                return static_cast<State*>(user)->handleSequence(format);
            };
            params.pfn_decode_picture = [](void* user, RocdecPicParams* picture) {
                return static_cast<State*>(user)->handleDecode(picture);
            };
            params.pfn_display_picture = [](void* user, RocdecParserDispInfo* display) {
                return static_cast<State*>(user)->handleDisplay(display);
            };
            const rocDecStatus status = rocDecCreateVideoParser(&parser, &params);
            if (status != ROCDEC_SUCCESS) {
                parser = nullptr;
                callback_error = std::format("rocDecCreateVideoParser failed: {}", rocDecGetErrorName(status));
                return false;
            }
            return true;
        }

        int handleSequence(const RocdecVideoFormat* const sequence) {
            const int bit_depth = sequence->bit_depth_luma_minus8 + 8;
            const int coded_width = static_cast<int>(sequence->coded_width);
            const int coded_height = static_cast<int>(sequence->coded_height);
            if (decoder && sequence->coded_width == format.coded_width &&
                sequence->coded_height == format.coded_height &&
                sequence->chroma_format == format.chroma_format &&
                sequence->bit_depth_luma_minus8 == format.bit_depth_luma_minus8 &&
                std::memcmp(&sequence->display_area, &format.display_area, sizeof(format.display_area)) == 0 &&
                sequence->min_num_decode_surfaces <= decode_surfaces) {
                format = *sequence;
                return static_cast<int>(decode_surfaces);
            }

            const auto refusal = capsRefusal(codec, sequence->chroma_format, bit_depth, coded_width, coded_height);
            if (!refusal) {
                callback_error = refusal.error();
                return 0;
            }
            if (!refusal->empty()) {
                callback_error = "The stream's sequence header is not decodable: " + *refusal;
                return 0;
            }

            if (decoder) {
                rocDecDestroyDecoder(decoder);
                decoder = nullptr;
            }
            format = *sequence;
            decode_surfaces = sequence->min_num_decode_surfaces;
            width = sequence->display_area.right - sequence->display_area.left;
            height = sequence->display_area.bottom - sequence->display_area.top;
            pixel_format = surfacePixelFormat(bit_depth);
            bytes_per_sample = bit_depth > 8 ? 2 : 1;

            RocDecoderCreateInfo info{};
            info.device_id = static_cast<uint8_t>(device);
            info.codec_type = codec.codec;
            info.chroma_format = sequence->chroma_format;
            info.bit_depth_minus_8 = sequence->bit_depth_luma_minus8;
            info.output_format = bit_depth > 8 ? rocDecVideoSurfaceFormat_P016 : rocDecVideoSurfaceFormat_NV12;
            info.num_decode_surfaces = decode_surfaces;
            info.width = sequence->coded_width;
            info.height = sequence->coded_height;
            info.max_width = sequence->coded_width;
            info.max_height = sequence->coded_height;
            info.display_rect.left = static_cast<int16_t>(sequence->display_area.left);
            info.display_rect.top = static_cast<int16_t>(sequence->display_area.top);
            info.display_rect.right = static_cast<int16_t>(sequence->display_area.right);
            info.display_rect.bottom = static_cast<int16_t>(sequence->display_area.bottom);
            info.target_width = static_cast<uint32_t>((width + 1) & ~1);
            info.target_height = static_cast<uint32_t>((height + 1) & ~1);
            info.num_output_surfaces = 1;
            if (const rocDecStatus status = rocDecCreateDecoder(&decoder, &info); status != ROCDEC_SUCCESS) {
                decoder = nullptr;
                callback_error = std::format("rocDecCreateDecoder failed for {}x{} {}: {}", coded_width,
                                             coded_height, codec.name, rocDecGetErrorName(status));
                return 0;
            }

            const int row_bytes = ((width + 1) & ~1) * bytes_per_sample;
            pitch = (row_bytes + PITCH_ALIGNMENT - 1) / PITCH_ALIGNMENT * PITCH_ALIGNMENT;
            const size_t frame_bytes = static_cast<size_t>(pitch) * (height + (height + 1) / 2);
            av_buffer_pool_uninit(&pool);
            pool = av_buffer_pool_init2(frame_bytes, nullptr, allocateDeviceBuffer, nullptr);
            if (!pool) {
                callback_error = "Failed to create the decoded frame pool";
                return 0;
            }
            return static_cast<int>(decode_surfaces);
        }

        int handleDecode(RocdecPicParams* const picture) {
            if (!decoder) {
                callback_error = "rocDecode delivered a picture before its sequence header";
                return 0;
            }
            if (const rocDecStatus status = rocDecDecodeFrame(decoder, picture); status != ROCDEC_SUCCESS) {
                callback_error = std::format("rocDecDecodeFrame failed: {}", rocDecGetErrorName(status));
                return 0;
            }
            return 1;
        }

        int handleDisplay(const RocdecParserDispInfo* const display) {
            if (!display)
                return 1;
            if (timestamps && discard.erase(static_cast<int64_t>(display->pts)))
                return 1;
            RocdecProcParams processing{};
            processing.progressive_frame = display->progressive_frame;
            processing.top_field_first = display->top_field_first;
            void* planes[3] = {};
            uint32_t pitches[3] = {};
            if (const rocDecStatus status =
                    rocDecGetVideoFrame(decoder, display->picture_index, planes, pitches, &processing);
                status != ROCDEC_SUCCESS) {
                callback_error = std::format("rocDecGetVideoFrame failed: {}", rocDecGetErrorName(status));
                return 0;
            }
            RocdecDecodeStatus decode_status{};
            if (const rocDecStatus status = rocDecGetDecodeStatus(decoder, display->picture_index, &decode_status);
                status != ROCDEC_SUCCESS) {
                callback_error = std::format("rocDecGetDecodeStatus failed: {}", rocDecGetErrorName(status));
                return 0;
            }
            if (decode_status.decode_status == rocDecodeStatus_Error) {
                callback_error = "rocDecode reported a picture decode error";
                return 0;
            }
            if (decode_status.decode_status == rocDecodeStatus_Error_Concealed && !logged_concealment) {
                LOG_WARN("rocDecode concealed a decode error in the video stream");
                logged_concealment = true;
            }

            AVBufferRef* buffer = av_buffer_pool_get(pool);
            if (!buffer) {
                callback_error = "Failed to allocate a decoded frame in device memory";
                return 0;
            }
            uint8_t* const luma = buffer->data;
            uint8_t* const chroma = luma + static_cast<size_t>(pitch) * height;
            const size_t luma_bytes = static_cast<size_t>(width) * bytes_per_sample;
            const size_t chroma_bytes = static_cast<size_t>((width + 1) & ~1) * bytes_per_sample;
            const int left = format.display_area.left;
            const int top = format.display_area.top;
            const auto* const luma_source = static_cast<const uint8_t*>(planes[0]) +
                                            static_cast<size_t>(top) * pitches[0] + left * bytes_per_sample;
            const auto* const chroma_source = static_cast<const uint8_t*>(planes[1]) +
                                              static_cast<size_t>(top / 2) * pitches[1] + left * bytes_per_sample;
            hipError_t result = hipMemcpy2DAsync(luma, pitch, luma_source, pitches[0], luma_bytes, height,
                                                 hipMemcpyDeviceToDevice, stream);
            if (result == hipSuccess)
                result = hipMemcpy2DAsync(chroma, pitch, chroma_source, pitches[1], chroma_bytes, (height + 1) / 2,
                                          hipMemcpyDeviceToDevice, stream);
            if (result == hipSuccess)
                result = hipStreamSynchronize(stream);
            if (result != hipSuccess) {
                av_buffer_unref(&buffer);
                callback_error = std::format("Copying a decoded frame failed: {}", hipGetErrorString(result));
                return 0;
            }

            AVFrame* frame = av_frame_alloc();
            if (!frame) {
                av_buffer_unref(&buffer);
                callback_error = "Failed to allocate a video frame";
                return 0;
            }
            frame->buf[0] = buffer;
            frame->data[0] = luma;
            frame->data[1] = chroma;
            frame->linesize[0] = pitch;
            frame->linesize[1] = pitch;
            frame->format = pixel_format;
            frame->width = width;
            frame->height = height;
            frame->pts = timestamps ? static_cast<int64_t>(display->pts) : AV_NOPTS_VALUE;
            frame->best_effort_timestamp = frame->pts;
            const auto& signal = format.video_signal_description;
            if (stream_signal) {
                frame->color_range = color_range;
                frame->colorspace = colorspace;
                frame->color_primaries = color_primaries;
                frame->color_trc = color_trc;
            } else {
                frame->color_range = signal.video_full_range_flag ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
                frame->colorspace = specified(signal.matrix_coefficients)
                                        ? static_cast<AVColorSpace>(signal.matrix_coefficients)
                                        : colorspace;
                frame->color_primaries = specified(signal.color_primaries)
                                             ? static_cast<AVColorPrimaries>(signal.color_primaries)
                                             : color_primaries;
                frame->color_trc = specified(signal.transfer_characteristics)
                                       ? static_cast<AVColorTransferCharacteristic>(signal.transfer_characteristics)
                                       : color_trc;
            }
            frame->chroma_location = chroma_location;
            ready.push_back(frame);
            return 1;
        }
    };

    std::expected<std::string, std::string> RocDecodeDecoder::unsupportedReason(
        const AVCodecParameters* const parameters) {
        const auto codec = rocDecodeCodec(parameters->codec_id);
        if (!codec)
            return std::format("rocDecode has no {} decoder", avcodec_get_name(parameters->codec_id));
        if (!profileSupported(parameters->codec_id, parameters->profile)) {
            const char* const profile = avcodec_profile_name(parameters->codec_id, parameters->profile);
            return std::format("rocDecode does not decode the {} {} profile", codec->name,
                               profile ? std::string(profile) : std::to_string(parameters->profile));
        }
        const auto layout = streamLayout(parameters);
        if (!layout)
            return layout.error();
        const auto refusal =
            capsRefusal(*codec, layout->chroma, layout->bit_depth, parameters->width, parameters->height);
        if (!refusal || !refusal->empty())
            return refusal;
        return selfCheckRefusal(parameters->codec_id, layout->bit_depth);
    }

    // rocDecode reports success for every picture even where a GPU's video
    // driver returns wrong pixels, so the frames of a short valid clip are
    // compared with FFmpeg's software decode before the codec is trusted.
    std::expected<std::string, std::string> RocDecodeDecoder::selfCheckRefusal(const AVCodecID codec_id,
                                                                               const int bit_depth) {
        const RocDecodeReferenceClip* const clip = referenceClip(codec_id, bit_depth);
        if (!clip)
            return std::string{};
        int device = 0;
        if (const hipError_t result = hipGetDevice(&device); result != hipSuccess)
            return std::unexpected(std::format("hipGetDevice failed: {}", hipGetErrorString(result)));

        SelfCheckRegistry& registry = selfCheckRegistry();
        const std::lock_guard lock(registry.mutex);
        const auto key = std::tuple{device, codec_id, clip->bit_depth};
        if (const auto found = registry.results.find(key); found != registry.results.end())
            return found->second;

        std::array<std::string, 2> expected{clip->frame_md5[0], clip->frame_md5[1]};
        if (const auto found = registry.overrides.find({codec_id, clip->bit_depth});
            found != registry.overrides.end())
            expected = found->second;
        const char* const codec_name = rocDecodeCodec(codec_id)->name;
        const auto start = std::chrono::steady_clock::now();
        ++registry.runs;

        auto demuxer = std::make_unique<MemoryDemuxer>();
        demuxer->input.data = clip->data;
        constexpr int IO_BUFFER_SIZE = 4096;
        auto* const io_buffer = static_cast<uint8_t*>(av_malloc(IO_BUFFER_SIZE));
        if (!io_buffer)
            return std::unexpected("Failed to allocate the reference clip reader");
        demuxer->io = avio_alloc_context(io_buffer, IO_BUFFER_SIZE, 0, &demuxer->input, readMemory, nullptr,
                                         seekMemory);
        if (!demuxer->io) {
            av_free(io_buffer);
            return std::unexpected("Failed to allocate the reference clip reader");
        }
        demuxer->format = avformat_alloc_context();
        demuxer->packet = av_packet_alloc();
        if (!demuxer->format || !demuxer->packet)
            return std::unexpected("Failed to allocate the reference clip reader");
        demuxer->format->pb = demuxer->io;
        demuxer->format->flags |= AVFMT_FLAG_CUSTOM_IO;
        if (avformat_open_input(&demuxer->format, nullptr, av_find_input_format(clip->demuxer), nullptr) < 0 ||
            demuxer->format->nb_streams != 1)
            return std::unexpected(std::format("FFmpeg could not read the {} reference clip", codec_name));
        const AVStream* const stream = demuxer->format->streams[0];

        auto decoder = open(stream->codecpar, stream->time_base);
        if (!decoder)
            return std::unexpected(std::format("rocDecode could not open the {} reference clip: {}", codec_name,
                                               decoder.error()));
        std::vector<std::string> frames;
        AVFrame* frame = av_frame_alloc();
        AVFrame* host = av_frame_alloc();
        std::string error = frame && host ? "" : "failed to allocate a video frame";
        bool drained = false;
        while (error.empty()) {
            const int received = (*decoder)->receiveFrame(frame);
            if (received == 0) {
                if (RocDecodeDecoder::transferToHost(host, frame) < 0)
                    error = "copying a frame to host memory failed";
                else
                    frames.push_back(frameMd5(host));
                continue;
            }
            if (received == AVERROR_EOF)
                break;
            if (received != AVERROR(EAGAIN) || drained) {
                error = (*decoder)->lastError();
                break;
            }
            int read = av_read_frame(demuxer->format, demuxer->packet);
            drained = read < 0;
            read = (*decoder)->sendPacket(drained ? nullptr : demuxer->packet);
            av_packet_unref(demuxer->packet);
            if (read < 0)
                error = (*decoder)->lastError();
        }
        av_frame_free(&frame);
        av_frame_free(&host);
        if (!error.empty())
            return std::unexpected(
                std::format("rocDecode could not decode the {} reference clip: {}", codec_name, error));

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        hipDeviceProp_t properties{};
        const bool named = hipGetDeviceProperties(&properties, device) == hipSuccess;
        const std::string gpu = named ? std::format("{} ({})", properties.name, properties.gcnArchName)
                                      : std::format("HIP device {}", device);
        std::size_t matching = 0;
        while (matching < frames.size() && matching < expected.size() && frames[matching] == expected[matching])
            ++matching;
        std::string refusal;
        if (frames.size() == expected.size() && matching == expected.size()) {
            LOG_INFO("rocDecode decodes {} {}-bit correctly on {} (reference clip checked in {} ms)", codec_name,
                     clip->bit_depth, gpu, elapsed);
        } else {
            refusal = std::format("rocDecode produced wrong frames for {} {}-bit on this GPU and driver", codec_name,
                                  clip->bit_depth);
            LOG_WARN("{}: {} decoded {} frames of a reference clip, and frame {} differs from FFmpeg's software "
                     "decode of the same clip (checked in {} ms)",
                     refusal, gpu, frames.size(), matching + 1, elapsed);
        }
        registry.results.emplace(key, refusal);
        return refusal;
    }

    void RocDecodeDecoder::overrideSelfCheckDigestsForTesting(const AVCodecID codec, const int bit_depth,
                                                              const std::array<std::string, 2>& frame_md5) {
        SelfCheckRegistry& registry = selfCheckRegistry();
        const std::lock_guard lock(registry.mutex);
        registry.overrides[{codec, referenceBitDepth(bit_depth)}] = frame_md5;
    }

    void RocDecodeDecoder::resetSelfCheckForTesting() {
        SelfCheckRegistry& registry = selfCheckRegistry();
        const std::lock_guard lock(registry.mutex);
        registry.results.clear();
        registry.overrides.clear();
        registry.runs = 0;
    }

    int RocDecodeDecoder::selfCheckRunsForTesting() {
        SelfCheckRegistry& registry = selfCheckRegistry();
        const std::lock_guard lock(registry.mutex);
        return registry.runs;
    }

    std::expected<std::unique_ptr<RocDecodeDecoder>, std::string> RocDecodeDecoder::create(
        const AVCodecParameters* const parameters, const AVRational time_base) {
        const auto refusal = unsupportedReason(parameters);
        if (!refusal)
            return std::unexpected(refusal.error());
        if (!refusal->empty())
            return std::unexpected(*refusal);
        return open(parameters, time_base);
    }

    std::expected<std::unique_ptr<RocDecodeDecoder>, std::string> RocDecodeDecoder::createWithoutSelfCheckForTesting(
        const AVCodecParameters* const parameters, const AVRational time_base) {
        const auto codec = rocDecodeCodec(parameters->codec_id);
        if (!codec)
            return std::unexpected(std::format("rocDecode has no {} decoder", avcodec_get_name(parameters->codec_id)));
        return open(parameters, time_base);
    }

    std::expected<std::unique_ptr<RocDecodeDecoder>, std::string> RocDecodeDecoder::open(
        const AVCodecParameters* const parameters, const AVRational time_base) {
        std::unique_ptr<RocDecodeDecoder> decoder(new RocDecodeDecoder());
        decoder->state_ = std::make_unique<State>();
        State& state = *decoder->state_;
        state.stream_signal = parameters->codec_id == AV_CODEC_ID_VP9 || parameters->codec_id == AV_CODEC_ID_AV1;
        state.color_range = parameters->color_range;
        state.colorspace = parameters->color_space;
        state.color_primaries = parameters->color_primaries;
        state.color_trc = parameters->color_trc;
        state.chroma_location = parameters->chroma_location;
        state.codec = *rocDecodeCodec(parameters->codec_id);
        if (const hipError_t result = hipGetDevice(&state.device); result != hipSuccess)
            return std::unexpected(std::format("hipGetDevice failed: {}", hipGetErrorString(result)));
        if (const hipError_t result = hipStreamCreateWithFlags(&state.stream, hipStreamNonBlocking);
            result != hipSuccess) {
            state.stream = nullptr;
            return std::unexpected(std::format("hipStreamCreate failed: {}", hipGetErrorString(result)));
        }

        // rocDecode parses Annex B H.264 and HEVC; MP4 and Matroska store length-prefixed NAL units.
        const char* const filter_name = parameters->codec_id == AV_CODEC_ID_H264   ? "h264_mp4toannexb"
                                        : parameters->codec_id == AV_CODEC_ID_HEVC ? "hevc_mp4toannexb"
                                                                                   : nullptr;
        if (filter_name) {
            const AVBitStreamFilter* const filter = av_bsf_get_by_name(filter_name);
            if (!filter)
                return std::unexpected(std::format("FFmpeg has no {} bitstream filter", filter_name));
            if (av_bsf_alloc(filter, &state.bsf) < 0 ||
                avcodec_parameters_copy(state.bsf->par_in, parameters) < 0)
                return std::unexpected(std::format("Failed to set up the {} bitstream filter", filter_name));
            state.bsf->time_base_in = time_base;
            if (av_bsf_init(state.bsf) < 0)
                return std::unexpected(std::format("Failed to initialize the {} bitstream filter", filter_name));
            state.filtered = av_packet_alloc();
            if (!state.filtered)
                return std::unexpected("Failed to allocate a video packet");
        }

        // MP4 and Matroska carry the AV1 sequence header in the av1C record, which
        // packets need not repeat; the parser needs it before the first frame.
        if (parameters->codec_id == AV_CODEC_ID_AV1 && parameters->extradata_size > 4 &&
            (parameters->extradata[0] & 0x80)) {
            state.sequence_header.assign(parameters->extradata + 4,
                                         parameters->extradata + parameters->extradata_size);
            state.send_sequence_header = true;
        }

        if (!state.createParser())
            return std::unexpected(state.callback_error);
        return decoder;
    }

    RocDecodeDecoder::~RocDecodeDecoder() = default;

    int RocDecodeDecoder::parse(const uint8_t* data, int size, const int64_t pts, const bool end_of_stream,
                                const bool discard) {
        State& state = *state_;
        if (!state.parser) {
            error_ = "rocDecode has no parser: " + state.callback_error;
            return AVERROR_EXTERNAL;
        }
        if (discard && pts != AV_NOPTS_VALUE)
            state.discard.insert(pts);
        if (state.send_sequence_header && data && size > 0) {
            state.scratch.assign(state.sequence_header.begin(), state.sequence_header.end());
            state.scratch.insert(state.scratch.end(), data, data + size);
            data = state.scratch.data();
            size = static_cast<int>(state.scratch.size());
            state.send_sequence_header = false;
        }
        RocdecSourceDataPacket packet{};
        packet.payload = data;
        packet.payload_size = static_cast<uint32_t>(size);
        if (end_of_stream)
            packet.flags |= ROCDEC_PKT_ENDOFSTREAM;
        if (pts != AV_NOPTS_VALUE) {
            packet.flags |= ROCDEC_PKT_TIMESTAMP;
            packet.pts = static_cast<RocdecTimeStamp>(pts);
            state.timestamps = true;
        }
        state.callback_error.clear();
        const rocDecStatus status = rocDecParseVideoData(state.parser, &packet);
        if (!state.callback_error.empty()) {
            error_ = state.callback_error;
            return AVERROR_EXTERNAL;
        }
        if (status != ROCDEC_SUCCESS) {
            error_ = std::format("rocDecParseVideoData failed: {}", rocDecGetErrorName(status));
            return AVERROR_INVALIDDATA;
        }
        return 0;
    }

    int RocDecodeDecoder::sendPacket(const AVPacket* const packet) {
        State& state = *state_;
        if (!packet || packet->size == 0) {
            if (state.draining)
                return AVERROR_EOF;
            state.draining = true;
            return parse(nullptr, 0, AV_NOPTS_VALUE, true, false);
        }
        if (state.draining)
            return AVERROR_EOF;
        if (!state.bsf)
            return parse(packet->data, packet->size, packet->pts, false, packet->flags & AV_PKT_FLAG_DISCARD);

        AVPacket* input = av_packet_clone(packet);
        if (!input) {
            error_ = "Failed to copy a video packet";
            return AVERROR(ENOMEM);
        }
        int result = av_bsf_send_packet(state.bsf, input);
        av_packet_free(&input);
        if (result < 0) {
            error_ = "Bitstream filter rejected a video packet";
            return result;
        }
        while ((result = av_bsf_receive_packet(state.bsf, state.filtered)) == 0) {
            result = parse(state.filtered->data, state.filtered->size, state.filtered->pts, false,
                           state.filtered->flags & AV_PKT_FLAG_DISCARD);
            av_packet_unref(state.filtered);
            if (result < 0)
                return result;
        }
        if (result != AVERROR(EAGAIN)) {
            error_ = "Bitstream filter failed on a video packet";
            return result;
        }
        return 0;
    }

    int RocDecodeDecoder::receiveFrame(AVFrame* const frame) {
        State& state = *state_;
        if (state.ready.empty())
            return state.draining ? AVERROR_EOF : AVERROR(EAGAIN);
        AVFrame* next = state.ready.front();
        state.ready.pop_front();
        av_frame_unref(frame);
        av_frame_move_ref(frame, next);
        av_frame_free(&next);
        return 0;
    }

    void RocDecodeDecoder::flush() {
        State& state = *state_;
        state.clearReady();
        state.discard.clear();
        // A new parser forgets every reference picture; the decoder and its
        // surfaces stay, and are reused when the next sequence header matches.
        if (state.parser)
            rocDecDestroyVideoParser(state.parser);
        state.parser = nullptr;
        if (!state.createParser())
            error_ = state.callback_error;
        if (state.bsf)
            av_bsf_flush(state.bsf);
        state.send_sequence_header = !state.sequence_header.empty();
        state.draining = false;
    }

    int RocDecodeDecoder::transferToHost(AVFrame* const dst, const AVFrame* const src) {
        av_frame_unref(dst);
        dst->format = src->format;
        dst->width = src->width;
        dst->height = src->height;
        if (const int result = av_frame_get_buffer(dst, 0); result < 0)
            return result;
        const AVPixFmtDescriptor* const descriptor = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(src->format));
        const size_t bytes_per_sample = descriptor && descriptor->comp[0].depth > 8 ? 2 : 1;
        const size_t luma_bytes = static_cast<size_t>(src->width) * bytes_per_sample;
        const size_t chroma_bytes = static_cast<size_t>((src->width + 1) & ~1) * bytes_per_sample;
        hipError_t result = hipMemcpy2D(dst->data[0], dst->linesize[0], src->data[0], src->linesize[0], luma_bytes,
                                        src->height, hipMemcpyDeviceToHost);
        if (result == hipSuccess)
            result = hipMemcpy2D(dst->data[1], dst->linesize[1], src->data[1], src->linesize[1], chroma_bytes,
                                 (src->height + 1) / 2, hipMemcpyDeviceToHost);
        if (result != hipSuccess) {
            LOG_ERROR("Copying a rocDecode frame to host memory failed: {}", hipGetErrorString(result));
            av_frame_unref(dst);
            return AVERROR_EXTERNAL;
        }
        return 0;
    }

} // namespace lfs::io::video
