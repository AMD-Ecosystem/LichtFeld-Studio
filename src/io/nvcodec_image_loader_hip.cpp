/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// NvCodecImageLoader for AMD GPUs. JPEG decodes on the GPU with rocJPEG. ROCm
// has no JPEG encoder and no JPEG 2000 codec, so JPEG encodes run on the CPU
// (libjpeg) and the lossless HTJ2K caches are coded on the CPU (OpenJPH); the
// sample conversions around them stay on the GPU, as with nvImageCodec.

#include "io/nvcodec_image_loader.hpp"
#include "core/assert.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/mask_normalization.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "cuda/image_format_kernels.cuh"
#include "cuda/masked_image_fill.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <csetjmp>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#if LFS_HAVE_ROCJPEG
#include <rocjpeg/rocjpeg.h>
#endif

#include <jpeglib.h>
#include <jerror.h>
#include <openjph/ojph_codestream.h>
#include <openjph/ojph_file.h>
#include <openjph/ojph_mem.h>
#include <openjph/ojph_params.h>

namespace lfs::io {

    namespace {
        constexpr size_t DECODER_POOL_SIZE = 8;
        constexpr int LANCZOS_KERNEL_SIZE = 2;

        std::atomic<size_t> live_loaders{0};

        void check_hip(const cudaError_t status, const char* const what) {
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
            }
        }

        struct JpegError {
            jpeg_error_mgr base{};
            std::jmp_buf jump{};
            char message[JMSG_LENGTH_MAX] = {};
        };

        [[noreturn]] void jpeg_error_exit(j_common_ptr info) {
            auto* const error = reinterpret_cast<JpegError*>(info->err);
            (*info->err->format_message)(info, error->message);
            std::longjmp(error->jump, 1);
        }

        // Compressed output in a buffer this class owns, so libjpeg never frees or
        // reallocates it and an error part way through leaves nothing to double free.
        struct JpegVectorDestination {
            jpeg_destination_mgr base{};
            std::vector<uint8_t> bytes;

            static JpegVectorDestination& of(j_compress_ptr codec) {
                return *static_cast<JpegVectorDestination*>(codec->client_data);
            }

            static void init(j_compress_ptr codec) {
                auto& self = of(codec);
                self.grow(codec, 0, size_t{1} << 16);
            }

            static boolean empty(j_compress_ptr codec) {
                auto& self = of(codec);
                const size_t used = self.bytes.size();
                self.grow(codec, used, used * 2);
                return TRUE;
            }

            static void term(j_compress_ptr codec) {
                auto& self = of(codec);
                self.bytes.resize(self.bytes.size() - self.base.free_in_buffer);
            }

            void grow(j_compress_ptr codec, const size_t used, const size_t capacity) {
                bool grown = true;
                try {
                    bytes.resize(capacity);
                } catch (const std::bad_alloc&) {
                    grown = false;
                }
                if (!grown) {
                    ERREXIT(codec, JERR_OUT_OF_MEMORY);
                }
                base.next_output_byte = bytes.data() + used;
                base.free_in_buffer = capacity - used;
            }
        };

        struct JpegMemoryEncoder {
            jpeg_compress_struct codec{};
            JpegError error{};
            JpegVectorDestination destination{};
            bool created = false;

            ~JpegMemoryEncoder() {
                if (created)
                    jpeg_destroy_compress(&codec);
            }

            // The jump stays inside a frame with no destructors to skip.
            bool encode(const uint8_t* const pixels, const int width, const int height,
                        const int channels, const int quality) {
                codec.err = jpeg_std_error(&error.base);
                error.base.error_exit = jpeg_error_exit;
                if (setjmp(error.jump)) {
                    return false;
                }
                jpeg_create_compress(&codec);
                created = true;
                destination.base.init_destination = JpegVectorDestination::init;
                destination.base.empty_output_buffer = JpegVectorDestination::empty;
                destination.base.term_destination = JpegVectorDestination::term;
                codec.dest = &destination.base;
                codec.client_data = &destination;
                codec.image_width = static_cast<JDIMENSION>(width);
                codec.image_height = static_cast<JDIMENSION>(height);
                codec.input_components = channels;
                codec.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
                jpeg_set_defaults(&codec);
                if (channels == 3) {
                    // 4:4:4, as the GPU encoder writes it.
                    codec.comp_info[0].h_samp_factor = 1;
                    codec.comp_info[0].v_samp_factor = 1;
                }
                jpeg_set_quality(&codec, std::clamp(quality, 1, 100), TRUE);
                jpeg_start_compress(&codec, TRUE);
                const size_t row_bytes = static_cast<size_t>(width) * static_cast<size_t>(channels);
                while (codec.next_scanline < codec.image_height) {
                    JSAMPROW row = const_cast<JSAMPROW>(pixels + codec.next_scanline * row_bytes);
                    jpeg_write_scanlines(&codec, &row, 1);
                }
                jpeg_finish_compress(&codec);
                return true;
            }
        };

        std::vector<uint8_t> encode_jpeg_on_host(const uint8_t* const pixels, const int width,
                                                 const int height, const int channels, const int quality) {
            auto encoder = std::make_unique<JpegMemoryEncoder>();
            if (!encoder->encode(pixels, width, height, channels, quality)) {
                throw std::runtime_error(std::string("JPEG encoding failed: ") + encoder->error.message);
            }
            return std::move(encoder->destination.bytes);
        }

        std::vector<uint8_t> encode_device_jpeg(const lfs::core::Tensor& device_u8, const int width,
                                                const int height, const int channels, const int quality,
                                                void* const cuda_stream) {
            std::vector<uint8_t> host(static_cast<size_t>(width) * height * channels);
            check_hip(cudaMemcpyAsync(host.data(), device_u8.data_ptr(), host.size(),
                                      cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(cuda_stream)),
                      "JPEG encode readback");
            check_hip(cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream)), "JPEG encode readback sync");
            return encode_jpeg_on_host(host.data(), width, height, channels, quality);
        }

        // Interleaved samples; 8-bit codestreams are widened into the same container.
        struct HostJpeg2k {
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t components = 0;
            uint32_t bit_depth = 0;
            std::vector<uint16_t> samples;
        };

        bool is_jpeg2k(const uint8_t* const data, const size_t size) {
            return size >= 2 && data[0] == 0xFF && data[1] == 0x4F;
        }

        // Lossless HTJ2K with the parameters the nvJPEG2000 encoder is given:
        // reversible 5/3 transform, RPCL order, 64x64 code blocks, 6 resolutions.
        std::vector<uint8_t> encode_jpeg2k_on_host(const uint16_t* const samples, const uint32_t width,
                                                   const uint32_t height, const uint32_t components,
                                                   const uint32_t bit_depth, const bool high_throughput) {
            if (!high_throughput) {
                throw std::runtime_error("only HTJ2K block coding is available for JPEG 2000 encoding");
            }
            ojph::codestream codestream;
            ojph::param_siz siz = codestream.access_siz();
            siz.set_image_extent(ojph::point(width, height));
            siz.set_num_components(components);
            for (uint32_t c = 0; c < components; ++c)
                siz.set_component(c, ojph::point(1, 1), bit_depth, false);
            siz.set_image_offset(ojph::point(0, 0));
            siz.set_tile_size(ojph::size(width, height));
            uint32_t decompositions = 5;
            while (decompositions > 0 && (1u << decompositions) > std::min(width, height))
                --decompositions;
            ojph::param_cod cod = codestream.access_cod();
            cod.set_num_decomposition(decompositions);
            cod.set_block_dims(64, 64);
            cod.set_progression_order("RPCL");
            cod.set_color_transform(components == 3);
            cod.set_reversible(true);
            codestream.set_planar(false);

            ojph::mem_outfile file;
            file.open();
            codestream.write_headers(&file);
            ojph::ui32 next_component = 0;
            ojph::line_buf* line = codestream.exchange(nullptr, next_component);
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t c = 0; c < components; ++c) {
                    const uint16_t* const row = samples + static_cast<size_t>(y) * width * components + next_component;
                    if (line->flags & ojph::line_buf::LFT_32BIT) {
                        for (uint32_t x = 0; x < width; ++x)
                            line->i32[x] = row[static_cast<size_t>(x) * components];
                    } else {
                        for (uint32_t x = 0; x < width; ++x)
                            line->i64[x] = row[static_cast<size_t>(x) * components];
                    }
                    line = codestream.exchange(line, next_component);
                }
            }
            codestream.flush();
            const auto* const data = file.get_data();
            std::vector<uint8_t> encoded(data, data + file.tell());
            codestream.close();
            return encoded;
        }

        HostJpeg2k decode_jpeg2k_on_host(const uint8_t* const data, const size_t size) {
            if (!is_jpeg2k(data, size)) {
                throw std::runtime_error("expected a JPEG 2000 codestream");
            }
            ojph::mem_infile file;
            file.open(data, size);
            ojph::codestream codestream;
            codestream.read_headers(&file);
            const ojph::param_siz siz = codestream.access_siz();
            HostJpeg2k image;
            image.components = siz.get_num_components();
            image.width = siz.get_recon_width(0);
            image.height = siz.get_recon_height(0);
            image.bit_depth = siz.get_bit_depth(0);
            for (uint32_t c = 0; c < image.components; ++c) {
                if (siz.get_recon_width(c) != image.width || siz.get_recon_height(c) != image.height ||
                    siz.get_bit_depth(c) != image.bit_depth || siz.is_signed(c)) {
                    codestream.close();
                    throw std::runtime_error("JPEG 2000 components differ in size, depth or sign");
                }
            }
            if (image.bit_depth > 16) {
                codestream.close();
                throw std::runtime_error("JPEG 2000 samples wider than 16 bits are not supported");
            }
            codestream.set_planar(false);
            codestream.create();
            image.samples.resize(static_cast<size_t>(image.width) * image.height * image.components);
            for (uint32_t y = 0; y < image.height; ++y) {
                for (uint32_t c = 0; c < image.components; ++c) {
                    ojph::ui32 component = 0;
                    const ojph::line_buf* const line = codestream.pull(component);
                    uint16_t* const row =
                        image.samples.data() + static_cast<size_t>(y) * image.width * image.components + component;
                    for (uint32_t x = 0; x < image.width; ++x) {
                        const int64_t value = (line->flags & ojph::line_buf::LFT_32BIT) ? line->i32[x] : line->i64[x];
                        row[static_cast<size_t>(x) * image.components] =
                            static_cast<uint16_t>(std::clamp<int64_t>(value, 0, (int64_t{1} << image.bit_depth) - 1));
                    }
                }
            }
            codestream.close();
            return image;
        }

        // Copies host samples to a device tensor, as uint8 or in a Float16 container of uint16.
        lfs::core::Tensor upload_samples(const HostJpeg2k& image, const bool as_uint8, const cudaStream_t stream) {
            using namespace lfs::core;
            const TensorShape shape = image.components == 1
                                          ? TensorShape({size_t{image.height}, size_t{image.width}})
                                          : TensorShape({size_t{image.height}, size_t{image.width}, size_t{image.components}});
            Tensor tensor = Tensor::empty(shape, Device::CUDA, as_uint8 ? DataType::UInt8 : DataType::Float16);
            if (stream) {
                tensor.set_stream(stream);
            }
            if (as_uint8) {
                std::vector<uint8_t> bytes(image.samples.size());
                const bool narrow16 = image.bit_depth > 8;
                for (size_t i = 0; i < bytes.size(); ++i)
                    bytes[i] = narrow16 ? static_cast<uint8_t>((image.samples[i] * 255u + 32767u) / 65535u)
                                        : static_cast<uint8_t>(image.samples[i]);
                check_hip(cudaMemcpyAsync(tensor.data_ptr(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream),
                          "JPEG 2000 upload");
                check_hip(cudaStreamSynchronize(stream), "JPEG 2000 upload sync");
            } else {
                check_hip(cudaMemcpyAsync(tensor.data_ptr(), image.samples.data(), image.samples.size() * sizeof(uint16_t),
                                          cudaMemcpyHostToDevice, stream),
                          "JPEG 2000 upload");
                check_hip(cudaStreamSynchronize(stream), "JPEG 2000 upload sync");
            }
            return tensor;
        }

        std::vector<uint16_t> download_u16(const lfs::core::Tensor& u16_container, const cudaStream_t stream) {
            std::vector<uint16_t> samples(u16_container.numel());
            check_hip(cudaMemcpyAsync(samples.data(), u16_container.data_ptr(), samples.size() * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost, stream),
                      "JPEG 2000 readback");
            check_hip(cudaStreamSynchronize(stream), "JPEG 2000 readback sync");
            return samples;
        }

        bool is_jpeg(const uint8_t* const data, const size_t size) {
            return size >= 2 && data[0] == 0xFF && data[1] == 0xD8;
        }

        // Device layouts of a decoded JPEG: planar RGB [3,H,W], interleaved RGB [H,W,3], luma [H,W].
        enum class JpegLayout { PlanarRgb, InterleavedRgb, Luma };

        struct JpegDecompressor {
            jpeg_decompress_struct codec{};
            JpegError error{};
            bool created = false;
            ~JpegDecompressor() {
                if (created)
                    jpeg_destroy_decompress(&codec);
            }

            // The jump stays inside a frame with no destructors to skip.
            bool decode(const uint8_t* const data, const size_t size, const bool luma, std::vector<uint8_t>& pixels,
                        uint32_t& width, uint32_t& height) {
                codec.err = jpeg_std_error(&error.base);
                error.base.error_exit = jpeg_error_exit;
                if (setjmp(error.jump)) {
                    return false;
                }
                jpeg_create_decompress(&codec);
                created = true;
                jpeg_mem_src(&codec, data, static_cast<unsigned long>(size));
                jpeg_read_header(&codec, TRUE);
                codec.out_color_space = luma ? JCS_GRAYSCALE : JCS_RGB;
                jpeg_start_decompress(&codec);
                width = codec.output_width;
                height = codec.output_height;
                const size_t row_bytes = static_cast<size_t>(width) * codec.output_components;
                pixels.resize(row_bytes * height);
                while (codec.output_scanline < codec.output_height) {
                    JSAMPROW row = pixels.data() + codec.output_scanline * row_bytes;
                    jpeg_read_scanlines(&codec, &row, 1);
                }
                jpeg_finish_decompress(&codec);
                return true;
            }
        };

        // CPU decode for streams rocJPEG does not support, as nvImageCodec falls back to
        // its CPU decoder for streams nvJPEG does not support.
        std::vector<uint8_t> decode_jpeg_on_host(const uint8_t* const data, const size_t size, const JpegLayout layout,
                                                 uint32_t& width, uint32_t& height) {
            std::vector<uint8_t> pixels;
            auto decompressor = std::make_unique<JpegDecompressor>();
            if (!decompressor->decode(data, size, layout == JpegLayout::Luma, pixels, width, height)) {
                throw std::runtime_error(std::string("JPEG decoding failed: ") + decompressor->error.message);
            }
            if (layout != JpegLayout::PlanarRgb) {
                return pixels;
            }
            const size_t plane = static_cast<size_t>(width) * height;
            std::vector<uint8_t> planar(pixels.size());
            for (size_t i = 0; i < plane; ++i)
                for (size_t c = 0; c < 3; ++c)
                    planar[c * plane + i] = pixels[i * 3 + c];
            return planar;
        }

        std::atomic<uint64_t> cpu_decoded_jpegs{0};

        void log_cpu_decode(const std::string& reason, const size_t bytes) {
            const uint64_t n = cpu_decoded_jpegs.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 3) {
                LOG_WARN("[NvCodecImageLoader] Decoding a JPEG on the CPU (libjpeg): {} ({} bytes)", reason, bytes);
            } else if (n % 500 == 0) {
                LOG_WARN("[NvCodecImageLoader] {} JPEGs decoded on the CPU (libjpeg) so far", n);
            }
        }

#if LFS_HAVE_ROCJPEG
        std::string rocjpeg_error(const char* const what, const RocJpegStatus status) {
            return std::string(what) + " failed: " + rocJpegGetErrorName(status);
        }

        // rocJPEG declined a stream it does not support, as opposed to failing on it.
        struct RocJpegUnsupported : std::runtime_error {
            using std::runtime_error::runtime_error;
        };

        void check_rocjpeg(const RocJpegStatus status, const char* const what) {
            if (status == ROCJPEG_STATUS_JPEG_NOT_SUPPORTED) {
                throw RocJpegUnsupported(rocjpeg_error(what, status));
            }
            if (status != ROCJPEG_STATUS_SUCCESS) {
                throw std::runtime_error(rocjpeg_error(what, status));
            }
        }

        struct JpegFrame {
            uint8_t marker = 0;
            uint8_t precision = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t components = 0;
            uint8_t id[4] = {};
            uint8_t h[4] = {};
            uint8_t v[4] = {};
            uint8_t quant_table[4] = {};
        };

        struct JpegInspection {
            std::optional<JpegFrame> frame;
            std::string defect; // empty when the stream is structurally complete
        };

        // Walks the marker segments of a JPEG stream: the frame header, defined
        // quantization tables, a well-formed first scan header, scan data, and an
        // end-of-image marker. rocJPEG parses a stream with no scan at all, submits
        // it to the hardware decoder, and the decoder hangs until the driver resets
        // it. The other checks reject corrupt streams as a precaution.
        JpegInspection inspect_jpeg(const uint8_t* const data, const size_t size) {
            JpegInspection result;
            const auto defect = [&](std::string what) {
                result.defect = std::move(what);
                return result;
            };
            bool quant_defined[4] = {};
            size_t i = 2;
            while (true) {
                if (i + 2 > size) {
                    return defect("the data ends before the first scan");
                }
                if (data[i] != 0xFF) {
                    return defect("no marker at byte " + std::to_string(i));
                }
                const uint8_t marker = data[i + 1];
                if (marker == 0xFF) {
                    ++i;
                    continue;
                }
                i += 2;
                if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
                    continue;
                }
                if (marker == 0xD8 || marker == 0xD9) {
                    return defect("the data ends before the first scan");
                }
                if (i + 2 > size) {
                    return defect("a marker segment runs past the end of the data");
                }
                const size_t length = (size_t{data[i]} << 8) | data[i + 1];
                if (length < 2 || i + length > size) {
                    return defect("a marker segment runs past the end of the data");
                }
                const uint8_t* const segment = data + i + 2;
                const size_t n = length - 2;
                if (marker == 0xDB) {
                    for (size_t j = 0; j < n;) {
                        const uint8_t table = segment[j] & 0x0F;
                        if (table > 3) {
                            return defect("a quantization table number is out of range");
                        }
                        quant_defined[table] = true;
                        j += 1 + ((segment[j] >> 4) ? 128 : 64);
                        if (j > n) {
                            return defect("a quantization table is truncated");
                        }
                    }
                } else if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 &&
                           marker != 0xCC) {
                    if (result.frame) {
                        return defect("more than one frame header");
                    }
                    if (n < 6 || n < 6 + 3 * size_t{segment[5]}) {
                        return defect("the frame header is truncated");
                    }
                    JpegFrame frame;
                    frame.marker = marker;
                    frame.precision = segment[0];
                    frame.height = (uint32_t{segment[1]} << 8) | segment[2];
                    frame.width = (uint32_t{segment[3]} << 8) | segment[4];
                    frame.components = segment[5];
                    for (uint32_t c = 0; c < std::min<uint32_t>(frame.components, 4); ++c) {
                        frame.id[c] = segment[6 + 3 * c];
                        frame.h[c] = segment[7 + 3 * c] >> 4;
                        frame.v[c] = segment[7 + 3 * c] & 0x0F;
                        frame.quant_table[c] = segment[8 + 3 * c];
                    }
                    result.frame = frame;
                } else if (marker == 0xDA) {
                    if (!result.frame) {
                        return defect("a scan precedes the frame header");
                    }
                    const JpegFrame& frame = *result.frame;
                    if (n < 1 || segment[0] < 1 || segment[0] > 4 || n != 4 + 2 * size_t{segment[0]}) {
                        return defect("the scan header is malformed");
                    }
                    for (size_t k = 0; k < segment[0]; ++k) {
                        const uint8_t* const id_end = frame.id + std::min<uint32_t>(frame.components, 4);
                        if (std::find(frame.id, id_end, segment[1 + 2 * k]) == id_end) {
                            return defect("the scan names a component the frame does not have");
                        }
                    }
                    for (uint32_t c = 0; c < std::min<uint32_t>(frame.components, 4); ++c) {
                        if (frame.quant_table[c] > 3 || !quant_defined[frame.quant_table[c]]) {
                            return defect("a component uses an undefined quantization table");
                        }
                    }
                    const size_t scan_start = i + length;
                    for (size_t k = scan_start; k + 1 < size; ++k) {
                        if (data[k] == 0xFF && data[k + 1] == 0xD9) {
                            if (k == scan_start) {
                                return defect("the scan has no data");
                            }
                            return result;
                        }
                    }
                    return defect("the data is truncated (no end-of-image marker)");
                }
                i += length;
            }
        }

        // Why rocJPEG cannot decode a stream with this frame header, or nullopt when it can:
        // it decodes 8-bit baseline JPEG, grayscale or YCbCr at 4:4:4, 4:4:0, 4:2:2 or
        // 4:2:0, at least 64x64. It rejects every other frame type at parse, including
        // extended sequential (SOF1).
        std::optional<std::string> rocjpeg_unsupported_reason(const JpegFrame& frame) {
            if (frame.marker != 0xC0) {
                const char* const process = frame.marker == 0xC1   ? "extended sequential"
                                            : frame.marker == 0xC2 ? "progressive"
                                            : frame.marker == 0xC3 ? "lossless"
                                            : frame.marker <= 0xC7 ? "hierarchical"
                                                                   : "arithmetic-coded";
                return std::string(process) + " JPEG";
            }
            if (frame.precision != 8) {
                return std::to_string(frame.precision) + "-bit JPEG";
            }
            if (frame.components != 1 && frame.components != 3) {
                return std::to_string(frame.components) + "-component JPEG";
            }
            if (frame.width < 64 || frame.height < 64) {
                return "JPEG smaller than 64x64 (" + std::to_string(frame.width) + "x" +
                       std::to_string(frame.height) + ")";
            }
            if (frame.components == 3) {
                const bool chroma_match = frame.h[1] == frame.h[2] && frame.v[1] == frame.v[2] &&
                                          frame.h[1] != 0 && frame.v[1] != 0;
                const bool supported = chroma_match && frame.h[0] % frame.h[1] == 0 &&
                                       frame.v[0] % frame.v[1] == 0 && frame.h[0] / frame.h[1] <= 2 &&
                                       frame.v[0] / frame.v[1] <= 2;
                if (!supported) {
                    return "JPEG with chroma sampling " + std::to_string(frame.h[0]) + "x" +
                           std::to_string(frame.v[0]) + "," + std::to_string(frame.h[1]) + "x" +
                           std::to_string(frame.v[1]) + "," + std::to_string(frame.h[2]) + "x" +
                           std::to_string(frame.v[2]);
                }
            }
            return std::nullopt;
        }

        // Decodes one baseline JPEG. rocJPEG writes from its own stream and returns once the
        // image is complete; it writes into a private buffer, and the copy into the caller's
        // tensor is ordered on the caller's stream.
        struct RocJpegDecoder {
            RocJpegHandle handle = nullptr;
            RocJpegStreamHandle stream = nullptr;
            void* staging = nullptr;
            size_t staging_bytes = 0;

            RocJpegDecoder(const RocJpegDecoder&) = delete;
            RocJpegDecoder& operator=(const RocJpegDecoder&) = delete;

            explicit RocJpegDecoder(const int device_id) {
                if (const auto status = rocJpegCreate(ROCJPEG_BACKEND_HARDWARE, device_id, &handle);
                    status != ROCJPEG_STATUS_SUCCESS) {
                    handle = nullptr;
                    throw std::runtime_error(rocjpeg_error("rocJpegCreate", status));
                }
                if (const auto status = rocJpegStreamCreate(&stream); status != ROCJPEG_STATUS_SUCCESS) {
                    stream = nullptr;
                    rocJpegDestroy(handle);
                    handle = nullptr;
                    throw std::runtime_error(rocjpeg_error("rocJpegStreamCreate", status));
                }
            }

            ~RocJpegDecoder() {
                if (staging)
                    cudaFree(staging);
                if (stream)
                    rocJpegStreamDestroy(stream);
                if (handle)
                    rocJpegDestroy(handle);
            }

            void parse(const uint8_t* const data, const size_t size, uint32_t& width, uint32_t& height) {
                check_rocjpeg(rocJpegStreamParse(data, size, stream), "rocJpegStreamParse");
                uint8_t num_components = 0;
                RocJpegChromaSubsampling subsampling{};
                uint32_t widths[ROCJPEG_MAX_COMPONENT] = {};
                uint32_t heights[ROCJPEG_MAX_COMPONENT] = {};
                check_rocjpeg(rocJpegGetImageInfo(handle, stream, &num_components, &subsampling, widths, heights),
                              "rocJpegGetImageInfo");
                if (widths[0] == 0 || heights[0] == 0) {
                    throw std::runtime_error("rocJPEG reported an empty image");
                }
                width = widths[0];
                height = heights[0];
            }

            // Decodes the parsed image: planar RGB [3,H,W], interleaved RGB [H,W,3] or luma [H,W].
            // rocJPEG's output conversion writes each row out to a multiple of 8 pixels, past
            // a tight pitch and into the next row, so it decodes into rows padded to 16 pixels
            // (and 16 spare rows) and only the visible region is copied out.
            void decode_to(void* const destination, const uint32_t width, const uint32_t height,
                           const RocJpegOutputFormat format, const cudaStream_t cuda_stream) {
                const size_t planes = format == ROCJPEG_OUTPUT_RGB_PLANAR ? 3 : 1;
                const size_t pixel_bytes = format == ROCJPEG_OUTPUT_RGB ? 3 : 1;
                const size_t row_bytes = static_cast<size_t>(width) * pixel_bytes;
                const size_t pitch = (static_cast<size_t>(width) + 15) / 16 * 16 * pixel_bytes;
                const size_t plane_bytes = pitch * ((static_cast<size_t>(height) + 15) / 16 * 16 + 16);
                const size_t bytes = plane_bytes * planes;
                if (bytes > staging_bytes) {
                    if (staging)
                        check_hip(cudaFree(staging), "rocJPEG staging free");
                    staging = nullptr;
                    staging_bytes = 0;
                    check_hip(cudaMalloc(&staging, bytes), "rocJPEG staging allocation");
                    staging_bytes = bytes;
                }
                RocJpegImage image{};
                auto* const base = static_cast<uint8_t*>(staging);
                for (size_t c = 0; c < planes; ++c) {
                    image.channel[c] = base + c * plane_bytes;
                    image.pitch[c] = static_cast<uint32_t>(pitch);
                }
                RocJpegDecodeParams params{};
                params.output_format = format;
                check_rocjpeg(rocJpegDecode(handle, stream, &params, &image), "rocJpegDecode");
                for (size_t c = 0; c < planes; ++c) {
                    check_hip(cudaMemcpy2DAsync(static_cast<uint8_t*>(destination) + c * row_bytes * height, row_bytes,
                                                base + c * plane_bytes, pitch, row_bytes, height,
                                                cudaMemcpyDeviceToDevice, cuda_stream),
                              "rocJPEG output copy");
                }
                // The next decode reuses the staging buffer.
                check_hip(cudaStreamSynchronize(cuda_stream), "rocJPEG output copy sync");
            }
        };

        RocJpegOutputFormat rocjpeg_format(const JpegLayout layout) {
            switch (layout) {
            case JpegLayout::PlanarRgb: return ROCJPEG_OUTPUT_RGB_PLANAR;
            case JpegLayout::InterleavedRgb: return ROCJPEG_OUTPUT_RGB;
            case JpegLayout::Luma: return ROCJPEG_OUTPUT_Y;
            }
            return ROCJPEG_OUTPUT_RGB;
        }

        std::vector<uint8_t> make_probe_jpeg() {
            constexpr int size = 64;
            std::vector<uint8_t> pixels(size * size * 3);
            for (size_t i = 0; i < pixels.size(); ++i)
                pixels[i] = static_cast<uint8_t>(i * 7);
            return encode_jpeg_on_host(pixels.data(), size, size, 3, 90);
        }

        bool rocjpeg_decodes_probe() {
            try {
                int device = 0;
                check_hip(cudaGetDevice(&device), "cudaGetDevice");
                RocJpegDecoder decoder(device);
                const auto probe = make_probe_jpeg();
                uint32_t width = 0;
                uint32_t height = 0;
                decoder.parse(probe.data(), probe.size(), width, height);
                void* output = nullptr;
                check_hip(cudaMalloc(&output, static_cast<size_t>(width) * height * 3), "probe allocation");
                std::unique_ptr<void, decltype(&cudaFree)> owner(output, &cudaFree);
                decoder.decode_to(output, width, height, ROCJPEG_OUTPUT_RGB, nullptr);
                return true;
            } catch (const std::exception& e) {
                LOG_DEBUG("[NvCodecImageLoader] rocJPEG probe failed: {}", e.what());
                return false;
            }
        }
#endif
    } // namespace

    struct NvCodecImageLoader::Impl {
        int device_id = 0;
#if LFS_HAVE_ROCJPEG
        std::mutex pool_mutex;
        std::condition_variable pool_cv;
        std::vector<std::unique_ptr<RocJpegDecoder>> idle;
        size_t created = 0;
        size_t pool_size = DECODER_POOL_SIZE;

        std::unique_ptr<RocJpegDecoder> acquire() {
            std::unique_lock lock(pool_mutex);
            pool_cv.wait(lock, [&] { return !idle.empty() || created < pool_size; });
            if (!idle.empty()) {
                auto decoder = std::move(idle.back());
                idle.pop_back();
                return decoder;
            }
            ++created;
            lock.unlock();
            try {
                check_hip(cudaSetDevice(device_id), "cudaSetDevice");
                return std::make_unique<RocJpegDecoder>(device_id);
            } catch (...) {
                std::lock_guard relock(pool_mutex);
                --created;
                pool_cv.notify_one();
                throw;
            }
        }

        void release(std::unique_ptr<RocJpegDecoder> decoder) {
            {
                std::lock_guard lock(pool_mutex);
                idle.push_back(std::move(decoder));
            }
            pool_cv.notify_one();
        }

        struct Lease {
            Impl* impl;
            std::unique_ptr<RocJpegDecoder> decoder;
            ~Lease() {
                if (decoder)
                    impl->release(std::move(decoder));
            }
        };
#endif
        bool fallback_enabled = true;

        // Decodes a JPEG into the tensor allocate(height, width) returns, with rocJPEG, or on
        // the CPU when rocJPEG does not support the stream and fallback is enabled. Any other
        // failure is an error.
        template <typename Allocate>
        lfs::core::Tensor decode_jpeg(const uint8_t* const data, const size_t size, const JpegLayout layout,
                                      const cudaStream_t stream, Allocate&& allocate) {
#if LFS_HAVE_ROCJPEG
            const JpegInspection inspection = inspect_jpeg(data, size);
            std::optional<std::string> unsupported =
                inspection.frame ? rocjpeg_unsupported_reason(*inspection.frame) : std::nullopt;
            if (!unsupported) {
                if (!inspection.defect.empty()) {
                    LOG_ERROR("[NvCodecImageLoader] Decode failed: corrupt JPEG, {} ({} bytes)", inspection.defect,
                              size);
                    throw std::runtime_error("Corrupt JPEG: " + inspection.defect);
                }
                try {
                    Lease lease{this, acquire()};
                    uint32_t width = 0;
                    uint32_t height = 0;
                    lease.decoder->parse(data, size, width, height);
                    lfs::core::Tensor decoded = allocate(height, width);
                    lease.decoder->decode_to(decoded.data_ptr(), width, height, rocjpeg_format(layout), stream);
                    return decoded;
                } catch (const RocJpegUnsupported& e) {
                    unsupported = e.what();
                } catch (const std::exception& e) {
                    LOG_ERROR("[NvCodecImageLoader] Decode failed: {} ({} bytes)", e.what(), size);
                    throw;
                }
            }
            if (!fallback_enabled)
                throw std::runtime_error("rocJPEG does not support this stream: " + *unsupported);
            log_cpu_decode("rocJPEG does not support it: " + *unsupported, size);
#else
            if (!fallback_enabled)
                throw std::runtime_error("rocJPEG is not available in this build");
            log_cpu_decode("rocJPEG is not available in this build", size);
#endif
            uint32_t width = 0;
            uint32_t height = 0;
            const auto pixels = decode_jpeg_on_host(data, size, layout, width, height);
            lfs::core::Tensor decoded = allocate(height, width);
            check_hip(cudaMemcpyAsync(decoded.data_ptr(), pixels.data(), pixels.size(), cudaMemcpyHostToDevice, stream),
                      "JPEG upload");
            check_hip(cudaStreamSynchronize(stream), "JPEG upload sync");
            return decoded;
        }
    };

    NvCodecImageLoader::NvCodecImageLoader(const Options& options)
        : impl_(std::make_unique<Impl>()) {
        impl_->device_id = options.device_id;
        impl_->fallback_enabled = options.enable_fallback;
#if LFS_HAVE_ROCJPEG
        impl_->pool_size = options.decoder_pool_size > 0 ? options.decoder_pool_size : DECODER_POOL_SIZE;
        if (options.create_eagerly) {
            std::vector<std::unique_ptr<RocJpegDecoder>> decoders;
            for (size_t i = 0; i < impl_->pool_size; ++i)
                decoders.push_back(impl_->acquire());
            for (auto& decoder : decoders)
                impl_->release(std::move(decoder));
        }
#endif
        live_loaders.fetch_add(1, std::memory_order_relaxed);
    }

    NvCodecImageLoader::~NvCodecImageLoader() {
        live_loaders.fetch_sub(1, std::memory_order_relaxed);
    }

    size_t NvCodecImageLoader::live_count() {
        return live_loaders.load(std::memory_order_relaxed);
    }

    bool NvCodecImageLoader::is_available() {
        static std::once_flag once;
        static bool available = false;
        std::call_once(once, [] {
#if LFS_HAVE_ROCJPEG
            available = rocjpeg_decodes_probe();
#endif
            if (available) {
                LOG_INFO("JPEG decoding runs on the GPU (rocJPEG). JPEGs rocJPEG does not support (extended "
                         "sequential, progressive, arithmetic-coded, 12-bit, CMYK, 4:1:1, or smaller than 64x64) decode on the CPU "
                         "(libjpeg) with a warning");
            } else {
                LOG_WARN("JPEG decoding runs on the CPU: rocJPEG cannot decode JPEG on this system");
            }
        });
        return available;
    }

    std::string_view NvCodecImageLoader::jpeg_decoder() {
        if (!is_available()) {
            return "CPU";
        }
        return cpu_decoded_jpegs.load(std::memory_order_relaxed) == 0
                   ? "rocJPEG (GPU)"
                   : "rocJPEG (GPU), CPU (libjpeg) for JPEGs rocJPEG does not support";
    }

    std::vector<uint8_t> NvCodecImageLoader::read_file(const std::filesystem::path& path) {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file)) {
            throw std::runtime_error("Failed to open file: " + lfs::core::path_to_utf8(path));
        }
        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file: " + lfs::core::path_to_utf8(path));
        }
        return buffer;
    }

    lfs::core::Tensor NvCodecImageLoader::load_image_gpu(
        const std::filesystem::path& path,
        int resize_factor,
        int max_width,
        void* cuda_stream,
        DecodeFormat format,
        bool output_uint8, const MaskedImageFill* masked_fill) {
        LOG_DEBUG("NvCodecImageLoader: Loading {}", lfs::core::path_to_utf8(path));
        const auto file_data = read_file(path);
        return load_image_from_memory_gpu(file_data, resize_factor, max_width, cuda_stream, format,
                                          output_uint8, masked_fill);
    }

    lfs::core::Tensor NvCodecImageLoader::load_image_from_memory_gpu(
        const std::vector<uint8_t>& jpeg_data,
        int resize_factor,
        int max_width,
        void* cuda_stream,
        DecodeFormat format,
        bool output_uint8, const MaskedImageFill* masked_fill) {
        using namespace lfs::core;
        const bool is_grayscale = (format != DecodeFormat::RGB);
        const int num_channels = is_grayscale ? 1 : 3;
        const auto stream = static_cast<cudaStream_t>(cuda_stream);

        // Decode into the layouts nvImageCodec produces: [H,W] uint8 for grayscale, [H,W,3]
        // uint8 for RGB, or [H,W,3] uint16 in a Float16 container for 16-bit RGB JPEG 2000.
        Tensor image_tensor_aux;
        bool decode_u16 = false;
        if (is_jpeg2k(jpeg_data.data(), jpeg_data.size())) {
            HostJpeg2k image = decode_jpeg2k_on_host(jpeg_data.data(), jpeg_data.size());
            if (image.components != 1 && image.components != 3) {
                throw std::runtime_error("Expected grayscale or RGB JPEG2000 input, got " +
                                         std::to_string(image.components) + " components");
            }
            if (static_cast<int>(image.components) != num_channels) {
                HostJpeg2k converted = image;
                converted.components = static_cast<uint32_t>(num_channels);
                converted.samples.assign(static_cast<size_t>(image.width) * image.height * num_channels, 0);
                const size_t pixels = static_cast<size_t>(image.width) * image.height;
                for (size_t i = 0; i < pixels; ++i) {
                    if (is_grayscale) {
                        const uint16_t* const rgb = &image.samples[i * 3];
                        converted.samples[i] = static_cast<uint16_t>(
                            std::lround(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]));
                    } else {
                        for (int c = 0; c < 3; ++c)
                            converted.samples[i * 3 + c] = image.samples[i];
                    }
                }
                image = std::move(converted);
            }
            // Grayscale consumers (masks, depth) stay 8-bit, as nvImageCodec converts them.
            decode_u16 = !is_grayscale && image.bit_depth > 8;
            check_hip(cudaSetDevice(impl_->device_id), "cudaSetDevice");
            image_tensor_aux = upload_samples(image, !decode_u16, stream);
        } else if (is_jpeg(jpeg_data.data(), jpeg_data.size())) {
            check_hip(cudaSetDevice(impl_->device_id), "cudaSetDevice");
            image_tensor_aux = impl_->decode_jpeg(
                jpeg_data.data(), jpeg_data.size(), is_grayscale ? JpegLayout::Luma : JpegLayout::InterleavedRgb, stream,
                [&](const uint32_t height, const uint32_t width) {
                    return is_grayscale
                               ? Tensor::empty(TensorShape({height, width}), Device::CUDA, DataType::UInt8)
                               : Tensor::empty(TensorShape({height, width, size_t{3}}), Device::CUDA, DataType::UInt8);
                });
        } else {
            throw std::runtime_error("GPU decode supports JPEG and JPEG 2000 only");
        }

        const int src_height = static_cast<int>(image_tensor_aux.shape()[0]);
        const int src_width = static_cast<int>(image_tensor_aux.shape()[1]);
        int target_width = src_width;
        int target_height = src_height;
        if (resize_factor > 1) {
            target_width /= resize_factor;
            target_height /= resize_factor;
        }
        if (max_width > 0 && (target_width > max_width || target_height > max_width)) {
            if (target_width > target_height) {
                target_height = std::max(1, max_width * target_height / target_width);
                target_width = max_width;
            } else {
                target_width = std::max(1, max_width * target_width / target_height);
                target_height = max_width;
            }
        }
        const bool needs_resize = (target_width != src_width || target_height != src_height);

        if (masked_fill)
            fill_masked_image(image_tensor_aux, *masked_fill, stream);

        // Lanczos reads uint8 or float32; widen uint16 to normalized float first.
        std::optional<Tensor> u16_as_float;
        if (decode_u16 && needs_resize) {
            u16_as_float.emplace(Tensor::empty(
                TensorShape({static_cast<size_t>(src_height), static_cast<size_t>(src_width),
                             static_cast<size_t>(num_channels)}),
                Device::CUDA,
                DataType::Float32));
            cuda::launch_uint16_hwc_to_float32_hwc(
                reinterpret_cast<const uint16_t*>(image_tensor_aux.data_ptr()),
                u16_as_float->ptr<float>(),
                src_height, src_width, num_channels, stream);
        }
        const Tensor& resize_input_image = u16_as_float ? *u16_as_float : image_tensor_aux;

        Tensor output_tensor;
        if (needs_resize) {
            if (is_grayscale) {
                output_tensor = lanczos_resize_grayscale(resize_input_image, target_height, target_width,
                                                         LANCZOS_KERNEL_SIZE, stream, format == DecodeFormat::Mask);
            } else {
                output_tensor = lanczos_resize(resize_input_image, target_height, target_width,
                                               LANCZOS_KERNEL_SIZE, stream);
                if (output_uint8) {
                    auto output_uint8_tensor = Tensor::empty(output_tensor.shape(), Device::CUDA, DataType::UInt8);
                    cuda::launch_float32_chw_to_uint8_chw(
                        output_tensor.ptr<float>(), output_uint8_tensor.ptr<uint8_t>(),
                        output_tensor.shape()[1], output_tensor.shape()[2], output_tensor.shape()[0], stream);
                    output_tensor = std::move(output_uint8_tensor);
                }
            }
        } else if (is_grayscale) {
            const size_t H = image_tensor_aux.shape()[0], W = image_tensor_aux.shape()[1];
            if (format == DecodeFormat::Mask) {
                output_tensor = lfs::core::cuda::normalize_mask_bytes(image_tensor_aux).reshape(TensorShape({H, W}));
            } else {
                output_tensor = Tensor::empty(TensorShape({H, W}), Device::CUDA, DataType::Float32);
                cuda::launch_uint8_hw_to_float32_hw(
                    reinterpret_cast<const uint8_t*>(image_tensor_aux.data_ptr()),
                    reinterpret_cast<float*>(output_tensor.data_ptr()), H, W, stream);
            }
        } else {
            const size_t H = image_tensor_aux.shape()[0], W = image_tensor_aux.shape()[1], C = 3;
            if (output_uint8) {
                output_tensor = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::UInt8);
                if (decode_u16) {
                    cuda::launch_uint16_hwc_to_uint8_chw(
                        reinterpret_cast<const uint16_t*>(image_tensor_aux.data_ptr()),
                        reinterpret_cast<uint8_t*>(output_tensor.data_ptr()), H, W, C, stream);
                } else {
                    cuda::launch_uint8_hwc_to_uint8_chw(
                        reinterpret_cast<const uint8_t*>(image_tensor_aux.data_ptr()),
                        reinterpret_cast<uint8_t*>(output_tensor.data_ptr()), H, W, C, stream);
                }
            } else {
                output_tensor = Tensor::empty(TensorShape({C, H, W}), Device::CUDA, DataType::Float32);
                if (decode_u16) {
                    cuda::launch_uint16_hwc_to_float32_chw(
                        reinterpret_cast<const uint16_t*>(image_tensor_aux.data_ptr()),
                        reinterpret_cast<float*>(output_tensor.data_ptr()), H, W, C, stream);
                } else {
                    cuda::launch_uint8_hwc_to_float32_chw(
                        reinterpret_cast<const uint8_t*>(image_tensor_aux.data_ptr()),
                        reinterpret_cast<float*>(output_tensor.data_ptr()), H, W, C, stream);
                }
            }
        }

        // Materialize before handoff, on the caller's stream when it has one.
        check_hip(stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize(), "CUDA sync");
        return output_tensor;
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::decode_jpeg_batch_from_spans(
        const std::vector<std::pair<const uint8_t*, size_t>>& jpeg_spans,
        void* cuda_stream,
        const bool output_uint8,
        const bool synchronize,
        std::vector<lfs::core::Tensor*>* reusable_hwc,
        std::vector<lfs::core::Tensor*>* reusable_outputs) {
        using namespace lfs::core;
        if (jpeg_spans.empty()) {
            return {};
        }
        const auto stream = static_cast<cudaStream_t>(cuda_stream);
        const auto reuse = [](std::vector<Tensor*>* const pool, const size_t i, const TensorShape& shape,
                              const DataType dtype) -> std::optional<Tensor> {
            if (pool && i < pool->size() && (*pool)[i] && (*pool)[i]->is_valid() &&
                (*pool)[i]->shape() == shape && (*pool)[i]->device() == Device::CUDA &&
                (*pool)[i]->dtype() == dtype) {
                return *(*pool)[i];
            }
            return std::nullopt;
        };
        const auto publish = [](std::vector<Tensor*>* const pool, const size_t i, const Tensor& tensor) {
            if (pool && i < pool->size() && (*pool)[i]) {
                *(*pool)[i] = tensor;
            }
        };

        check_hip(cudaSetDevice(impl_->device_id), "cudaSetDevice");
        std::vector<Tensor> outputs;
        outputs.reserve(jpeg_spans.size());
        for (size_t i = 0; i < jpeg_spans.size(); ++i) {
            const auto [data, size] = jpeg_spans[i];
            if (!data || size == 0) {
                throw std::runtime_error("Empty JPEG batch input");
            }
            if (!is_jpeg(data, size)) {
                throw std::runtime_error("JPEG batch input is not an 8-bit JPEG");
            }
            // uint8 requests decode planar RGB straight into the CHW destination;
            // float requests decode interleaved RGB into a staging tensor and widen.
            auto* const decoded_pool = output_uint8 ? reusable_outputs : reusable_hwc;
            Tensor decoded = impl_->decode_jpeg(
                data, size, output_uint8 ? JpegLayout::PlanarRgb : JpegLayout::InterleavedRgb, stream,
                [&](const uint32_t height, const uint32_t width) {
                    const size_t H = height, W = width;
                    const TensorShape decoded_shape =
                        output_uint8 ? TensorShape({size_t{3}, H, W}) : TensorShape({H, W, size_t{3}});
                    Tensor tensor = reuse(decoded_pool, i, decoded_shape, DataType::UInt8)
                                        .value_or(Tensor::empty_exact(decoded_shape, DataType::UInt8));
                    if (stream) {
                        tensor.set_stream(stream);
                    }
                    return tensor;
                });
            publish(decoded_pool, i, decoded);
            const size_t H = output_uint8 ? decoded.shape()[1] : decoded.shape()[0];
            const size_t W = output_uint8 ? decoded.shape()[2] : decoded.shape()[1];
            if (output_uint8) {
                outputs.push_back(std::move(decoded));
                continue;
            }
            const TensorShape output_shape({size_t{3}, H, W});
            Tensor output = reuse(reusable_outputs, i, output_shape, DataType::Float32)
                                .value_or(Tensor::empty(output_shape, Device::CUDA, DataType::Float32));
            if (stream) {
                output.set_stream(stream);
            }
            cuda::launch_uint8_hwc_to_float32_chw(decoded.ptr<uint8_t>(), output.ptr<float>(), H, W, 3, stream);
            publish(reusable_outputs, i, output);
            outputs.push_back(std::move(output));
        }
        if (synchronize) {
            check_hip(stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize(), "CUDA sync");
        }
        return outputs;
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_to_jpeg(
        const lfs::core::Tensor& image,
        const int quality,
        void* cuda_stream) {
        using namespace lfs::core;
        const auto& shape = image.shape();
        if (shape.rank() != 3) {
            throw std::runtime_error("Expected 3D tensor, got " + std::to_string(shape.rank()) + "D");
        }
        const bool is_hwc = (shape[2] == 3);
        const bool is_chw = !is_hwc && (shape[0] == 3 && shape[1] > 3 && shape[2] > 3);
        const int height = static_cast<int>(is_chw ? shape[1] : shape[0]);
        const int width = static_cast<int>(is_chw ? shape[2] : shape[1]);

        Tensor hwc_uint8;
        if (is_chw) {
            auto permuted = image.permute({1, 2, 0}).contiguous();
            hwc_uint8 = (permuted.dtype() == DataType::Float32)
                            ? (permuted * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8)
                            : permuted.to(DataType::UInt8);
        } else {
            hwc_uint8 = (image.dtype() == DataType::Float32)
                            ? (image * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8)
                            : image.to(DataType::UInt8);
        }
        if (hwc_uint8.device() != Device::CUDA) {
            hwc_uint8 = hwc_uint8.to(Device::CUDA);
        }
        hwc_uint8 = hwc_uint8.contiguous();
        return encode_device_jpeg(hwc_uint8, width, height, 3, quality, cuda_stream);
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_to_jpeg2k(
        const lfs::core::Tensor& image,
        const RgbLayout layout,
        void* cuda_stream,
        bool high_throughput) {
        using namespace lfs::core;
        const auto& shape = image.shape();
        if (shape.rank() != 3) {
            throw std::runtime_error("Expected 3D tensor, got " + std::to_string(shape.rank()) + "D");
        }
        if (image.dtype() != DataType::Float32 || image.device() != Device::CUDA) {
            throw std::runtime_error("encode_to_jpeg2k expects a Float32 CUDA tensor");
        }
        const bool chw = layout == RgbLayout::CHW;
        if ((chw ? shape[0] : shape[2]) != 3) {
            throw std::runtime_error("encode_to_jpeg2k expects 3 RGB channels in the stated layout");
        }
        const uint32_t height = static_cast<uint32_t>(chw ? shape[1] : shape[0]);
        const uint32_t width = static_cast<uint32_t>(chw ? shape[2] : shape[1]);
        constexpr uint32_t channels = 3;
        const auto stream = static_cast<cudaStream_t>(cuda_stream);
        // Float16 is only a 2-byte container for the uint16 samples (no UInt16 dtype).
        const Tensor hwc_float = chw ? image.permute({1, 2, 0}).contiguous() : image.contiguous();
        Tensor hwc_uint16 = Tensor::empty(TensorShape({height, width, channels}), Device::CUDA, DataType::Float16);
        cuda::launch_float32_hwc_to_uint16_hwc(
            hwc_float.ptr<float>(), reinterpret_cast<uint16_t*>(hwc_uint16.data_ptr()),
            height, width, channels, stream);
        const auto samples = download_u16(hwc_uint16, stream);
        return encode_jpeg2k_on_host(samples.data(), width, height, channels, 16, high_throughput);
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_grayscale_to_jpeg2k(
        const lfs::core::Tensor& image,
        void* cuda_stream,
        bool high_throughput,
        bool eight_bit) {
        using namespace lfs::core;
        // The eight-bit staging expression uses Tensor pointwise operations, whose
        // execution stream is not the caller-supplied codec stream.
        LFS_ASSERT_MSG(
            !eight_bit || image.dtype() == DataType::UInt8 || cuda_stream == nullptr,
            "eight-bit JPEG2000 staging currently requires the legacy CUDA stream");
        const auto& shape = image.shape();
        if (shape.rank() != 2) {
            throw std::runtime_error("Expected 2D tensor, got " + std::to_string(shape.rank()) + "D");
        }
        if ((image.dtype() != DataType::Float32 && !(eight_bit && image.dtype() == DataType::UInt8)) ||
            image.device() != Device::CUDA) {
            throw std::runtime_error("encode_grayscale_to_jpeg2k expects Float32 CUDA, or UInt8 for an eight-bit mask");
        }
        const uint32_t height = static_cast<uint32_t>(shape[0]);
        const uint32_t width = static_cast<uint32_t>(shape[1]);
        const auto stream = static_cast<cudaStream_t>(cuda_stream);
        const Tensor hw_float = image.contiguous();
        std::vector<uint16_t> samples;
        if (eight_bit) {
            const Tensor bytes = (hw_float.dtype() == DataType::UInt8
                                      ? hw_float
                                      : (hw_float * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8))
                                     .contiguous();
            std::vector<uint8_t> host(bytes.numel());
            check_hip(cudaMemcpy(host.data(), bytes.data_ptr(), host.size(), cudaMemcpyDeviceToHost),
                      "JPEG 2000 readback");
            samples.assign(host.begin(), host.end());
        } else {
            Tensor u16 = Tensor::empty(TensorShape({height, width}), Device::CUDA, DataType::Float16);
            cuda::launch_float32_hwc_to_uint16_hwc(
                hw_float.ptr<float>(), reinterpret_cast<uint16_t*>(u16.data_ptr()), height, width, 1, stream);
            samples = download_u16(u16, stream);
        }
        return encode_jpeg2k_on_host(samples.data(), width, height, 1, eight_bit ? 8 : 16, high_throughput);
    }

    lfs::core::Tensor NvCodecImageLoader::decode_jpeg2k_16bit_from_memory_gpu(
        const std::vector<uint8_t>& jpeg2k_data,
        void* cuda_stream,
        const bool synchronize,
        const bool allow_uint8,
        const bool exact_mask) {
        using namespace lfs::core;
        const auto stream = static_cast<cudaStream_t>(cuda_stream);
        const HostJpeg2k image = decode_jpeg2k_on_host(jpeg2k_data.data(), jpeg2k_data.size());
        const bool decode_uint8 = image.bit_depth <= 8;
        if (decode_uint8 && !allow_uint8) {
            throw std::runtime_error("Expected UINT16 JPEG2000 input, got " + std::to_string(image.bit_depth) + "-bit samples");
        }
        if (image.components != 1 && image.components != 3) {
            throw std::runtime_error("Expected grayscale or RGB JPEG2000 input, got " +
                                     std::to_string(image.components) + " components");
        }
        check_hip(cudaSetDevice(impl_->device_id), "cudaSetDevice");
        Tensor decoded = upload_samples(image, decode_uint8, stream);
        if (decode_uint8) {
            return decoded;
        }
        const size_t height = image.height, width = image.width, components = image.components;
        Tensor output = Tensor::empty(decoded.shape(), Device::CUDA, DataType::Float32);
        if (stream) {
            output.set_stream(stream);
        }
        if (exact_mask) {
            lfs::core::cuda::normalize_mask_u16(
                reinterpret_cast<const uint16_t*>(decoded.data_ptr()), output.ptr<float>(),
                height * width * components, stream);
        } else {
            cuda::launch_uint16_hwc_to_float32_hwc(
                reinterpret_cast<const uint16_t*>(decoded.data_ptr()), output.ptr<float>(),
                height, width, components, stream);
        }
        if (synchronize) {
            check_hip(stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize(), "CUDA sync");
        }
        return output;
    }

    std::vector<lfs::core::Tensor> NvCodecImageLoader::decode_jpeg2k_16bit_batch_from_spans(
        const std::vector<std::pair<const uint8_t*, size_t>>& jpeg2k_spans,
        void* cuda_stream,
        const bool synchronize) {
        std::vector<lfs::core::Tensor> outputs;
        outputs.reserve(jpeg2k_spans.size());
        for (const auto& [data, size] : jpeg2k_spans) {
            if (!data || size == 0) {
                throw std::runtime_error("Empty JPEG2000 batch input");
            }
            outputs.push_back(decode_jpeg2k_16bit_from_memory_gpu(
                std::vector<uint8_t>(data, data + size), cuda_stream, false, false, false));
        }
        if (synchronize) {
            const auto stream = static_cast<cudaStream_t>(cuda_stream);
            check_hip(stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize(), "CUDA sync");
        }
        return outputs;
    }

    std::vector<uint8_t> NvCodecImageLoader::encode_grayscale_to_jpeg(
        const lfs::core::Tensor& image,
        const int quality,
        void* cuda_stream) {
        using namespace lfs::core;
        const auto& shape = image.shape();
        if (shape.rank() != 2) {
            throw std::runtime_error("Expected 2D tensor for grayscale, got " +
                                     std::to_string(shape.rank()) + "D");
        }
        const int height = static_cast<int>(shape[0]);
        const int width = static_cast<int>(shape[1]);
        Tensor hw_uint8 = (image.dtype() == DataType::Float32)
                              ? (image * 255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8)
                              : image.to(DataType::UInt8);
        if (hw_uint8.device() != Device::CUDA) {
            hw_uint8 = hw_uint8.to(Device::CUDA);
        }
        hw_uint8 = hw_uint8.contiguous();
        return encode_device_jpeg(hw_uint8, width, height, 1, quality, cuda_stream);
    }

    std::vector<std::vector<uint8_t>> NvCodecImageLoader::encode_batch_rgb_to_jpeg(
        const std::vector<void*>& gpu_ptrs,
        const int width,
        const int height,
        const int quality,
        void* cuda_stream) {
        const auto stream = static_cast<cudaStream_t>(cuda_stream);
        const size_t bytes = static_cast<size_t>(width) * height * 3;
        std::vector<uint8_t> host(bytes);
        std::vector<std::vector<uint8_t>> encoded(gpu_ptrs.size());
        // A frame that fails to encode comes back empty and the rest of the batch is kept.
        size_t failed_count = 0;
        for (size_t i = 0; i < gpu_ptrs.size(); ++i) {
            if (gpu_ptrs[i]) {
                check_hip(cudaMemcpyAsync(host.data(), gpu_ptrs[i], bytes, cudaMemcpyDeviceToHost, stream),
                          "JPEG encode readback");
                check_hip(cudaStreamSynchronize(stream), "JPEG encode readback sync");
            }
            try {
                if (!gpu_ptrs[i]) {
                    throw std::runtime_error("null frame pointer");
                }
                encoded[i] = encode_jpeg_on_host(host.data(), width, height, 3, quality);
            } catch (const std::exception& e) {
                LOG_ERROR("Batch encode frame {} failed: {}", i, e.what());
                failed_count++;
            }
        }
        if (failed_count > 0) {
            LOG_WARN("Batch encode: {} of {} frames failed", failed_count, gpu_ptrs.size());
        }
        return encoded;
    }

} // namespace lfs::io
