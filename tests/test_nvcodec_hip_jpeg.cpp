/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// JPEG paths of the image loader on AMD GPUs: rocJPEG decode, the CPU decode for
// streams rocJPEG does not support, and the libjpeg encoder.

#include "core/tensor.hpp"
#include "io/nvcodec_image_loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <jpeglib.h>

namespace {

    using lfs::io::NvCodecImageLoader;

    std::vector<uint8_t> test_pixels(const int width, const int height, const uint32_t seed) {
        std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 3);
        uint32_t state = seed;
        for (size_t i = 0; i < pixels.size(); ++i) {
            const size_t x = (i / 3) % width;
            const size_t y = (i / 3) / width;
            state = state * 1664525u + 1013904223u;
            pixels[i] = static_cast<uint8_t>((x * 3 + y * 5 + (i % 3) * 60) + (state >> 28));
        }
        return pixels;
    }

    struct EncodeOptions {
        bool progressive = false;
        int luma_h = 1;
        int luma_v = 1;
        bool extended_sequential = false; // baseline stream relabelled SOF1
    };

    std::vector<uint8_t> encode_jpeg(const std::vector<uint8_t>& pixels, const int width, const int height,
                                     const EncodeOptions& options = {}) {
        jpeg_compress_struct codec{};
        jpeg_error_mgr error{};
        codec.err = jpeg_std_error(&error);
        jpeg_create_compress(&codec);
        unsigned char* buffer = nullptr;
        unsigned long size = 0;
        jpeg_mem_dest(&codec, &buffer, &size);
        codec.image_width = static_cast<JDIMENSION>(width);
        codec.image_height = static_cast<JDIMENSION>(height);
        codec.input_components = 3;
        codec.in_color_space = JCS_RGB;
        jpeg_set_defaults(&codec);
        jpeg_set_quality(&codec, 95, TRUE);
        codec.comp_info[0].h_samp_factor = options.luma_h;
        codec.comp_info[0].v_samp_factor = options.luma_v;
        if (options.progressive) {
            jpeg_simple_progression(&codec);
        }
        jpeg_start_compress(&codec, TRUE);
        const size_t row_bytes = static_cast<size_t>(width) * 3;
        while (codec.next_scanline < codec.image_height) {
            JSAMPROW row = const_cast<JSAMPROW>(pixels.data() + codec.next_scanline * row_bytes);
            jpeg_write_scanlines(&codec, &row, 1);
        }
        jpeg_finish_compress(&codec);
        jpeg_destroy_compress(&codec);
        std::vector<uint8_t> encoded(buffer, buffer + size);
        std::free(buffer);
        if (options.extended_sequential) {
            // Huffman tables and 8-bit precision are the same in both processes, so only
            // the frame marker changes.
            for (size_t i = 2; i + 4 <= encoded.size();
                 i += 2 + ((size_t{encoded[i + 2]} << 8) | encoded[i + 3])) {
                if (encoded[i] == 0xFF && encoded[i + 1] == 0xC0) {
                    encoded[i + 1] = 0xC1;
                    break;
                }
            }
        }
        return encoded;
    }

    // Interleaved RGB as libjpeg decodes it, the reference for the CPU decode path.
    std::vector<uint8_t> decode_jpeg_reference(const std::vector<uint8_t>& jpeg) {
        jpeg_decompress_struct codec{};
        jpeg_error_mgr error{};
        codec.err = jpeg_std_error(&error);
        jpeg_create_decompress(&codec);
        jpeg_mem_src(&codec, jpeg.data(), static_cast<unsigned long>(jpeg.size()));
        jpeg_read_header(&codec, TRUE);
        codec.out_color_space = JCS_RGB;
        jpeg_start_decompress(&codec);
        const size_t row_bytes = static_cast<size_t>(codec.output_width) * 3;
        std::vector<uint8_t> pixels(row_bytes * codec.output_height);
        while (codec.output_scanline < codec.output_height) {
            JSAMPROW row = pixels.data() + codec.output_scanline * row_bytes;
            jpeg_read_scanlines(&codec, &row, 1);
        }
        jpeg_finish_decompress(&codec);
        jpeg_destroy_decompress(&codec);
        return pixels;
    }

    std::unique_ptr<NvCodecImageLoader> make_loader(const bool fallback) {
        NvCodecImageLoader::Options options;
        options.enable_fallback = fallback;
        options.decoder_pool_size = 1;
        return std::make_unique<NvCodecImageLoader>(options);
    }

    class NvCodecHipJpegTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
                GTEST_SKIP() << "GPU unavailable";
            }
            if (!NvCodecImageLoader::is_available()) {
                GTEST_SKIP() << "rocJPEG cannot decode JPEG on this system";
            }
        }
    };

} // namespace

// Streams rocJPEG does not support decode on the CPU when fallback is enabled, are
// refused with the reason when it is not, and are reported as CPU decodes.
TEST_F(NvCodecHipJpegTest, UnsupportedStreamsFallBackToTheCpuAndAreReported) {
    struct Case {
        const char* name;
        int width;
        int height;
        EncodeOptions options;
        const char* reason;
    };
    const std::vector<Case> cases{
        {"extended sequential", 128, 96, {.extended_sequential = true}, "extended sequential JPEG"},
        {"progressive", 128, 96, {.progressive = true}, "progressive JPEG"},
        {"4:1:1", 128, 96, {.luma_h = 4, .luma_v = 1}, "chroma sampling 4x1,1x1,1x1"},
        {"48x48", 48, 48, {}, "smaller than 64x64"},
    };
    auto fallback = make_loader(true);
    auto strict = make_loader(false);
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        const auto pixels = test_pixels(c.width, c.height, 7);
        const auto jpeg = encode_jpeg(pixels, c.width, c.height, c.options);

        const auto image = fallback->load_image_from_memory_gpu(jpeg, 1, 0, nullptr,
                                                                lfs::io::DecodeFormat::RGB, true);
        ASSERT_EQ(image.shape()[0], 3u);
        ASSERT_EQ(image.shape()[1], static_cast<size_t>(c.height));
        ASSERT_EQ(image.shape()[2], static_cast<size_t>(c.width));
        const auto decoded = image.cpu().to_vector();
        const auto reference = decode_jpeg_reference(jpeg);
        const size_t plane = static_cast<size_t>(c.width) * c.height;
        size_t mismatches = 0;
        for (size_t i = 0; i < plane; ++i)
            for (size_t ch = 0; ch < 3; ++ch)
                mismatches += std::lround(decoded[ch * plane + i]) != reference[i * 3 + ch];
        EXPECT_EQ(mismatches, 0u);

        try {
            (void)strict->load_image_from_memory_gpu(jpeg, 1, 0, nullptr, lfs::io::DecodeFormat::RGB, true);
            ADD_FAILURE() << "decoded without fallback";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("rocJPEG does not support this stream"), std::string::npos)
                << e.what();
            EXPECT_NE(std::string(e.what()).find(c.reason), std::string::npos) << e.what();
        }
    }
    EXPECT_EQ(NvCodecImageLoader::jpeg_decoder(), "rocJPEG (GPU), CPU (libjpeg) for JPEGs rocJPEG does not support");
}

// A structurally incomplete JPEG is a decode error, not a CPU decode, and fails the
// batch that holds it. These streams are refused before they reach the hardware
// decoder, which hangs on some of them (a stream that ends before its scan).
TEST_F(NvCodecHipJpegTest, CorruptJpegInABatchIsRejectedBeforeDecoding) {
    constexpr int width = 128;
    constexpr int height = 96;
    const auto good = encode_jpeg(test_pixels(width, height, 1), width, height);
    size_t sos = 0;
    for (size_t i = 2; i + 1 < good.size(); ++i) {
        if (good[i] == 0xFF && good[i + 1] == 0xDA) {
            sos = i;
            break;
        }
    }
    ASSERT_GT(sos, 0u);
    const size_t scan_data = sos + 2 + ((size_t{good[sos + 2]} << 8) | good[sos + 3]);
    const auto prefix = [&](const size_t n) { return std::vector<uint8_t>(good.begin(), good.begin() + n); };
    struct Case {
        const char* name;
        std::vector<uint8_t> jpeg;
        const char* defect;
    };
    std::vector<uint8_t> no_eoi = prefix(good.size() - 2);
    std::vector<uint8_t> empty_scan = prefix(scan_data);
    empty_scan.insert(empty_scan.end(), {0xFF, 0xD9});
    const std::vector<Case> cases{
        {"ends before the scan", prefix(sos), "ends before the first scan"},
        {"scan without data", empty_scan, "the scan has no data"},
        {"no end-of-image marker", no_eoi, "no end-of-image marker"},
    };

    auto loader = make_loader(true);
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        const std::vector<std::pair<const uint8_t*, size_t>> spans{
            {good.data(), good.size()}, {c.jpeg.data(), c.jpeg.size()}, {good.data(), good.size()}};
        try {
            (void)loader->decode_jpeg_batch_from_spans(spans, nullptr, true, true);
            ADD_FAILURE() << "a batch with a corrupt JPEG decoded";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("Corrupt JPEG"), std::string::npos) << e.what();
            EXPECT_NE(std::string(e.what()).find(c.defect), std::string::npos) << e.what();
        }
    }

    // The loader still decodes good input afterwards.
    const std::vector<std::pair<const uint8_t*, size_t>> good_spans{{good.data(), good.size()}};
    const auto decoded = loader->decode_jpeg_batch_from_spans(good_spans, nullptr, true, true);
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0].shape()[1], static_cast<size_t>(height));
    EXPECT_EQ(decoded[0].shape()[2], static_cast<size_t>(width));
}

// A frame that fails to encode comes back empty; the other frames still encode.
TEST_F(NvCodecHipJpegTest, BadFrameInABatchEncodeDropsOnlyThatFrame) {
    constexpr int width = 160;
    constexpr int height = 120;
    const size_t bytes = static_cast<size_t>(width) * height * 3;
    std::vector<void*> frames(3, nullptr);
    for (const size_t i : {size_t{0}, size_t{2}}) {
        ASSERT_EQ(cudaMalloc(&frames[i], bytes), cudaSuccess);
        const auto pixels = test_pixels(width, height, static_cast<uint32_t>(i));
        ASSERT_EQ(cudaMemcpy(frames[i], pixels.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    }

    auto loader = make_loader(true);
    const auto encoded = loader->encode_batch_rgb_to_jpeg(frames, width, height, 90);
    cudaFree(frames[0]);
    cudaFree(frames[2]);
    ASSERT_EQ(encoded.size(), frames.size());
    EXPECT_TRUE(encoded[1].empty());
    for (const size_t i : {size_t{0}, size_t{2}}) {
        ASSERT_FALSE(encoded[i].empty()) << i;
        const auto image = loader->load_image_from_memory_gpu(encoded[i], 1, 0, nullptr,
                                                              lfs::io::DecodeFormat::RGB, true);
        EXPECT_EQ(image.shape()[1], static_cast<size_t>(height));
        EXPECT_EQ(image.shape()[2], static_cast<size_t>(width));
    }
}

// Noise compresses poorly, so the encoder's output buffer grows several times.
TEST_F(NvCodecHipJpegTest, LargeEncodeGrowsTheOutputBuffer) {
    constexpr int width = 1536;
    constexpr int height = 1024;
    std::vector<uint8_t> noise(static_cast<size_t>(width) * height * 3);
    uint32_t state = 12345;
    for (auto& value : noise) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<uint8_t>(state >> 24);
    }
    auto image = lfs::core::Tensor::empty(
        lfs::core::TensorShape({size_t{height}, size_t{width}, size_t{3}}), lfs::core::Device::CUDA,
        lfs::core::DataType::UInt8);
    ASSERT_EQ(cudaMemcpy(image.data_ptr(), noise.data(), noise.size(), cudaMemcpyHostToDevice), cudaSuccess);

    auto loader = make_loader(true);
    const auto jpeg = loader->encode_to_jpeg(image, 100);
    EXPECT_GT(jpeg.size(), size_t{1} << 20);
    ASSERT_GE(jpeg.size(), 4u);
    EXPECT_EQ(jpeg[0], 0xFF);
    EXPECT_EQ(jpeg[1], 0xD8);
    EXPECT_EQ(jpeg[jpeg.size() - 2], 0xFF);
    EXPECT_EQ(jpeg[jpeg.size() - 1], 0xD9);
    const auto decoded = loader->load_image_from_memory_gpu(jpeg, 1, 0, nullptr, lfs::io::DecodeFormat::RGB, true);
    EXPECT_EQ(decoded.shape()[1], static_cast<size_t>(height));
    EXPECT_EQ(decoded.shape()[2], static_cast<size_t>(width));
}
