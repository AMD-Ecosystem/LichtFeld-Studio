/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/* Windows stub for image_codecs.cpp and image_exr.cpp in the ROCm compute
 * build, which has no libjpeg/libpng/libtiff/libwebp/OpenEXR packages. The
 * compute tests never decode or encode image files. Every entry point reports
 * failure through the normal error path. */
#if defined(_WIN32)

#include "image_codecs.hpp"
#include "image_exr.hpp"

namespace lfs::core::image_codecs {

    namespace {
        bool unavailable(std::string& error) {
            error = "image codecs are not available in the ROCm compute-only Windows build";
            return false;
        }
    } // namespace

    bool probe(const std::filesystem::path&, Probe&, std::string& error) { return unavailable(error); }
    bool probe_bounded_header(const std::filesystem::path&, std::string& error) { return unavailable(error); }
    bool radiance_scanlines_complete(const std::filesystem::path&) { return false; }
    bool decode(const std::filesystem::path&, Image&, std::string& error) { return unavailable(error); }
    bool decode_to_buffer(const std::filesystem::path&, DecodeTarget&, Probe&, std::string& error) {
        return unavailable(error);
    }
    bool decode_memory(const std::uint8_t*, size_t, Image&, std::string& error) { return unavailable(error); }
    bool decode_memory_to_buffer(const std::uint8_t*, size_t, DecodeTarget&, Probe&, std::string& error) {
        return unavailable(error);
    }
    bool decode_jpeg_memory(const std::uint8_t*, size_t, Image&, std::string& error) { return unavailable(error); }

    bool write_jpeg(const std::filesystem::path&, const std::uint8_t*, int, int, int, int,
                    const std::optional<std::string>&, std::string& error) {
        return unavailable(error);
    }
    bool write_jpeg(const std::filesystem::path&, const std::uint8_t*, int, int, int, int,
                    const std::optional<std::string>&, std::string& error, bool) {
        return unavailable(error);
    }
    bool write_png(const std::filesystem::path&, const void*, int, int, int, int, int,
                   const std::optional<std::string>&, std::string& error) {
        return unavailable(error);
    }
    bool write_tiff(const std::filesystem::path&, const std::uint8_t*, int, int, int, std::string& error) {
        return unavailable(error);
    }

    bool probe_exr(const std::filesystem::path&, Probe&, std::string& error) { return unavailable(error); }
    bool decode_exr(const std::filesystem::path&, Image&, std::string& error, unsigned) { return unavailable(error); }

} // namespace lfs::core::image_codecs

#endif // _WIN32
