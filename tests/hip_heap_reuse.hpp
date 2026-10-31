/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Takes every heap chunk from 16 B to 64 KiB and fills it with a 32-bit
// pattern, so memory freed just before (a destroyed stream object, say) is
// handed out again and overwritten, as any later allocation may do.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace lfs::test {

    class HeapReuse {
    public:
        explicit HeapReuse(const std::uint32_t pattern = 1) {
            for (std::size_t size = 16; size <= 64 * 1024; size += 16) {
                auto* const words = static_cast<std::uint32_t*>(std::malloc(size));
                if (words == nullptr) {
                    continue;
                }
                for (std::size_t i = 0; i < size / sizeof(std::uint32_t); ++i) {
                    words[i] = pattern;
                }
                chunks_.push_back(words);
            }
        }

        ~HeapReuse() {
            for (void* const chunk : chunks_) {
                std::free(chunk);
            }
        }

        HeapReuse(const HeapReuse&) = delete;
        HeapReuse& operator=(const HeapReuse&) = delete;

    private:
        std::vector<void*> chunks_;
    };

} // namespace lfs::test
