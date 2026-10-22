/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "core/cuda/memory_arena.hpp"

namespace fast_lfs::rasterization::detail {

    inline bool native_gradient_atomics_disabled(const char* value) noexcept {
        return value != nullptr && value[0] != '\0' && value[0] != '0' &&
               value[0] != 'f' && value[0] != 'F' && value[0] != 'n' && value[0] != 'N';
    }

    bool native_gradient_atomics_disabled_from_environment() noexcept;

    inline bool native_gradient_atomics_arch_supported(std::string_view arch) noexcept {
        constexpr std::array<std::string_view, 6> supported_arches{
            "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"};
        for (const auto target : supported_arches) {
            if (arch.starts_with(target) &&
                (arch.size() == target.size() || arch[target.size()] == ':')) {
                return true;
            }
        }
        return false;
    }

    inline bool select_native_gradient_atomics(
        const bool requested,
        const std::array<bool, 4>& gradient_ranges_owned_by_current_device_arena,
        const bool architecture_supported,
        const bool disabled) noexcept {
        return requested && !disabled &&
               gradient_ranges_owned_by_current_device_arena[0] &&
               gradient_ranges_owned_by_current_device_arena[1] &&
               gradient_ranges_owned_by_current_device_arena[2] &&
               gradient_ranges_owned_by_current_device_arena[3] &&
               architecture_supported;
    }

    inline bool native_gradient_atomics_eligible(
        const lfs::core::RasterizerMemoryArena& arena,
        const std::array<const void*, 4>& gradient_ranges,
        const int n_primitives,
        const bool architecture_supported,
        const bool disabled) {
        if (n_primitives <= 0) {
            return false;
        }
        const auto primitive_count = static_cast<size_t>(n_primitives);
        const std::array<size_t, 4> range_sizes{
            primitive_count * 2 * sizeof(float),
            primitive_count * 3 * sizeof(float),
            primitive_count * sizeof(float),
            primitive_count * 3 * sizeof(float)};
        std::array<bool, 4> owned{};
        for (size_t i = 0; i < owned.size(); ++i) {
            owned[i] = arena.owns_device_allocation(gradient_ranges[i], range_sizes[i]);
        }
        return select_native_gradient_atomics(
            true, owned, architecture_supported, disabled);
    }

} // namespace fast_lfs::rasterization::detail
