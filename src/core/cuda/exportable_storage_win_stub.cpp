/* Windows stub for exportable_storage.cpp -- CUDA VMM IPC not supported in
 * the compute-tranche HIP build. The compute tests do not call these functions.
 * ROCm/HIP port: stub added for Windows build. */
#if defined(_WIN32) && defined(USE_HIP)

#include "core/exportable_storage.hpp"
#include <stdexcept>

namespace lfs::core {

std::size_t exportable_allocation_granularity(int) {
    return std::size_t{2} << 20;
}

std::expected<std::shared_ptr<ExportableBlock>, std::string>
allocateExportableDeviceBlock(std::size_t, int, bool, std::size_t) {
    return std::unexpected(
        "allocateExportableDeviceBlock: CUDA VMM IPC not available in compute-only Windows HIP build");
}

std::expected<void, std::string>
commitExportableDeviceRange(const std::shared_ptr<ExportableBlock>&, std::size_t, std::size_t) {
    return std::unexpected(
        "commitExportableDeviceRange: CUDA VMM IPC not available in compute-only Windows HIP build");
}

std::expected<bool, std::string>
growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>&, std::size_t) {
    return std::unexpected(
        "growExportableDeviceBlock: CUDA VMM IPC not available in compute-only Windows HIP build");
}

std::expected<bool, std::string>
shrinkExportableDeviceBlock(const std::shared_ptr<ExportableBlock>&, std::size_t) {
    return std::unexpected(
        "shrinkExportableDeviceBlock: CUDA VMM IPC not available in compute-only Windows HIP build");
}

} // namespace lfs::core

#endif // _WIN32 && USE_HIP
