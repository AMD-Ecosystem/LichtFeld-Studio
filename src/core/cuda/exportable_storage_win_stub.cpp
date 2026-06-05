/* Windows stub for exportable_storage.cpp -- CUDA VMM IPC not supported in
 * the compute-tranche HIP build. The compute tests do not call these functions.
 * MOAT ROCm/HIP port: stub added for Windows build. */
#if defined(_WIN32) && defined(USE_HIP)

#include "core/exportable_storage.hpp"
#include <stdexcept>

namespace lfs::core {

std::expected<std::shared_ptr<ExportableBlock>, std::string>
allocateExportableDeviceBlock(std::size_t, int, bool, std::size_t) {
    return std::unexpected(
        "allocateExportableDeviceBlock: CUDA VMM IPC not available in compute-only Windows HIP build");
}

std::expected<bool, std::string>
growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>&, std::size_t) {
    return std::unexpected(
        "growExportableDeviceBlock: CUDA VMM IPC not available in compute-only Windows HIP build");
}

} // namespace lfs::core

#endif // _WIN32 && USE_HIP
