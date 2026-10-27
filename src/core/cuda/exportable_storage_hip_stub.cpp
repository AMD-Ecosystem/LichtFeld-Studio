/* Stand-in for exportable_storage.cpp in the compute-only HIP build, which has
 * no viewer to share memory with. */
#if defined(USE_HIP)

#include "core/exportable_storage.hpp"
#include <stdexcept>

namespace lfs::core {

std::size_t exportable_allocation_granularity(int) {
    return std::size_t{2} << 20;
}

std::expected<std::shared_ptr<ExportableBlock>, std::string>
allocateExportableDeviceBlock(std::size_t, int, bool, std::size_t) {
    return std::unexpected(
        "allocateExportableDeviceBlock: VMM export is not part of the compute-only HIP build");
}

std::expected<void, std::string>
commitExportableDeviceRange(const std::shared_ptr<ExportableBlock>&, std::size_t, std::size_t) {
    return std::unexpected(
        "commitExportableDeviceRange: VMM export is not part of the compute-only HIP build");
}

std::expected<bool, std::string>
growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>&, std::size_t) {
    return std::unexpected(
        "growExportableDeviceBlock: VMM export is not part of the compute-only HIP build");
}

std::expected<bool, std::string>
shrinkExportableDeviceBlock(const std::shared_ptr<ExportableBlock>&, std::size_t) {
    return std::unexpected(
        "shrinkExportableDeviceBlock: VMM export is not part of the compute-only HIP build");
}

} // namespace lfs::core

#endif // USE_HIP
