/* Windows stub for mesh_data.cpp -- OpenMesh not required for compute tests.
 * The compute test tranche does not call compute_normals().
 * ROCm/HIP port: stub added for Windows build. */
#if defined(_WIN32)

#include "core/mesh_data.hpp"
#include <stdexcept>

namespace lfs::core {

void MeshData::compute_normals() {
    throw std::runtime_error("compute_normals: OpenMesh not available in compute-only Windows build");
}

} // namespace lfs::core

#endif // _WIN32
