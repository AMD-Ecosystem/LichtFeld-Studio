/* Windows stub for image_io.cpp -- OpenImageIO not required for compute tests.
 * The compute test tranche does not call image I/O functions.
 * MOAT ROCm/HIP port: stub added for Windows build. */
#if defined(_WIN32)

#include "core/image_io.hpp"
#include <stdexcept>

namespace lfs::core {

std::tuple<int, int, int> get_image_info(std::filesystem::path) {
    throw std::runtime_error("image_io: not available in compute-only Windows build");
}
std::tuple<unsigned char*, int, int, int>
load_image_with_alpha(std::filesystem::path, int, int) {
    throw std::runtime_error("image_io: not available in compute-only Windows build");
}
std::tuple<unsigned char*, int, int, int>
load_image_from_memory(const uint8_t*, size_t) {
    throw std::runtime_error("image_io: not available in compute-only Windows build");
}
std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path, int, int) {
    throw std::runtime_error("image_io: not available in compute-only Windows build");
}
void save_image(const std::filesystem::path&, Tensor) {
    throw std::runtime_error("image_io: not available in compute-only Windows build");
}
void save_image(const std::filesystem::path&, const std::vector<Tensor>&, bool, int) {
    throw std::runtime_error("image_io: not available in compute-only Windows build");
}
bool save_img_data(const std::filesystem::path&,
                   const std::tuple<unsigned char*, int, int, int>&) {
    throw std::runtime_error("image_io: not available in compute-only Windows build");
}
void free_image(unsigned char*) {}

} // namespace lfs::core

namespace lfs::core::image_io {

BatchImageSaver& BatchImageSaver::instance() {
    throw std::runtime_error("BatchImageSaver: not available in compute-only Windows build");
}
BatchImageSaver* BatchImageSaver::try_instance() { return nullptr; }
void BatchImageSaver::wait_all_if_initialized() {}
size_t BatchImageSaver::pending_count_if_initialized() { return 0; }
void BatchImageSaver::queue_save(const std::filesystem::path&, lfs::core::Tensor) {
    throw std::runtime_error("BatchImageSaver: not available in compute-only Windows build");
}
void BatchImageSaver::queue_save_multiple(const std::filesystem::path&,
                                          const std::vector<lfs::core::Tensor>&,
                                          bool, int) {
    throw std::runtime_error("BatchImageSaver: not available in compute-only Windows build");
}
void BatchImageSaver::wait_all() {}
void BatchImageSaver::shutdown() {}
size_t BatchImageSaver::pending_count() const { return 0; }
BatchImageSaver::BatchImageSaver(size_t) {}
BatchImageSaver::~BatchImageSaver() {}
void BatchImageSaver::worker_thread() {}
void BatchImageSaver::process_task(const SaveTask&) {}
void BatchImageSaver::enqueue_task(SaveTask) {}

} // namespace lfs::core::image_io

#endif // _WIN32
