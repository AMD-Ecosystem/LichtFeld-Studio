/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Pinned GT-cache blocks must not record CUDA events on streams severed via
// release_stream(), because the caller-supplied deleter stream may already be
// destroyed during teardown.

#include "core/pinned_memory_allocator.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

using lfs::core::PinnedMemoryAllocator;

namespace {

    class PinnedStreamTeardownTest : public ::testing::Test {
    protected:
        void SetUp() override {
            ASSERT_EQ(cudaFree(nullptr), cudaSuccess); // ensure context
        }
    };

} // namespace

// After release_stream(s), deallocate(ptr, s) must not record another use event
// on a stream that has been declared ready for destruction.
TEST_F(PinnedStreamTeardownTest, DeallocateAfterReleaseStreamRecordsNoEvent) {
    auto& alloc = PinnedMemoryAllocator::instance();

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    void* ptr = alloc.allocate(1 << 20);
    ASSERT_NE(ptr, nullptr);
    alloc.record_stream(ptr, stream);

    alloc.release_stream(stream);

    const auto before = alloc.get_stats();
    alloc.deallocate(ptr, stream);
    const auto after = alloc.get_stats();

    EXPECT_EQ(after.use_events_recorded, before.use_events_recorded)
        << "deallocate recorded a use event on a severed stream";

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Preserve the teardown order: record a use, sever the stream, destroy it, then
// deallocate using the stale handle. The final operation must not touch the
// destroyed stream.
TEST_F(PinnedStreamTeardownTest, DeallocateWithDestroyedSeveredStreamIsSafe) {
    auto& alloc = PinnedMemoryAllocator::instance();

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    void* ptr = alloc.allocate(1 << 20);
    ASSERT_NE(ptr, nullptr);
    alloc.record_stream(ptr, stream);

    alloc.release_stream(stream);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    alloc.deallocate(ptr, stream); // must not touch the dead handle
    SUCCEED();
}

// Handle-reuse: a recycled stream pointer that re-enters service via
// record_stream() must be treated as live again (tombstone cleared) so its
// pending work is still fenced by a real event.
TEST_F(PinnedStreamTeardownTest, RecycledStreamHandleIsLiveAgainAfterRecordStream) {
    auto& alloc = PinnedMemoryAllocator::instance();

    cudaStream_t s1 = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), cudaSuccess);
    alloc.release_stream(s1); // tombstoned
    ASSERT_EQ(cudaStreamDestroy(s1), cudaSuccess);

    // New stream; may or may not reuse the handle value. Either way, once it is
    // recorded on a live allocation it must be event-fenced at deallocate.
    cudaStream_t s2 = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking), cudaSuccess);

    void* ptr = alloc.allocate(1 << 20);
    ASSERT_NE(ptr, nullptr);
    alloc.record_stream(ptr, s2);

    const auto before = alloc.get_stats();
    alloc.deallocate(ptr, s2);
    const auto after = alloc.get_stats();

    EXPECT_GE(after.use_events_recorded, before.use_events_recorded + 1)
        << "live stream use was not event-fenced";

    ASSERT_EQ(cudaStreamDestroy(s2), cudaSuccess);
}
#if defined(USE_HIP)

#include "hip_heap_reuse.hpp"

#include <cstdint>
#include <thread>

namespace tensor_hardening {
    // tensor_hardening_cuda_utils.cu
    cudaError_t launch_delay_kernel(cudaStream_t stream, uint64_t cycles);
} // namespace tensor_hardening

// A cached block keeps the use event recorded on a stream that is later
// released and destroyed. Reusing the block must not depend on the destroyed
// stream's memory, which the heap is free to hand out again: HIP dereferences
// an event's recording stream in hipEventQuery.
TEST_F(PinnedStreamTeardownTest, CachedBlockReuseAfterStreamDestroyIgnoresFreedStream) {
    auto& alloc = PinnedMemoryAllocator::instance();
    constexpr size_t bytes = 1 << 20;

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    void* ptr = alloc.allocate(bytes);
    ASSERT_NE(ptr, nullptr);
    alloc.deallocate(ptr, stream);
    alloc.release_stream(stream);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    const lfs::test::HeapReuse heap;
    void* reused = alloc.allocate(bytes);
    EXPECT_NE(reused, nullptr);
    alloc.deallocate(reused);
}

// empty_cache() takes the cached blocks under the allocator lock and waits on
// their use events after dropping it. A stream released and destroyed while
// it waits on an earlier event must not break the wait on a later one.
TEST_F(PinnedStreamTeardownTest, EmptyCacheWaitOutlivesConcurrentStreamDestroy) {
    auto& alloc = PinnedMemoryAllocator::instance();
    alloc.empty_cache();

    cudaStream_t slow_stream = nullptr;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&slow_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    // The block's events are waited on in order: first the slow stream's
    // (about 2 s), then the one on the stream destroyed meanwhile.
    void* ptr = alloc.allocate(1 << 20);
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(tensor_hardening::launch_delay_kernel(slow_stream, 2'000'000'000), cudaSuccess);
    alloc.record_stream(ptr, slow_stream);
    alloc.deallocate(ptr, stream);
    ASSERT_GT(alloc.get_stats().cached_bytes, 0u);

    std::thread releaser([&alloc] { alloc.empty_cache(); });
    while (alloc.get_stats().cached_bytes != 0) {
        std::this_thread::yield();
    }
    alloc.release_stream(stream);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    {
        const lfs::test::HeapReuse heap;
        releaser.join();
    }

    alloc.release_stream(slow_stream);
    EXPECT_EQ(cudaStreamDestroy(slow_stream), cudaSuccess);
}
#endif
