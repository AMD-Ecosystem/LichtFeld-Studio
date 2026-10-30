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

#include <cstdint>
#include <cstdlib>
#include <vector>

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

    std::vector<void*> heap;
    for (size_t size = 16; size <= 64 * 1024; size += 16) {
        auto* const words = static_cast<std::uint32_t*>(std::malloc(size));
        ASSERT_NE(words, nullptr);
        for (size_t i = 0; i < size / sizeof(std::uint32_t); ++i) {
            words[i] = 1;
        }
        heap.push_back(words);
    }

    void* reused = alloc.allocate(bytes);
    EXPECT_NE(reused, nullptr);
    alloc.deallocate(reused);
    for (void* chunk : heap) {
        std::free(chunk);
    }
}
#endif
