#!/usr/bin/env python3
"""
Test script to understand why singleton KVCacheBlocks affects GC of multimodal data.
"""

import gc
import sys
import weakref
from dataclasses import dataclass
from typing import Sequence
import tracemalloc

# Simulate the KVCacheBlocks dataclass
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[object], ...]

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        return KVCacheBlocks(
            tuple(list(b1) + list(b2) for b1, b2 in zip(self.blocks, other.blocks))
        )


# Simulate large multimodal data (like images)
class MMFeature:
    def __init__(self, size_mb=5):
        # Allocate ~size_mb of memory
        self.data = bytearray(size_mb * 1024 * 1024)
        self.identifier = id(self)


# Simulate a Request with mm_features
class Request:
    def __init__(self, request_id: str, mm_features: list):
        self.request_id = request_id
        self.mm_features = mm_features


# Simulate KVCacheManager
class KVCacheManager:
    def __init__(self, use_singleton: bool):
        self.use_singleton = use_singleton
        self.num_groups = 2

        if use_singleton:
            # The singleton pattern from the buggy commit
            self.empty_kv_cache_blocks = KVCacheBlocks(
                tuple(() for _ in range(self.num_groups))
            )

    def create_kv_cache_blocks(self, blocks: tuple) -> KVCacheBlocks:
        if self.use_singleton:
            return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
        else:
            # Always create new object
            return KVCacheBlocks(blocks)

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        # Simulate the case where we return empty blocks (no cache hit)
        empty_blocks = tuple([] for _ in range(self.num_groups))
        return self.create_kv_cache_blocks(empty_blocks), 0


# Simulate the scheduler's handling of requests
class Scheduler:
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.requests: dict[str, Request] = {}
        self.req_to_blocks: dict[str, KVCacheBlocks] = {}

    def add_request(self, request: Request):
        self.requests[request.request_id] = request

        # Get computed blocks (this returns the singleton for empty blocks)
        computed_blocks, _ = self.kv_cache_manager.get_computed_blocks(request)

        # Store the KVCacheBlocks
        self.req_to_blocks[request.request_id] = computed_blocks

    def finish_request(self, request_id: str):
        # Clean up like the real scheduler does
        del self.requests[request_id]
        del self.req_to_blocks[request_id]


def test_gc_behavior(use_singleton: bool, num_requests: int = 100):
    """Test if requests are properly garbage collected."""

    print(f"\n{'='*60}")
    print(f"Testing with singleton={'ENABLED' if use_singleton else 'DISABLED'}")
    print(f"{'='*60}")

    tracemalloc.start()
    gc.collect()

    snapshot1 = tracemalloc.take_snapshot()

    kv_cache_manager = KVCacheManager(use_singleton=use_singleton)
    scheduler = Scheduler(kv_cache_manager)

    # Track weak references to requests to see when they're GC'd
    weak_refs = []

    # Process requests
    for i in range(num_requests):
        # Create request with ~5MB of multimodal data
        mm_features = [MMFeature(size_mb=5)]
        request = Request(f"req_{i}", mm_features)

        # Track with weak reference
        weak_refs.append(weakref.ref(request))

        # Add to scheduler
        scheduler.add_request(request)

        # Finish request (should allow GC)
        scheduler.finish_request(request.request_id)

        # Delete local reference
        del request
        del mm_features

    # Force garbage collection
    gc.collect()
    gc.collect()
    gc.collect()

    # Check how many requests were actually GC'd
    alive_count = sum(1 for ref in weak_refs if ref() is not None)
    gc_count = num_requests - alive_count

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print(f"\nResults:")
    print(f"  Requests processed: {num_requests}")
    print(f"  Requests GC'd: {gc_count}")
    print(f"  Requests still alive: {alive_count}")

    print(f"\nTop memory allocations:")
    for stat in top_stats[:10]:
        print(f"  {stat}")

    tracemalloc.stop()

    return alive_count


if __name__ == "__main__":
    # Test without singleton (should GC properly)
    alive_without = test_gc_behavior(use_singleton=False)

    # Test with singleton (might not GC properly)
    alive_with = test_gc_behavior(use_singleton=True)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Without singleton: {alive_without} requests still alive")
    print(f"With singleton: {alive_with} requests still alive")

    if alive_with > alive_without:
        print("\n⚠️  Singleton pattern is preventing garbage collection!")
    else:
        print("\n✓ Both patterns have similar GC behavior in this simplified test")
        print("  The issue may involve more complex interactions in the real code.")
