#!/usr/bin/env python3
"""
Detailed GC test to understand the memory leak.
"""

import gc
import sys
import weakref
import tracemalloc
import itertools
from dataclasses import dataclass
from typing import Sequence
from collections import defaultdict


# Exactly match the real KVCacheBlocks
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[object], ...]

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    def new_empty(self) -> "KVCacheBlocks":
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))

    def get_block_ids(self) -> tuple[list[int], ...]:
        return tuple([id(blk) for blk in group] for group in self.blocks)


# Simulate the coordinator's req_to_blocks
class SingleTypeKVCacheManager:
    def __init__(self):
        self.req_to_blocks: defaultdict[str, list] = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}

    def save_new_computed_blocks(self, request_id: str, new_computed_blocks: Sequence):
        if request_id not in self.num_cached_block:
            req_blocks = self.req_to_blocks[request_id]
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)

    def free(self, request_id: str):
        self.req_to_blocks.pop(request_id, None)
        self.num_cached_block.pop(request_id, None)


# Simulate large multimodal data
class MMFeature:
    def __init__(self, size_mb=5):
        self.data = bytearray(size_mb * 1024 * 1024)
        self.identifier = id(self)


@dataclass
class NewRequestData:
    req_id: str
    mm_features: list
    block_ids: tuple


class CachedRequestState:
    def __init__(self, req_id: str, mm_features: list, block_ids: tuple):
        self.req_id = req_id
        self.mm_features = mm_features
        self.block_ids = block_ids


class Request:
    def __init__(self, request_id: str, mm_features: list):
        self.request_id = request_id
        self.mm_features = mm_features


# Simulate the full flow
class KVCacheManager:
    def __init__(self, use_singleton: bool):
        self.use_singleton = use_singleton
        self.num_groups = 2
        self.coordinator = SingleTypeKVCacheManager()

        if use_singleton:
            self.empty_kv_cache_blocks = KVCacheBlocks(
                tuple(() for _ in range(self.num_groups))
            )

    def create_kv_cache_blocks(self, blocks: tuple) -> KVCacheBlocks:
        if self.use_singleton:
            return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
        else:
            return KVCacheBlocks(blocks)

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        # Simulate no cache hits (common for multimodal)
        empty_blocks = tuple([] for _ in range(self.num_groups))
        return self.create_kv_cache_blocks(empty_blocks), 0

    def allocate_slots(self, request: Request, new_computed_blocks: KVCacheBlocks):
        new_computed_block_list = new_computed_blocks.blocks
        self.coordinator.save_new_computed_blocks(request.request_id, new_computed_block_list[0])
        # Return new blocks (simulated)
        return self.create_kv_cache_blocks(tuple([] for _ in range(self.num_groups)))

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return self.create_kv_cache_blocks(tuple([] for _ in range(self.num_groups)))

    def free(self, request_id: str):
        self.coordinator.free(request_id)


class Scheduler:
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.requests: dict[str, Request] = {}

    def schedule(self, request: Request) -> NewRequestData:
        self.requests[request.request_id] = request

        # Get computed blocks (returns singleton for empty)
        computed_blocks, _ = self.kv_cache_manager.get_computed_blocks(request)

        # Allocate slots
        new_blocks = self.kv_cache_manager.allocate_slots(request, computed_blocks)

        # Create req_to_new_blocks (like real scheduler)
        req_to_new_blocks = {request.request_id: new_blocks}

        # Create NewRequestData (holds mm_features reference!)
        new_req_data = NewRequestData(
            req_id=request.request_id,
            mm_features=request.mm_features,
            block_ids=req_to_new_blocks[request.request_id].get_block_ids()
        )

        return new_req_data

    def finish_request(self, request_id: str):
        del self.requests[request_id]
        self.kv_cache_manager.free(request_id)


class ModelRunner:
    def __init__(self):
        self.requests: dict[str, CachedRequestState] = {}

    def process(self, new_req_data: NewRequestData):
        # Create CachedRequestState (holds mm_features reference!)
        state = CachedRequestState(
            req_id=new_req_data.req_id,
            mm_features=new_req_data.mm_features,
            block_ids=new_req_data.block_ids
        )
        self.requests[new_req_data.req_id] = state

    def finish_request(self, request_id: str):
        self.requests.pop(request_id, None)


def test_gc_behavior(use_singleton: bool, num_requests: int = 100):
    print(f"\n{'='*60}")
    print(f"Testing with singleton={'ENABLED' if use_singleton else 'DISABLED'}")
    print(f"{'='*60}")

    gc.collect()
    gc.set_debug(gc.DEBUG_STATS)

    kv_cache_manager = KVCacheManager(use_singleton=use_singleton)
    scheduler = Scheduler(kv_cache_manager)
    model_runner = ModelRunner()

    weak_refs = []
    weak_mm_refs = []

    for i in range(num_requests):
        # Create request with multimodal data
        mm_features = [MMFeature(size_mb=5)]
        request = Request(f"req_{i}", mm_features)

        # Track with weak references
        weak_refs.append(weakref.ref(request))
        weak_mm_refs.append(weakref.ref(mm_features[0]))

        # Full scheduling flow
        new_req_data = scheduler.schedule(request)
        model_runner.process(new_req_data)

        # Finish request
        scheduler.finish_request(request.request_id)
        model_runner.finish_request(request.request_id)

        # Delete local references
        del new_req_data
        del request
        del mm_features

        # Force GC every 10 requests
        if i % 10 == 9:
            gc.collect()

    # Final GC
    gc.collect()
    gc.collect()
    gc.collect()

    gc.set_debug(0)

    alive_requests = sum(1 for ref in weak_refs if ref() is not None)
    alive_mm = sum(1 for ref in weak_mm_refs if ref() is not None)

    print(f"\nResults after {num_requests} requests:")
    print(f"  Requests still alive: {alive_requests}")
    print(f"  MMFeatures still alive: {alive_mm}")

    # Check for garbage
    print(f"  Uncollectable garbage: {len(gc.garbage)}")

    return alive_requests, alive_mm


if __name__ == "__main__":
    req_alive_without, mm_alive_without = test_gc_behavior(use_singleton=False)
    req_alive_with, mm_alive_with = test_gc_behavior(use_singleton=True)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Without singleton: {req_alive_without} requests, {mm_alive_without} mm_features alive")
    print(f"With singleton: {req_alive_with} requests, {mm_alive_with} mm_features alive")

    if mm_alive_with > mm_alive_without:
        print("\n⚠️  Singleton pattern is preventing MM features from being GC'd!")
    else:
        print("\n✓ Both patterns GC mm_features similarly")
