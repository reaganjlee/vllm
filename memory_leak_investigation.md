# vLLM Memory Leak Investigation Context Document

## Summary

**Problematic Commit:** `ad430a67cab89ddc6060cf493f730c291826eb9d`
**Title:** [Metrics] Log multi-modal cache stats and fix reset (#26285)
**Author:** Cyrus Leung
**Date:** 2025-10-10

## Problem Description

When using vLLM online serving with multimodal models (specifically Qwen2-VL-2B/Qwen2.5-VL-3B), CPU memory usage continuously increases over time. Even when no new requests are coming in, memory is not released, eventually causing the server to restart due to OOM.

### Key Observations
- The memory leak occurs with **prefix caching enabled** (default)
- Disabling the multimodal input processor cache did NOT fix the issue
- **Disabling prefix caching DID fix the issue** - CPU memory remained stable
- This points to an interaction between prefix caching and multimodal processing

## Affected Version Range

- **Good (no leak):** v0.11.0 (`b8b302cde434df8c9289a2b465406b47ebab1c2d`)
- **Bad (has leak):** v0.11.1 (`439368496db48d8f992ba8c606a0c0b1eebbfa69`)

## Bisect Results

Using automated bisect verification with the following criteria:
- **Success:** Benchmark completes 6 consecutive times
- **Failure:** RAM exceeds 32GB threshold OR OOM/crash/timeout

### Commits Tested (newest to oldest)

| Commit | Status | Runs | RAM After 6 Runs | Notes |
|--------|--------|------|------------------|-------|
| ad430a67 | **BAD** | 0/6 | - | Timeout during benchmark |
| 2e54db4d | GOOD | 6/6 | 27.6GB | |
| ddcbc2f3 | GOOD | 6/6 | 27.7GB | |
| d24cf322 | GOOD | 6/6 | 29.7GB | |
| cd989054 | GOOD | 6/6 | 30.1GB | |
| d100d78e | GOOD | 6/6 | 30.0GB | |
| 201c971e | GOOD | 6/6 | 29.8GB | |
| 8bf8f458 | GOOD | 6/6 | **22.5GB** | Notably lower memory usage |

### Memory Growth Pattern

Commits after `ad430a67`:
- ~27-30GB after 6 benchmark runs
- ~4-4.4GB growth per run

Older commit `8bf8f458`:
- Only ~22.5GB after 6 runs
- ~3.2GB growth per run

## Files Most Likely Related to the Bug

Based on investigation, these files are the primary suspects for the memory leak:

### Core Prefix Caching
| File | Relevance |
|------|-----------|
| `vllm/v1/core/kv_cache_utils.py` | Block hashing utilities, `generate_block_hash_extra_keys()` |
| `vllm/v1/core/block_pool.py` | `BlockHashToBlockMap` caches blocks by hash |
| `vllm/v1/core/kv_cache_manager.py` | Block allocation/deallocation logic |

### Request & State Management
| File | Relevance |
|------|-----------|
| `vllm/v1/request.py` | Request class holds `block_hashes` and `mm_features` lists |
| `vllm/v1/core/sched/scheduler.py` | Request lifecycle, `_free_request()` cleanup |

### Multimodal + Encoder Cache (Critical)
| File | Relevance |
|------|-----------|
| `vllm/v1/core/encoder_cache_manager.py` | Encoder cache for multimodal - tracks `mm_hash` → request IDs |
| `vllm/multimodal/hasher.py` | `MultiModalHasher` for hashing multimodal inputs |
| `vllm/multimodal/cache.py` | Multimodal processor cache (LRU cache) |

## The Problematic Commit

```
commit ad430a67cab89ddc6060cf493f730c291826eb9d
Author: Cyrus Leung
Date:   2025-10-10

    [Metrics] Log multi-modal cache stats and fix reset (#26285)
```

### What to Look For

Given the commit title mentions "multi-modal cache stats and fix reset", investigate:

1. **Cache reset logic** - Is something not being properly reset/cleared?
2. **Stats tracking** - Are references being held for metrics that prevent GC?
3. **Multimodal cache interaction with prefix caching** - The leak only manifests when prefix caching is enabled

### Likely Root Causes

1. **Reference retention in hash maps** - `BlockHashToBlockMap` or encoder cache may hold references to multimodal data that aren't released when requests complete

2. **Incomplete cleanup in `_free_request()`** - When a request finishes, its `block_hashes` and `mm_features` may not be fully dereferenced

3. **Cache stats objects holding references** - If the metrics/stats changes in this commit store references to cache entries or request data

4. **Hash computation callbacks** - The `block_hasher` callable stored in Request might capture references in closures

## Reproduction Steps

### Server Command
```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --limit-mm-per-prompt.video 0 --max-model-len 25000
```

### Benchmark Command (run repeatedly)
```bash
vllm bench serve --backend openai-chat --model Qwen/Qwen2.5-VL-3B-Instruct \
    --endpoint /v1/chat/completions --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat --hf-split train --num-prompts 1000
```

### Verification
- Monitor RAM usage after each benchmark run
- With the bug: RAM grows ~4-6GB per run, never releases
- Without the bug (prefix caching disabled): RAM stabilizes after initial warmup

## Tools Created

The following scripts were created for this investigation:

1. **`bisect_verify.py`** - Automated git bisect verification script
   - Starts vLLM server, runs benchmarks, monitors memory
   - Supports `--bisect` for binary search, `--run-all` for linear scan
   - Outputs results to `bisect_results.csv`

2. **`analyze_results.py`** - Results analysis script
   - Displays commits in order with test results
   - Shows memory progression for each tested commit
   - Identifies first bad / last good commits

3. **`target_commits.csv`** - List of commits to test between v0.11.0 and v0.11.1

## Next Steps for Debugging

1. Review the diff of `ad430a67cab89ddc6060cf493f730c291826eb9d`
2. Look for any new data structures that store references
3. Check if cache reset/clear methods properly release all references
4. Add memory profiling (e.g., `tracemalloc`) to identify growing objects
5. Test with the specific changes from this commit reverted

---

# Addendum: Filtered Commits Deep Analysis (2026-01-29)

This section provides detailed analysis of commits from `filtered_commits.txt` that touch KV cache and memory management code.

## Commits Analyzed

| Commit | Lines Changed | Description | Suspicion Level |
|--------|---------------|-------------|-----------------|
| `acaa2c0a4` | +39/-20 | Reuse empty block lists in KVCacheBlocks to mitigate GC costs | **HIGH** ⚠️ |
| `cd9890544` | +19/-29 | Fix async KV transfer bug in cascade attention | Low |
| `8bf8f4582` | +53/-40 | Don't count preempted tokens in prefix cache hit rate | Low |
| `d100d78eb` | +55/-29 | Optimize KV cache distribution for asymmetric pipeline parallelism | Very Low |
| `cbd5e07a5` | +2/-46 | Use merge_by_field_config for MM models (Qwen series) | Very Low |
| `bd51f78e3` | +1/-49 | Remove V0 condition for mm embeddings merge | Very Low |
| `9c5ee91b2` | +18/-22 | Fix vit flash attn dispatcher logic for ROCm | N/A (ROCm only) |

---

## Detailed Analysis

### 1. `acaa2c0a4` - **HIGH SUSPICION** ⚠️

**Title**: "Reuse empty block lists whenever possible in KVCacheBlocks to mitigate GC costs"

**Files Changed**:
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`

**Summary of Changes**:

This commit introduces object reuse to reduce garbage collection overhead. Ironically, this pattern is a classic source of memory leaks.

#### Key Change 1: Shared Singleton for Empty Blocks

```python
# In KVCacheManager.__init__
self.empty_kv_cache_blocks = KVCacheBlocks(
    tuple(() for _ in range(self.num_kv_cache_groups))
)
```

#### Key Change 2: Returns Shared Instance Instead of New Objects

```python
def create_kv_cache_blocks(
    self, blocks: tuple[list[KVCacheBlock], ...]
) -> KVCacheBlocks:
    # Only create new KVCacheBlocks for non-empty blocks
    return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
```

#### Key Change 3: Type Signature Change

- Changed from: `blocks: tuple[list[KVCacheBlock], ...]`
- Changed to: `blocks: tuple[Sequence[KVCacheBlock], ...]`
- Empty blocks now use immutable tuples `()` instead of mutable lists `[]`

#### Key Change 4: Modified `__add__` Method

```python
def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
    return KVCacheBlocks(
        tuple(
            list(itertools.chain(blk1, blk2))
            for blk1, blk2 in zip(self.blocks, other.blocks)
        )
    )
```

#### Key Change 5: Multiple Call Sites Return Singleton

The shared `empty_kv_cache_blocks` is now returned from:
- `get_computed_blocks()` - when prompt_logprobs is set
- `allocate_slots()` - for new_computed_block_list default
- `get_blocks()` - wraps coordinator results
- Scheduler directly accesses `self.kv_cache_manager.empty_kv_cache_blocks`

**Why This Could Cause Memory Leak**:

1. **Reference accumulation**: Code that stores what it expects to be a temporary empty KVCacheBlocks object now holds a reference to a long-lived singleton. This can prevent garbage collection of related objects.

2. **Unexpected object lifetime**: Previously, fresh empty objects would be GC'd when no longer needed. Now they're references to a singleton that lives for the KVCacheManager's entire lifetime.

3. **Type change side effects**: Changing from mutable `list` to immutable `tuple` could cause issues if any code:
   - Checks type/identity of blocks
   - Attempts modification (fails silently or raises)
   - Uses objects as dictionary keys or in sets

4. **Shared state contamination**: Any accidental modification to the singleton affects all users.

**Investigation Priority**: HIGH - Test this commit specifically with:
```bash
git checkout acaa2c0a4^  # Before
# Run benchmark
git checkout acaa2c0a4   # After
# Run benchmark
```

---

### 2. `cd9890544` - **LOW SUSPICION**

**Title**: "Fix async KV transfer bug in cascade attention"

**Files Changed**:
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`

**Summary**: Changes how "common prefix blocks" are counted for cascade attention.

```python
# Before:
if block.ref_cnt == num_running_requests:  # count passed from scheduler

# After:
if block.ref_cnt == len(self.req_to_blocks):  # count from dictionary size
```

**Why Unlikely to Cause Leak**:
- Only changes counting/calculation logic
- No changes to allocation, deallocation, or reference management
- `req_to_blocks` is read-only in this context
- Does not affect how blocks are freed

---

### 3. `8bf8f4582` - **LOW SUSPICION**

**Title**: "Don't count preempted tokens in prefix cache hit rate"

**Files Changed**:
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/metrics/stats.py`
- `vllm/v1/request.py`

**Summary**: Adds preemption counter and separate stats tracking.

```python
# Added to Request class
self.num_preemptions = 0

# Incremented on preemption
preempted_req.num_preemptions += 1
```

**Why Unlikely to Cause Leak**:
- Only adds integer counters
- No changes to object lifecycle or references
- Scheduler refactoring is purely structural

---

### 4. `d100d78eb` - **VERY LOW SUSPICION**

**Title**: "Optimize KV cache distribution for asymmetric pipeline parallelism"

**Summary**: Moves logging to separate function and **shrinks** tensor sizes:

```python
for tensor in kv_cache_config.kv_cache_tensors:
    tensor.size = tensor.size // num_blocks_old * min_num_blocks
```

**Why Unlikely to Cause Leak**:
- Actually REDUCES memory allocation
- Changes in initialization, not runtime
- No reference management changes

---

### 5-7. Other Commits - **VERY LOW / N/A**

- **`cbd5e07a5`**: Removes code (-46 lines), configuration changes only
- **`bd51f78e3`**: Removes deprecated V0 code (-49 lines)
- **`9c5ee91b2`**: ROCm-specific, irrelevant on NVIDIA hardware

---

## Relationship to Primary Investigation

The primary investigation (above) identified `ad430a67` ("[Metrics] Log multi-modal cache stats and fix reset") as the problematic commit. However:

1. **`acaa2c0a4` is NOT in the bisect results table** - It may not have been tested
2. **`acaa2c0a4` touches the same core files** - `block_pool.py`, `kv_cache_manager.py`
3. **Both commits relate to caching/GC** - `ad430a67` deals with cache stats reset, `acaa2c0a4` deals with object reuse for GC

**Possible scenarios**:
- Both commits contribute to the leak (compounding effect)
- `acaa2c0a4` creates conditions that `ad430a67` exploits
- They are independent issues

## Bisect Results - CONFIRMED

**`acaa2c0a4` IS THE BUG.** The bisect testing confirms this commit introduced the memory leak:

| Commit | Date | Status | Successful Runs | Final RAM | Notes |
|--------|------|--------|-----------------|-----------|-------|
| `8bf8f458` | Sep 27 | ✅ GOOD | 6/6 | 25,927 MB | Stable baseline |
| `d100d78e` | Oct 7 | ✅ GOOD | 6/6 | 30,894 MB | |
| `cd989054` | Oct 8 | ✅ GOOD | 6/6 | 30,630 MB | Last good commit |
| **`acaa2c0a`** | **Oct 14** | **❌ BAD** | **3/6** | **OOM at 32,952 MB** | **FIRST BAD COMMIT** |
| `cbd5e07a` | Oct 27 | ❌ BAD | 4/6 | OOM at 36,814 MB | Inherits bug |

### Memory Growth Pattern

**Before `acaa2c0a4`** (commit `8bf8f458`):
- Run 1: 11,300 MB → Run 6: 25,927 MB
- Growth: ~2.9 GB per run, stabilizes

**After `acaa2c0a4`**:
- Run 1: 16,652 MB → Run 4: 32,952 MB (OOM)
- Growth: ~5.4 GB per run, does NOT stabilize
- Fails before completing 6 runs

## Root Cause Analysis

The bug is in the singleton reuse pattern introduced by `acaa2c0a4`. The commit creates a shared `empty_kv_cache_blocks` instance that is returned instead of creating new empty objects:

```python
# The problematic singleton
self.empty_kv_cache_blocks = KVCacheBlocks(
    tuple(() for _ in range(self.num_kv_cache_groups))
)

# Returns shared instance instead of new object
def create_kv_cache_blocks(self, blocks: tuple[list[KVCacheBlock], ...]) -> KVCacheBlocks:
    return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
```

### Why This Causes Memory Leak

1. **Reference accumulation**: Code that stores references to "empty" blocks now holds references to a long-lived singleton, which may prevent garbage collection of associated request data.

2. **Type change from `list` to `tuple`**: Empty blocks changed from mutable `[]` to immutable `()`. Any code that previously relied on list identity or mutability may behave unexpectedly.

3. **Shared across all requests**: The singleton is shared across ALL requests, potentially creating a reference web that prevents cleanup.

## Recommended Fix

Revert the singleton pattern or ensure that the shared instance doesn't prevent GC of request-scoped data. Options:

1. **Full revert**: `git revert acaa2c0a4`
2. **Partial fix**: Keep the optimization but ensure no request data is transitively reachable from the singleton
3. **Weak references**: If reuse is needed, use weak references for any request-scoped data

## Relationship to `ad430a67`

The original investigation identified `ad430a67` ("[Metrics] Log multi-modal cache stats and fix reset", Oct 10) as problematic. This commit comes BEFORE `acaa2c0a4` chronologically but was NOT in the filtered commits list.

Possible explanations:
- `ad430a67` may have introduced a **different** memory issue
- `acaa2c0a4` may have **exacerbated** an existing issue from `ad430a67`
- The two commits may interact to cause the leak

Both commits should be investigated if the fix for `acaa2c0a4` alone doesn't fully resolve the issue.
