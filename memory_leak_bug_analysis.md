# Memory Leak Bug Analysis - Commit ad430a67

## Overview

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

### Memory Growth Pattern

Commits after `ad430a67`:
- ~27-30GB after 6 benchmark runs
- ~4-4.4GB growth per run

Older commit `8bf8f458`:
- Only ~22.5GB after 6 runs
- ~3.2GB growth per run

---

## Comprehensive Line-by-Line Analysis

### 1. vllm/inputs/preprocess.py

#### LEAK SUSPECT #1: Line 60
```python
self.mm_cache_stats = MultiModalCacheStats() if mm_processor_cache else None
```
**Issue:** Creates a stats object that continuously accumulates data
**Why:** Stats are only reset when `stat_mm_cache()` is called, not automatically
**Severity:** Medium

#### LEAK SUSPECT #2: Lines 713-715
```python
self.mm_cache_stats.requests += 1
self.mm_cache_stats.queries += delta.total
self.mm_cache_stats.hits += delta.hits
```
**Issue:** Unbounded accumulation
**Why:** These counters grow indefinitely until `stat_mm_cache()` is called. If logging is disabled or infrequent, stats accumulate unbounded
**Severity:** High

#### LEAK SUSPECT #3: Lines 720-726
```python
def stat_mm_cache(self) -> Optional[MultiModalCacheStats]:
    mm_cache_stats = self.mm_cache_stats
    if mm_cache_stats is None:
        return None

    self.mm_cache_stats = MultiModalCacheStats()

    return mm_cache_stats
```
**Issue:** Returns stats object that caller may hold
**Why:** The returned object is stored by callers (see loggers.py). Creates a reference chain
**Severity:** Critical - **This is part of the leak chain**

---

### 2. vllm/multimodal/cache.py

#### LEAK SUSPECT #4: Lines 451-453 in ShmObjectStoreSenderCache.__init__
```python
self._hits = 0
self._total = 0
self._last_info = CacheInfo(hits=0, total=0)
```
**Issue:** `_hits` and `_total` grow unbounded
**Why:** Only reset in `clear_cache()`, accumulate forever otherwise
**Note:** Integers don't leak memory directly, but if they're part of a larger object that's retained...
**Severity:** Low-Medium

#### LEAK SUSPECT #5: Lines 457-463
```python
def _stat(self, *, delta: bool = False) -> CacheInfo:
    info = CacheInfo(hits=self._hits, total=self._total)

    if delta:
        info_delta = info - self._last_info
        self._last_info = info  # STORES NEW OBJECT
        info = info_delta

    return info
```
**Issue:** `self._last_info` stores new CacheInfo objects
**Why:** Every call with `delta=True` creates and stores a new CacheInfo. Called on EVERY request preprocessing!
**Severity:** High - **This creates many small objects**

#### LEAK SUSPECT #6: Lines 476-479 & 485
```python
self._hits += 1
self._total += 1
# ... later ...
self._total += 1
```
**Issue:** Called on every cache operation
**Why:** If these increments happen millions of times, the integer objects themselves may contribute to memory (Python integers can grow)
**Severity:** Low

---

### 3. vllm/v1/metrics/loggers.py

#### LEAK SUSPECT #7: Line 63
```python
self.last_mm_cache_stats: Optional[MultiModalCacheStats] = None
```
**Issue:** Instance variable that stores stats objects
**Why:** Declares storage for stats objects
**Severity:** N/A (declaration only)

#### LEAK SUSPECT #8: Lines 115-117 ⭐ **PRIMARY SUSPECT**
```python
if mm_cache_stats:
    self.mm_caching_metrics.observe(mm_cache_stats)

    self.last_mm_cache_stats = mm_cache_stats  # STORES REFERENCE!
```
**Issue:** **PRIMARY SUSPECT FOR MEMORY LEAK**
**Why:** Stores reference to MultiModalCacheStats object returned from `stat_mm_cache()`. This prevents the stats object from being garbage collected. While it's overwritten each logging cycle, if there are multiple logger instances or if the stats contain large data, this accumulates
**Critical:** The stats object being stored may hold references to the accumulated cache data structures
**Severity:** CRITICAL - **This is the smoking gun**

---

### 4. vllm/v1/metrics/stats.py

#### LEAK SUSPECT #9: Line 50 in CachingMetrics.__init__
```python
self.query_queue = deque[tuple[int, int, int]]()
```
**Issue:** Deque that stores tuples
**Why:** While there's cleanup logic, if `stats.requests == 0` returns early (line 71-72), empty stats never get added but the deque never shrinks
**Severity:** Medium

#### LEAK SUSPECT #10: Lines 75-78
```python
self.query_queue.append((stats.requests, stats.queries, stats.hits))
self.aggregated_requests += stats.requests
self.aggregated_query_total += stats.queries
self.aggregated_query_hit += stats.hits
```
**Issue:** Appends to deque unconditionally
**Why:** Cleanup logic (lines 83-90) only triggers when `aggregated_requests > max_recent_requests`. If individual stats have `requests=0` often, cleanup may not trigger properly
**Severity:** Medium

#### LEAK SUSPECT #11: Lines 83-90
```python
while (
    len(self.query_queue) > 1
    and self.aggregated_requests > self.max_recent_requests
):
    old_requests, old_queries, old_hits = self.query_queue.popleft()
    self.aggregated_requests -= old_requests
    self.aggregated_query_total -= old_queries
    self.aggregated_query_hit -= old_hits
```
**Issue:** Cleanup condition may not trigger
**Why:** Requires `aggregated_requests > 1000`. With multimodal, if each stat reports fractional or zero requests frequently, this threshold might never be hit
**Severity:** Medium-High

---

### 5. vllm/v1/worker/worker_base.py

#### LEAK SUSPECT #12: Lines 310-313 in WorkerWrapperBase.__init__
```python
self.mm_receiver_cache = worker_receiver_cache_from_config(
    self.vllm_config,
    MULTIMODAL_REGISTRY,
    shared_worker_lock,
)
```
**Issue:** Creates cache instances
**Why:** If multiple workers are created and not properly cleaned up, each holds a cache
**Severity:** Low

#### LEAK SUSPECT #13: Lines 357-360 in _apply_mm_cache
```python
for req_data in scheduler_output.scheduled_new_reqs:
    req_data.mm_features = mm_cache.get_and_update_features(
        req_data.mm_features
    )
```
**Issue:** Modifies request data with cached features
**Why:** If `req_data` objects are retained after request completion, they hold multimodal features
**Severity:** Medium

---

## Most Likely Culprits (Ranked)

1. **⭐ vllm/v1/metrics/loggers.py:117** - `self.last_mm_cache_stats = mm_cache_stats`
   - **TOP SUSPECT - Stores references preventing garbage collection**

2. **vllm/inputs/preprocess.py:713-715** - Unbounded stat accumulation if logging is disabled

3. **vllm/multimodal/cache.py:461** - `self._last_info = info` creates new objects on every request

4. **vllm/v1/metrics/stats.py:75** - Deque growth if cleanup logic doesn't trigger

5. **vllm/inputs/preprocess.py:726** - Returning stats objects that are stored by callers

---

## Investigation Method

1. Git bisect identified commit `ad430a67` as the first bad commit
2. Analyzed full diff of the commit
3. Identified all new data structures and accumulation patterns
4. Traced object lifetime and reference chains
5. Identified potential garbage collection prevention patterns

---

## Next Steps

1. Check which of these suspects are still present in current main code
2. Verify if any fixes were applied in subsequent commits
3. Test memory usage with specific suspects patched
4. Use memory profiling tools (tracemalloc) to confirm the leak source

---

## Status in Current Main Code (as of commit 71b1c8b66)

### ✅ FIXED (Not Present in Current Code)

| Suspect # | Location | Status | Notes |
|-----------|----------|--------|-------|
| **#7** | vllm/v1/metrics/loggers.py:63 | ✅ REMOVED | `self.last_mm_cache_stats` declaration removed |
| **#8** | vllm/v1/metrics/loggers.py:117 | ✅ REMOVED | **PRIMARY SUSPECT - Assignment removed, now only calls observe()** |

### ❌ STILL PRESENT (Potential Ongoing Issues)

| Suspect # | Location | Status | Severity | Notes |
|-----------|----------|--------|----------|-------|
| **#1** | vllm/inputs/preprocess.py:60 | ❌ PRESENT | Medium | Stats object creation |
| **#2** | vllm/inputs/preprocess.py:713-715 | ❌ PRESENT | High | Unbounded accumulation if logging disabled |
| **#3** | vllm/inputs/preprocess.py:720-726 | ❌ PRESENT | Medium | Returns stats object (but caller no longer stores it) |
| **#4** | vllm/multimodal/cache.py:451-453 | ❌ PRESENT | Low-Medium | Counter accumulation |
| **#5** | vllm/multimodal/cache.py:460 | ❌ PRESENT | **High** | `self._last_info = info` creates objects on every request |
| **#6** | vllm/multimodal/cache.py:476-477, 485 | ❌ PRESENT | Low | Integer increments |
| **#9** | vllm/v1/metrics/stats.py:50 | ❌ PRESENT | Medium | Deque growth |
| **#10** | vllm/v1/metrics/stats.py:75-78 | ❌ PRESENT | Medium | Deque append |
| **#11** | vllm/v1/metrics/stats.py:83-90 | ❌ PRESENT | Medium-High | Cleanup condition logic |
| **#12** | vllm/v1/worker/worker_base.py:291 | ❌ PRESENT | Low | Cache instance creation |
| **#13** | vllm/v1/worker/worker_base.py:340-342 | ❌ PRESENT | Medium | Request data modification |

---

## Key Findings

### What Was Fixed
The **primary suspect (#8)** - storing `mm_cache_stats` reference in `self.last_mm_cache_stats` - has been removed from the current code. This was the most obvious garbage collection prevention pattern.

### What's Still Present
Despite the fix for suspect #8, **the memory leak persists** in the current main code, which means:

1. **The real culprit is one (or more) of the remaining suspects** (#1-6, #9-13)

2. **Top remaining suspects for the ongoing leak:**
   - **Suspect #5** (vllm/multimodal/cache.py:460) - `self._last_info = info`
     - Creates new CacheInfo object on EVERY preprocessed request
     - Called with `delta=True` on every request
     - **This is highly suspicious** for continuous object creation

   - **Suspect #2** (vllm/inputs/preprocess.py:713-715) - Unbounded stat accumulation
     - If `stat_mm_cache()` isn't called frequently enough, stats grow unbounded
     - Counters increment on every request

   - **Suspect #11** (vllm/v1/metrics/stats.py:83-90) - Cleanup condition
     - May not trigger properly with multimodal workloads
     - Deque could grow without bounds

### New Primary Suspects (Ranked)

Given that #8 was fixed but the leak persists:

1. **🔴 Suspect #5** - `self._last_info = info` in ShmObjectStoreSenderCache
   - Creates object on every request preprocessing
   - No apparent cleanup mechanism
   - **HIGH PROBABILITY**

2. **🟠 Suspect #2** - Unbounded accumulation in InputPreprocessor
   - Depends on logging frequency
   - Could grow unbounded if logging is slow
   - **MEDIUM-HIGH PROBABILITY**

3. **🟠 Suspect #11** - CachingMetrics deque cleanup
   - Cleanup logic may not trigger correctly
   - Could cause deque growth
   - **MEDIUM PROBABILITY**

---

## Summary & Conclusion

### Original Problem (Commit ad430a67)
The commit introduced multimodal cache statistics tracking which created a memory leak when using prefix caching with multimodal models.

### Partial Fix Applied
The most obvious bug (Suspect #8) - storing references to `MultiModalCacheStats` objects in `self.last_mm_cache_stats` - was removed in a subsequent commit (e51928192).

### Current Situation
**The memory leak still exists in current main code**, indicating that:
- Either the fix was incomplete
- Or there are multiple independent leaks working together

### Most Likely Current Culprit
**Suspect #5: `self._last_info = info` in vllm/multimodal/cache.py:460**

This line executes on EVERY request that goes through preprocessing when `delta=True` is passed to `_stat()`. Since the `make_stats(delta=True)` is called from `preprocess()` for every single request:
- It creates a new `CacheInfo` object every time
- Stores it in `self._last_info`
- The old object may not be properly garbage collected
- Over millions of requests, this could accumulate significant memory

### Recommended Next Steps
1. **Add memory profiling** to track `CacheInfo` object creation
2. **Test patch for Suspect #5**: Don't store `_last_info`, or use a different delta calculation method
3. **Monitor** if logging is actually being called regularly (Suspect #2)
4. **Verify** deque cleanup is working correctly (Suspect #11)
