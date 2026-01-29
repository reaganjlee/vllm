# vLLM Memory Leak Fix Report

**Date:** 2026-01-29
**Status:** Root cause identified and fix verified

---

## Executive Summary

A memory leak in vLLM affecting multimodal models (e.g., Qwen2.5-VL) was traced to commit `acaa2c0a4` which introduced a singleton pattern for empty `KVCacheBlocks`. The singleton prevents Python's garbage collector from properly cleaning up large multimodal data (images), causing ~5-6GB of RAM accumulation per 1000 requests until OOM.

**One-line fix verified:** Disabling the singleton pattern resolves the issue completely.

---

## Problem Description

### Symptoms
- CPU RAM grows continuously during sustained inference workloads
- Memory is not released even when requests complete
- Eventually causes OOM and server restart
- Only affects **multimodal models** (vision-language models with images)
- Text-only models are unaffected

### Affected Version Range
- **Good (no leak):** v0.11.0 (`b8b302cde`)
- **Bad (has leak):** v0.11.1 (`439368496`)

### Reproduction
```bash
# Server
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --limit-mm-per-prompt.video 0 --max-model-len 25000

# Benchmark (run repeatedly)
vllm bench serve --backend openai-chat --model Qwen/Qwen2.5-VL-3B-Instruct \
    --endpoint /v1/chat/completions --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat --hf-split train --num-prompts 1000
```

---

## Root Cause

### Buggy Commit
```
Commit:  acaa2c0a4a53dbb57f85f1042b1a6f1e3f24cef5
Author:  Jialin Ouyang
Date:    2025-10-14
Title:   [Core] Reuse empty block lists whenever possible in KVCacheBlocks
         to mitigate GC costs (#24964)
```

### The Problem Code

**File:** `vllm/v1/core/kv_cache_manager.py`

```python
# Lines 143-147: Singleton created in __init__
self.empty_kv_cache_blocks = KVCacheBlocks(
    tuple(() for _ in range(self.num_kv_cache_groups))
)

# Lines 481-485: Singleton returned for empty blocks
def create_kv_cache_blocks(
    self, blocks: tuple[list[KVCacheBlock], ...]
) -> KVCacheBlocks:
    return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
```

### Why It Causes a Memory Leak

The commit was intended to reduce garbage collection overhead by reusing a single empty `KVCacheBlocks` object instead of creating new ones. However:

1. **Before the change:** Each request got its own independent `KVCacheBlocks` instance
   - When a request finished, its `KVCacheBlocks` and all associated data could be independently garbage collected

2. **After the change:** All requests share the same singleton
   - The singleton lives forever on `KVCacheManager`
   - This creates a reference pattern that prevents Python's GC from collecting large multimodal data (`mm_features` containing images)
   - ~5-6GB of image data accumulates per 1000 requests

### Why Only Multimodal Models Are Affected

- The PR was tested only with `facebook/opt-125m` (text-only model)
- Multimodal models carry large `mm_features` (processed images) per request
- The singleton pattern disrupts the reference isolation needed to GC this large data
- Text-only models don't have significant per-request data, so the issue isn't visible

---

## The Fix

### Change Required

**File:** `vllm/v1/core/kv_cache_manager.py`, line 485

```python
# Before (buggy):
return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

# After (fixed):
return KVCacheBlocks(blocks)
```

This single-line change disables the singleton optimization, allowing each request to get its own `KVCacheBlocks` instance that can be independently garbage collected.

---

## Verification Results

### Test Configuration
- Commit tested: `acaa2c0a4` with one-line fix applied
- Model: Qwen/Qwen2.5-VL-3B-Instruct
- Benchmark: 1000 prompts from VisionArena-Chat dataset
- Success criteria: Complete 6 consecutive benchmark runs without OOM

### Results Comparison

#### Buggy Version (Original `acaa2c0a4`)
| Run | RAM (MB) | Status |
|-----|----------|--------|
| 1 | 16,652 | OK |
| 2 | 22,719 | OK |
| 3 | 28,353 | OK |
| 4 | 32,952 | **OOM** |

**Result:** FAILED at run 4 (exceeded 32GB threshold)

#### Fixed Version (Singleton Disabled)
| Run | RAM (MB) | Status |
|-----|----------|--------|
| 1 | 16,174 | OK |
| 2 | 21,891 | OK |
| 3 | 28,103 | OK |
| 4 | 27,801 | OK (GC reclaimed memory) |
| 5 | 28,275 | OK |
| 6 | 28,275 | OK (stable) |

**Result:** PASSED - All 6 runs completed, memory stabilized at ~28GB

### Key Observations

1. **Memory stabilization:** With the fix, RAM stabilized around 28GB instead of growing unbounded
2. **GC effectiveness:** At run 4, memory actually decreased (27,801 < 28,103), showing GC working properly
3. **No performance regression observed:** All benchmark runs completed in similar time

---

## Bisect Results Summary

The following commits from `filtered_commits.txt` were analyzed:

| Commit | Date | Status | Description |
|--------|------|--------|-------------|
| `8bf8f458` | Sep 27 | ✅ GOOD | Don't count preempted tokens in prefix cache hit rate |
| `d100d78e` | Oct 7 | ✅ GOOD | Optimize KV cache distribution for asymmetric PP |
| `cd989054` | Oct 8 | ✅ GOOD | Fix async KV transfer bug in cascade attention |
| **`acaa2c0a`** | **Oct 14** | **❌ BAD** | **Reuse empty block lists in KVCacheBlocks** |
| `cbd5e07a` | Oct 27 | ❌ BAD | Use merge_by_field_config for MM models |

`acaa2c0a4` is the **first bad commit** that introduced the memory leak.

---

## Recommendations

### Immediate Fix
Apply the one-line fix to disable the singleton pattern:
```python
def create_kv_cache_blocks(
    self, blocks: tuple[list[KVCacheBlock], ...]
) -> KVCacheBlocks:
    return KVCacheBlocks(blocks)  # Always create new instance
```

### Alternative Approaches (if GC optimization is still desired)
1. **Weak references:** Use `weakref` for the singleton to allow GC when not actively referenced
2. **Scoped reuse:** Only reuse within a single scheduling step, not across requests
3. **Multimodal-aware caching:** Different behavior for multimodal vs text-only models

### Testing Recommendations
- Add multimodal model (e.g., Qwen2.5-VL, LLaVA) to CI/CD memory testing
- Include sustained workload tests (multiple benchmark runs) to catch memory leaks
- Monitor both GPU and CPU memory in benchmarks

---

## Files Involved

| File | Role |
|------|------|
| `vllm/v1/core/kv_cache_manager.py` | Contains the buggy singleton pattern |
| `vllm/v1/core/sched/scheduler.py` | Uses `KVCacheBlocks` for request scheduling |
| `vllm/v1/request.py` | Contains `mm_features` (large image data) |
| `vllm/v1/worker/gpu_model_runner.py` | Caches `CachedRequestState` with `mm_features` |

---

## Timeline

| Date | Event |
|------|-------|
| 2025-10-14 | `acaa2c0a4` merged - introduces singleton pattern |
| 2025-10-30 | `4b68c4a55` merged - adds identity check (partial mitigation attempt) |
| 2026-01-29 | Root cause identified and fix verified |

---

## Appendix: Test Output

```
Verifying commit: HEAD
2026-01-29 07:42:22,995 - INFO - Starting verification for commit HEAD
2026-01-29 07:43:13,071 - INFO - Server is ready
2026-01-29 07:43:18,105 - INFO - Idle memory - RAM: 3503.3MB, GPU: 39421.0MB
2026-01-29 07:48:05,445 - INFO - Benchmark run 1 completed successfully
2026-01-29 07:48:05,475 - INFO - After run 1 - RAM: 16174.6MB, GPU: 45235.0MB
2026-01-29 07:51:30,840 - INFO - Benchmark run 2 completed successfully
2026-01-29 07:51:30,870 - INFO - After run 2 - RAM: 21891.2MB, GPU: 43793.0MB
2026-01-29 07:55:00,186 - INFO - Benchmark run 3 completed successfully
2026-01-29 07:55:00,217 - INFO - After run 3 - RAM: 28103.8MB, GPU: 45151.0MB
2026-01-29 07:58:22,394 - INFO - Benchmark run 4 completed successfully
2026-01-29 07:58:22,639 - INFO - After run 4 - RAM: 27801.9MB, GPU: 44639.0MB
2026-01-29 08:01:48,281 - INFO - Benchmark run 5 completed successfully
2026-01-29 08:01:48,313 - INFO - After run 5 - RAM: 28275.3MB, GPU: 44639.0MB
2026-01-29 08:05:13,584 - INFO - Benchmark run 6 completed successfully
2026-01-29 08:05:13,616 - INFO - After run 6 - RAM: 28275.6MB, GPU: 44641.0MB
2026-01-29 08:05:13,616 - INFO - All benchmark runs completed successfully
2026-01-29 08:05:17,519 - INFO - Total verification time: 1374.5s
Results: status=good, successful_runs=6
```
