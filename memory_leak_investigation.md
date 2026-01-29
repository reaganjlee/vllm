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
