## Memory Leak Bisect Results

We identified a CPU memory leak affecting multimodal models (Qwen2-VL/Qwen2.5-VL) when prefix caching is enabled. Memory grows continuously with each request batch and is never released.

### Methodology
- **Test:** Run 6 consecutive benchmarks with 1000 multimodal prompts each
- **Pass criteria:** Complete all 6 runs with RAM staying under 32GB
- **Fail criteria:** RAM exceeds 32GB OR timeout/crash

### Bisect Results

p| Commit | Date | Status | Successful Runs | Final RAM | PR |
|--------|------|--------|-----------------|-----------|-----|
| `ddcbc2f3` | Oct 9 | ✅ GOOD | 6/6 | 27.7 GB | [#26450](https://github.com/vllm-project/vllm/pull/26450) |
| `2e54db4d` | Oct 9 | ✅ GOOD | 6/6 | 27.6 GB | [#26514](https://github.com/vllm-project/vllm/pull/26514) |
| **`ad430a67`** | **Oct 10** | **❌ BAD** | **4/6** | **35.7 GB** | [**#26285**](https://github.com/vllm-project/vllm/pull/26285) |
| `8a297115` | Oct 19 | ❌ BAD | 3/6 | 32.6 GB | [#27151](https://github.com/vllm-project/vllm/pull/27151) |
| `a55b6463` | Nov 16 | ❌ BAD | 4/6 | 35.7 GB | [#28194](https://github.com/vllm-project/vllm/pull/28194) |

### Summary

- **Total commits tested:** 10
- **First bad commit:** `ad430a67cab89ddc6060cf493f730c291826eb9d`
- **PR:** [#26285 - [Metrics] Log multi-modal cache stats and fix reset](https://github.com/vllm-project/vllm/pull/26285)
- **Author:** Cyrus Leung
- **Date:** 2025-10-10

### Memory Growth Comparison

| Commit | Status | RAM Growth per Run |
|--------|--------|-------------------|
| `ddcbc2f3` | ✅ GOOD | ~4.0 GB/run (stabilizes) |
| `2e54db4d` | ✅ GOOD | ~4.0 GB/run (stabilizes) |
| **`ad430a67`** | **❌ BAD** | **~6.5 GB/run (never stabilizes)** |

### Reproduction

```bash
# Server
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --limit-mm-per-prompt.video 0 --max-model-len 25000

# Benchmark (run repeatedly, observe RAM growth)
vllm bench serve --backend openai-chat --model Qwen/Qwen2.5-VL-3B-Instruct \
    --endpoint /v1/chat/completions --dataset-name hf \
    --dataset-path lmarena-ai/VisionArena-Chat --hf-split train --num-prompts 1000
```

**Workaround:** Disable prefix caching with `--enable-prefix-caching=false`
