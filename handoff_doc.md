# vLLM Git Bisect Verification Script

## Objective
Create a Python script to automate git bisect verification for vLLM, detecting memory-related regressions between v0.11.0 and v0.11.1.

## Commit Range
- **Good (start)**: v0.11.0 (`b8b302cde434df8c9289a2b465406b47ebab1c2d`)
- **Bad (end)**: v0.11.1 (`439368496db48d8f992ba8c606a0c0b1eebbfa69`)
- **Filtered commits**: 15 Python-only commits touching target files

## Target Files Being Investigated
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/encoder_cache_manager.py`
- `vllm/v1/request.py`

## Success/Failure Criteria
- **SUCCESS**: Benchmark completes successfully 6 consecutive times
- **FAILURE**: Any of the following before 6 successful runs:
  - Memory error (OOM, CUDA out of memory, allocation failure)
  - Server crash
  - Timeout (5 minutes per benchmark run)
  - Any other fatal error

## Commands

### Server Start
```bash
vllm serve Qwen2.5-VL-3B-Instruct --limit-mm-per-prompt.video 0 --max-model-len 25000
```
- Wait for server to be ready by polling health endpoint
- Server startup timeout: 120 seconds

### Benchmark (run 6 times)
```bash
vllm bench serve --backend openai-chat --model Qwen2.5-VL-3B-Instruct --endpoint /v1/chat/completions --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --hf-split train --num-prompts 1000
```
- Timeout per run: 5 minutes (300 seconds)
- Timeout is a safety net, not the primary failure mode

## Memory Error Detectio5n
Catch and log any of these patterns (case-insensitive):
- `CUDA out of memory`
- `OutOfMemoryError`
- `torch.cuda.OutOfMemoryError`
- `RuntimeError: CUDA error`
- `MemoryError`
- `std::bad_alloc`
- `cudaMalloc failed`
- `CUDA error: out of memory`
- `CUDA error: an illegal memory access`
- `cuMemAlloc failed`

## Output Requirements

### 1. Per-Commit Log Files
Location: `logs/{commit_hash}.log`

Contents:
- Timestamp for each operation
- Full stdout/stderr from server startup
- Memory readings at each checkpoint
- Full stdout/stderr from each benchmark run (labeled run1, run2, etc.)
- Any error messages captured
- Final status

### 2. DataFrame Output (CSV)
File: `bisect_results.csv`

| Column | Type | Description |
|--------|------|-------------|
| commit_hash | str | Full 40-char SHA |
| short_hash | str | First 8 chars of SHA |
| timestamp | datetime | Commit timestamp |
| author | str | Commit author |
| message | str | Commit message (first line) |
| status | str | "good" / "bad" / "skip" / "error" |
| error_type | str | null, "OOM", "timeout", "crash", "server_startup" |
| error_message | str | First 500 chars of error message if failed |
| successful_runs | int | Number of successful benchmark runs (0-6) |
| ram_idle_mb | float | System RAM after server startup (before benchmarks) |
| ram_run1_mb | float | System RAM after benchmark run 1 |
| ram_run2_mb | float | System RAM after benchmark run 2 |
| ram_run3_mb | float | System RAM after benchmark run 3 |
| ram_run4_mb | float | System RAM after benchmark run 4 |
| ram_run5_mb | float | System RAM after benchmark run 5 |
| ram_run6_mb | float | System RAM after benchmark run 6 |
| gpu_mem_idle_mb | float | GPU memory after server startup |
| gpu_mem_run1_mb | float | GPU memory after benchmark run 1 |
| gpu_mem_run2_mb | float | GPU memory after benchmark run 2 |
| gpu_mem_run3_mb | float | GPU memory after benchmark run 3 |
| gpu_mem_run4_mb | float | GPU memory after benchmark run 4 |
| gpu_mem_run5_mb | float | GPU memory after benchmark run 5 |
| gpu_mem_run6_mb | float | GPU memory after benchmark run 6 |
| gpu_mem_peak_mb | float | Peak GPU memory observed during all runs |
| total_duration_sec | float | Total time for this commit's verification |
| log_file | str | Relative path to detailed log file |

### 3. Memory Measurement Implementation
```python
import psutil
import subprocess

def get_ram_mb() -> float:
    """Get current process tree RAM usage in MB"""
    # Get vllm server process and children
    # Return RSS in MB
    pass

def get_gpu_mem_mb() -> float:
    """Get GPU memory usage in MB"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return float(result.stdout.strip().split('\n')[0])  # First GPU
```

## Script Structure
```python
#!/usr/bin/env python3
"""
vLLM Git Bisect Verification Script

Usage:
    # Manual single-commit test
    python bisect_verify.py --commit abc123
    
    # With git bisect
    git bisect start v0.11.1 v0.11.0
    git bisect run python bisect_verify.py
    
    # Test current HEAD
    python bisect_verify.py
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import requests

# Constants
SERVER_CMD = [
    "vllm", "serve", "Qwen2.5-VL-3B-Instruct",
    "--limit-mm-per-prompt.video", "0",
    "--max-model-len", "25000"
]
BENCHMARK_CMD = [
    "vllm", "bench", "serve",
    "--backend", "openai-chat",
    "--model", "Qwen2.5-VL-3B-Instruct",
    "--endpoint", "/v1/chat/completions",
    "--dataset-name", "hf",
    "--dataset-path", "lmarena-ai/VisionArena-Chat",
    "--hf-split", "train",
    "--num-prompts", "1000"
]
SERVER_HEALTH_URL = "http://localhost:8000/health"
SERVER_STARTUP_TIMEOUT = 120
BENCHMARK_TIMEOUT = 300
REQUIRED_SUCCESSFUL_RUNS = 6
LOG_DIR = Path("logs")
RESULTS_FILE = Path("bisect_results.csv")

MEMORY_ERROR_PATTERNS = [
    "cuda out of memory",
    "outofmemoryerror",
    "torch.cuda.outofmemoryerror",
    "runtimeerror: cuda error",
    "memoryerror",
    "std::bad_alloc",
    "cudamalloc failed",
    "cuda error: out of memory",
    "cuda error: an illegal memory access",
    "cumemalloc failed",
]


def get_current_commit() -> str:
    """Get current HEAD commit hash"""
    pass


def get_commit_info(commit_hash: str) -> dict:
    """Get commit metadata (timestamp, author, message)"""
    pass


def get_ram_mb() -> float:
    """Get system RAM usage in MB"""
    pass


def get_gpu_mem_mb() -> float:
    """Get GPU memory usage in MB"""
    pass


def check_for_memory_error(output: str) -> tuple[bool, Optional[str]]:
    """
    Check if output contains memory errors.
    Returns (is_memory_error, error_type)
    """
    pass


def start_server(log_file) -> subprocess.Popen:
    """Start vLLM server, return process handle"""
    pass


def wait_for_server_ready(timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll health endpoint until ready or timeout"""
    pass


def stop_server(process: subprocess.Popen) -> None:
    """Gracefully stop server, force kill if needed"""
    pass


def run_benchmark(timeout: int = BENCHMARK_TIMEOUT) -> tuple[bool, str, Optional[str]]:
    """
    Run benchmark once.
    Returns (success, stdout+stderr, error_type if failed)
    """
    pass


def verify_commit(commit_hash: str) -> dict:
    """
    Main verification function for a single commit.
    
    1. Start server
    2. Wait for ready
    3. Record idle memory
    4. Run benchmark up to 6 times
    5. Record memory after each run
    6. Stop server
    7. Return results dict matching CSV columns
    """
    pass


def append_results_to_csv(results: dict) -> None:
    """Append results to CSV, creating file with headers if needed"""
    pass


def main():
    parser = argparse.ArgumentParser(description="vLLM bisect verification")
    parser.add_argument("--commit", help="Specific commit to test (default: HEAD)")
    args = parser.parse_args()
    
    commit = args.commit or get_current_commit()
    
    LOG_DIR.mkdir(exist_ok=True)
    
    results = verify_commit(commit)
    append_results_to_csv(results)
    
    # Exit codes for git bisect
    if results["status"] == "good":
        sys.exit(0)
    elif results["status"] == "skip":
        sys.exit(125)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## Error Handling

| Scenario | status | error_type | Exit Code |
|----------|--------|------------|-----------|
| Server fails to start | bad | server_startup | 1 |
| OOM during benchmark | bad | OOM | 1 |
| Benchmark timeout | bad | timeout | 1 |
| Non-memory crash | bad | crash | 1 |
| 6 successful runs | good | null | 0 |
| Build/checkout fails | skip | build_error | 125 |

## Graceful OOM Handling
When OOM is detected:
1. Log the full error message
2. Attempt to kill the server process cleanly
3. Wait for GPU memory to be released (poll nvidia-smi)
4. Record the run number where failure occurred
5. Set appropriate status and error fields
6. Continue to CSV write (don't crash the script)

## Dependencies
```
psutil
pandas
requests
```

## File Structure
```
/workspace/vllm/
├── bisect_verify.py
├── bisect_results.csv
└── logs/
    ├── a55b64635....log
    ├── e15601789....log
    └── ...
```