#!/usr/bin/env python3
"""
vLLM Git Bisect Verification Script

Usage:
    # Binary search through filtered commits (recommended)
    python bisect_verify.py --bisect

    # Linear scan through all filtered commits
    python bisect_verify.py --run-all

    # Manual single-commit test
    python bisect_verify.py --commit abc123

    # Test current HEAD
    python bisect_verify.py

    # Test with multimodal processor cache disabled
    python bisect_verify.py --bisect --disable-mm-cache

    # With native git bisect (skips commits not in filtered list)
    git bisect start v0.11.1 v0.11.0
    git bisect run python bisect_verify.py --skip-unlisted

Notes:
    - Uses target_commits.csv as the list of commits to test
    - Skips commits already in bisect_results.csv (uses cached results)
    - Results are saved to bisect_results.csv
    - Detailed logs are saved to logs/<short_hash>.log
    - Use --disable-mm-cache to test with --mm-processor-cache-gb 0
"""

import argparse
import csv
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import requests

# Constants
BASE_SERVER_CMD = [
    "vllm", "serve", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--limit-mm-per-prompt.video", "0",
    "--max-model-len", "25000"
]
BENCHMARK_CMD = [
    "vllm", "bench", "serve",
    "--backend", "openai-chat",
    "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
    "--endpoint", "/v1/chat/completions",
    "--dataset-name", "hf",
    "--dataset-path", "lmarena-ai/VisionArena-Chat",
    "--hf-split", "train",
    "--num-prompts", "1000"
]

# Global flags for mm processor cache (set by command line)
DISABLE_MM_CACHE = False
USE_DEPRECATED_MM_FLAG = False
SERVER_HEALTH_URL = "http://localhost:8000/health"
SERVER_STARTUP_TIMEOUT = 600  # 10 minutes for model download/load
BENCHMARK_TIMEOUT = 300
BENCHMARK_TIMEOUT_MM_CACHE_DISABLED = 600  # 10 minutes when mm cache is disabled
DEFAULT_REQUIRED_RUNS = 6
REQUIRED_SUCCESSFUL_RUNS = 6  # Can be overridden by --num-runs
RAM_THRESHOLD_MB = 32 * 1024  # 32GB - fail if RAM exceeds this
LOG_DIR = Path("logs")
RESULTS_FILE = Path("bisect_results.csv")
COMMITS_FILE = Path("filtered_commits.txt")

# Commit range to search (inclusive)
# Set to None to use all commits in the file
RANGE_START = None  # Use all commits from filtered_commits.txt
RANGE_END = None    # Use all commits from filtered_commits.txt

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

# CSV columns
CSV_COLUMNS = [
    "commit_hash", "short_hash", "timestamp", "author", "message",
    "status", "error_type", "error_message", "successful_runs",
    "mm_cache_disabled",
    "ram_idle_mb", "ram_run1_mb", "ram_run2_mb", "ram_run3_mb",
    "ram_run4_mb", "ram_run5_mb", "ram_run6_mb", "ram_run7_mb",
    "ram_run8_mb", "ram_run9_mb", "ram_run10_mb",
    "gpu_mem_idle_mb", "gpu_mem_run1_mb", "gpu_mem_run2_mb",
    "gpu_mem_run3_mb", "gpu_mem_run4_mb", "gpu_mem_run5_mb",
    "gpu_mem_run6_mb", "gpu_mem_run7_mb", "gpu_mem_run8_mb",
    "gpu_mem_run9_mb", "gpu_mem_run10_mb",
    "gpu_mem_peak_mb", "total_duration_sec", "log_file"
]


def load_commit_list(filepath: Path) -> list[str]:
    """Load commit hashes from the commits file (CSV or text)"""
    commits = []
    if not filepath.exists():
        return commits

    with open(filepath) as f:
        first_line = f.readline().strip()
        f.seek(0)

        # Check if it's a CSV with header
        if first_line.startswith('commit_hash'):
            reader = csv.DictReader(f)
            for row in reader:
                commit_hash = row.get('commit_hash', '').strip()
                if commit_hash:
                    commits.append(commit_hash)
        else:
            # Plain text format
            for line in f:
                line = line.strip()
                if line:
                    # Extract commit hash (first field before space or comma)
                    commit_hash = line.split()[0].split(',')[0]
                    commits.append(commit_hash)

    # Filter to specified range if set
    if RANGE_START or RANGE_END:
        start_idx = 0
        end_idx = len(commits)

        for i, commit in enumerate(commits):
            if RANGE_START and commit.startswith(RANGE_START):
                start_idx = i
            if RANGE_END and commit.startswith(RANGE_END):
                end_idx = i + 1  # Include the end commit

        commits = commits[start_idx:end_idx]

    return commits


def is_commit_in_list(commit_hash: str, commit_list: list[str]) -> bool:
    """Check if commit (full or short hash) matches any in the list"""
    short_hash = commit_hash[:9]  # Match the 9-char format in the file
    for c in commit_list:
        if c.startswith(short_hash) or short_hash.startswith(c):
            return True
    return False


def load_existing_results() -> dict[tuple[str, bool], str]:
    """Load existing results from CSV, returns dict of (short_hash, mm_cache_disabled) -> status"""
    results = {}
    if not RESULTS_FILE.exists():
        return results

    with open(RESULTS_FILE, 'r', newline='') as f:
        # Peek at first line to check if it's a header
        first_line = f.readline().strip()
        if not first_line:
            return results

        f.seek(0)  # Reset to beginning

        # Check if first line is header (starts with 'commit_hash') or data (starts with hex)
        has_header = first_line.startswith('commit_hash')

        if has_header:
            reader = csv.DictReader(f)
        else:
            # No header - provide fieldnames explicitly
            reader = csv.DictReader(f, fieldnames=CSV_COLUMNS)

        for row in reader:
            short_hash = row.get('short_hash', '')
            status = row.get('status', '')
            # Parse mm_cache_disabled (handle missing column for old results)
            mm_disabled_str = row.get('mm_cache_disabled', 'False')
            mm_disabled = mm_disabled_str.lower() == 'true'

            if short_hash and status:
                results[(short_hash, mm_disabled)] = status
    return results


def get_commit_status(commit: str, existing_results: dict[tuple[str, bool], str]) -> Optional[str]:
    """Check if commit has already been benchmarked with current mm_cache setting, return status or None"""
    short_hash = commit[:8]
    return existing_results.get((short_hash, DISABLE_MM_CACHE))


def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both file and stdout"""
    logger = logging.getLogger("bisect_verify")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    return logger


def get_current_commit() -> str:
    """Get current HEAD commit hash"""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def get_commit_info(commit_hash: str) -> dict:
    """Get commit metadata (timestamp, author, message)"""
    # Get timestamp
    result = subprocess.run(
        ["git", "show", "-s", "--format=%ci", commit_hash],
        capture_output=True, text=True, check=True
    )
    timestamp = result.stdout.strip()

    # Get author
    result = subprocess.run(
        ["git", "show", "-s", "--format=%an", commit_hash],
        capture_output=True, text=True, check=True
    )
    author = result.stdout.strip()

    # Get message (first line)
    result = subprocess.run(
        ["git", "show", "-s", "--format=%s", commit_hash],
        capture_output=True, text=True, check=True
    )
    message = result.stdout.strip()

    return {
        "timestamp": timestamp,
        "author": author,
        "message": message
    }


def get_ram_mb(server_process: Optional[subprocess.Popen] = None) -> float:
    """Get system RAM usage in MB for server process tree"""
    if server_process is None:
        return psutil.virtual_memory().used / (1024 * 1024)

    try:
        parent = psutil.Process(server_process.pid)
        total_rss = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def get_gpu_mem_mb() -> float:
    """Get GPU memory usage in MB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip().split('\n')[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 0.0


def check_for_memory_error(output: str) -> tuple[bool, Optional[str]]:
    """
    Check if output contains memory errors.
    Returns (is_memory_error, error_type)
    """
    output_lower = output.lower()
    for pattern in MEMORY_ERROR_PATTERNS:
        if pattern in output_lower:
            return True, "OOM"
    return False, None


def get_server_cmd() -> list[str]:
    """Build server command with optional mm cache disable flag"""
    cmd = BASE_SERVER_CMD.copy()
    if DISABLE_MM_CACHE:
        if USE_DEPRECATED_MM_FLAG:
            # Older boolean flag (deprecated in v0.12+)
            cmd.append("--disable-mm-preprocessor-cache")
        else:
            # Newer flag that takes a value (recommended)
            cmd.extend(["--mm-processor-cache-gb", "0"])
    return cmd


def start_server(log_file: Path, logger: logging.Logger) -> subprocess.Popen:
    """Start vLLM server, return process handle"""
    server_cmd = get_server_cmd()
    logger.info(f"Starting server with command: {' '.join(server_cmd)}")
    logger.info(f"MM processor cache disabled: {DISABLE_MM_CACHE}")

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"SERVER START: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(server_cmd)}\n")
        f.write(f"MM processor cache disabled: {DISABLE_MM_CACHE}\n")
        f.write(f"{'='*60}\n\n")

    # Open log file for appending server output
    log_handle = open(log_file, 'a')

    process = subprocess.Popen(
        server_cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # Create new process group for clean termination
    )

    return process


def wait_for_server_ready(logger: logging.Logger, server_process: subprocess.Popen,
                          log_file: Path, timeout: int = SERVER_STARTUP_TIMEOUT) -> tuple[bool, Optional[str]]:
    """
    Poll health endpoint until ready, process exits, or timeout.
    Returns (success, error_message).
    """
    logger.info(f"Waiting for server to be ready (timeout: {timeout}s)")
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if server process has exited (crashed)
        exit_code = server_process.poll()
        if exit_code is not None:
            # Process has exited - read log file for error details
            error_msg = f"Server process exited with code {exit_code}"
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                # Get last 50 lines for error context
                log_lines = log_content.strip().split('\n')
                last_lines = '\n'.join(log_lines[-50:])
                error_msg = f"{error_msg}\n\nLast 50 lines of log:\n{last_lines}"
            except Exception as e:
                error_msg = f"{error_msg} (failed to read log: {e})"

            logger.error(f"Server process crashed with exit code {exit_code}")
            return False, error_msg

        try:
            response = requests.get(SERVER_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                logger.info("Server is ready")
                return True, None
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    logger.error(f"Server failed to start within {timeout} seconds")
    return False, f"Server failed to start within {timeout} seconds"


def stop_server(process: subprocess.Popen, logger: logging.Logger) -> None:
    """Gracefully stop server, force kill if needed"""
    logger.info("Stopping server...")

    try:
        # Try graceful termination first
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=10)
            logger.info("Server stopped gracefully")
        except subprocess.TimeoutExpired:
            # Force kill
            logger.warning("Server didn't stop gracefully, force killing")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
    except (ProcessLookupError, OSError) as e:
        logger.warning(f"Error stopping server: {e}")

    # Wait for GPU memory to be released
    logger.info("Waiting for GPU memory to be released...")
    for _ in range(30):  # Wait up to 30 seconds
        gpu_mem = get_gpu_mem_mb()
        if gpu_mem < 500:  # Less than 500MB indicates memory released
            break
        time.sleep(1)


def get_benchmark_timeout() -> int:
    """Get benchmark timeout based on mm cache setting"""
    if DISABLE_MM_CACHE:
        return BENCHMARK_TIMEOUT_MM_CACHE_DISABLED
    return BENCHMARK_TIMEOUT


def run_benchmark(logger: logging.Logger, log_file: Path, run_number: int,
                  timeout: int = None) -> tuple[bool, str, Optional[str]]:
    if timeout is None:
        timeout = get_benchmark_timeout()
    """
    Run benchmark once.
    Returns (success, stdout+stderr, error_type if failed)
    """
    logger.info(f"Starting benchmark run {run_number}")

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"BENCHMARK RUN {run_number}: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(BENCHMARK_CMD)}\n")
        f.write(f"{'='*60}\n\n")

    process = None
    try:
        # Use Popen with process group so we can kill entire tree on timeout
        process = subprocess.Popen(
            BENCHMARK_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            logger.warning(f"Benchmark run {run_number} timed out, killing process group")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, stderr = process.communicate()  # Collect any remaining output

            output = f"Timeout after {timeout} seconds\nStdout: {stdout}\nStderr: {stderr}"

            with open(log_file, 'a') as f:
                f.write(f"\nTIMEOUT: {output}\n")

            logger.error(f"Benchmark run {run_number} timed out")
            return False, output, "timeout"

        output = stdout + "\n" + stderr

        # Write output to log
        with open(log_file, 'a') as f:
            f.write(output)
            f.write(f"\nReturn code: {process.returncode}\n")

        # Check for memory errors
        is_memory_error, error_type = check_for_memory_error(output)
        if is_memory_error:
            logger.error(f"Memory error detected in run {run_number}")
            return False, output, error_type

        if process.returncode != 0:
            logger.error(f"Benchmark run {run_number} failed with return code {process.returncode}")
            return False, output, "crash"

        logger.info(f"Benchmark run {run_number} completed successfully")
        return True, output, None

    except Exception as e:
        # Clean up process if something went wrong
        if process and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        raise


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
    start_time = time.time()
    short_hash = commit_hash[:8]
    log_file = LOG_DIR / f"{short_hash}.log"

    logger = setup_logging(log_file)
    logger.info(f"Starting verification for commit {commit_hash}")

    # Get commit info
    try:
        commit_info = get_commit_info(commit_hash)
    except subprocess.CalledProcessError:
        commit_info = {"timestamp": "", "author": "", "message": ""}

    # Initialize results
    results = {
        "commit_hash": commit_hash,
        "short_hash": short_hash,
        "timestamp": commit_info["timestamp"],
        "author": commit_info["author"],
        "message": commit_info["message"],
        "status": "bad",
        "error_type": None,
        "error_message": None,
        "successful_runs": 0,
        "mm_cache_disabled": DISABLE_MM_CACHE,
        "ram_idle_mb": None,
        "ram_run1_mb": None,
        "ram_run2_mb": None,
        "ram_run3_mb": None,
        "ram_run4_mb": None,
        "ram_run5_mb": None,
        "ram_run6_mb": None,
        "ram_run7_mb": None,
        "ram_run8_mb": None,
        "ram_run9_mb": None,
        "ram_run10_mb": None,
        "gpu_mem_idle_mb": None,
        "gpu_mem_run1_mb": None,
        "gpu_mem_run2_mb": None,
        "gpu_mem_run3_mb": None,
        "gpu_mem_run4_mb": None,
        "gpu_mem_run5_mb": None,
        "gpu_mem_run6_mb": None,
        "gpu_mem_run7_mb": None,
        "gpu_mem_run8_mb": None,
        "gpu_mem_run9_mb": None,
        "gpu_mem_run10_mb": None,
        "gpu_mem_peak_mb": 0.0,
        "total_duration_sec": 0.0,
        "log_file": str(log_file)
    }

    server_process = None

    try:
        # Start server
        server_process = start_server(log_file, logger)

        # Wait for server to be ready
        server_ready, startup_error = wait_for_server_ready(logger, server_process, log_file)
        if not server_ready:
            results["status"] = "bad"
            results["error_type"] = "server_startup"
            results["error_message"] = startup_error[:500] if startup_error else "Server failed to start"
            return results

        # Record idle memory
        time.sleep(5)  # Let server stabilize
        results["ram_idle_mb"] = get_ram_mb(server_process)
        results["gpu_mem_idle_mb"] = get_gpu_mem_mb()
        results["gpu_mem_peak_mb"] = results["gpu_mem_idle_mb"]

        logger.info(f"Idle memory - RAM: {results['ram_idle_mb']:.1f}MB, GPU: {results['gpu_mem_idle_mb']:.1f}MB")

        # Run benchmarks
        for run_num in range(1, REQUIRED_SUCCESSFUL_RUNS + 1):
            success, output, error_type = run_benchmark(logger, log_file, run_num)

            # Record memory after run
            ram_key = f"ram_run{run_num}_mb"
            gpu_key = f"gpu_mem_run{run_num}_mb"

            results[ram_key] = get_ram_mb(server_process)
            results[gpu_key] = get_gpu_mem_mb()
            results["gpu_mem_peak_mb"] = max(results["gpu_mem_peak_mb"], results[gpu_key])

            logger.info(f"After run {run_num} - RAM: {results[ram_key]:.1f}MB, GPU: {results[gpu_key]:.1f}MB")

            if not success:
                results["status"] = "bad"
                results["error_type"] = error_type
                results["error_message"] = output[:500] if output else None
                results["successful_runs"] = run_num - 1
                logger.error(f"Verification failed at run {run_num}")
                return results

            # Check if RAM exceeds threshold
            if results[ram_key] > RAM_THRESHOLD_MB:
                results["status"] = "bad"
                results["error_type"] = "OOM"
                results["error_message"] = f"RAM usage ({results[ram_key]:.1f}MB) exceeded threshold ({RAM_THRESHOLD_MB}MB) after run {run_num}"
                results["successful_runs"] = run_num - 1  # Don't count this run as successful
                logger.error(f"RAM threshold exceeded: {results[ram_key]:.1f}MB > {RAM_THRESHOLD_MB}MB")
                return results

            results["successful_runs"] = run_num

        # All runs succeeded
        results["status"] = "good"
        logger.info("All benchmark runs completed successfully")

    except Exception as e:
        logger.exception(f"Unexpected error during verification: {e}")
        results["status"] = "bad"
        results["error_type"] = "crash"
        results["error_message"] = str(e)[:500]

    finally:
        # Clean up
        if server_process:
            stop_server(server_process, logger)

        results["total_duration_sec"] = time.time() - start_time
        logger.info(f"Total verification time: {results['total_duration_sec']:.1f}s")

        # Write final status to log
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"FINAL STATUS: {results['status']}\n")
            f.write(f"Successful runs: {results['successful_runs']}\n")
            if results['error_type']:
                f.write(f"Error type: {results['error_type']}\n")
            if results['error_message']:
                f.write(f"Error message: {results['error_message']}\n")
            f.write(f"Total duration: {results['total_duration_sec']:.1f}s\n")
            f.write(f"{'='*60}\n")

    return results


def append_results_to_csv(results: dict) -> None:
    """Append results to CSV, creating file with headers if needed"""
    # Check if file exists AND has content (not just empty file)
    needs_header = not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0

    # Also check if existing file is missing header
    if not needs_header:
        with open(RESULTS_FILE, 'r') as f:
            first_line = f.readline().strip()
            if first_line and not first_line.startswith('commit_hash'):
                needs_header = False  # Has data but no header - don't add header mid-file

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        if needs_header:
            writer.writeheader()

        writer.writerow(results)


def test_commit(commit: str, existing_results: dict[tuple[str, bool], str]) -> str:
    """
    Test a single commit - returns 'good', 'bad', or 'skip'.
    Uses cached result if available, otherwise runs verification.
    """
    # Check if already benchmarked with current mm_cache setting
    cached_status = get_commit_status(commit, existing_results)
    if cached_status:
        print(f"  Using cached result (mm_cache_disabled={DISABLE_MM_CACHE}): {cached_status}")
        return cached_status

    # Checkout and test
    try:
        subprocess.run(["git", "checkout", commit], check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  Failed to checkout: {e.stderr}")
        return "skip"

    full_commit = get_current_commit()
    results = verify_commit(full_commit)
    append_results_to_csv(results)

    # Update our cache
    existing_results[(commit[:8], DISABLE_MM_CACHE)] = results["status"]

    print(f"  Result: status={results['status']}, successful_runs={results['successful_runs']}")
    return results["status"]


def run_all_commits():
    """Run verification on all commits in the commit list (linear)"""
    commit_list = load_commit_list(COMMITS_FILE)
    if not commit_list:
        print(f"No commits found in {COMMITS_FILE}")
        sys.exit(1)

    print(f"Found {len(commit_list)} commits to verify")
    LOG_DIR.mkdir(exist_ok=True)

    existing_results = load_existing_results()
    print(f"Loaded {len(existing_results)} existing results from {RESULTS_FILE}")

    original_commit = get_current_commit()

    for i, commit in enumerate(commit_list, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(commit_list)}] Verifying: {commit}")
        print(f"{'='*60}")

        status = test_commit(commit, existing_results)

        if status == "bad":
            print(f"\n*** Found bad commit: {commit} ***")

    # Return to original commit
    print(f"\nReturning to original commit: {original_commit[:9]}")
    subprocess.run(["git", "checkout", original_commit], capture_output=True)
    print(f"\nResults saved to {RESULTS_FILE}")


def run_bisect():
    """
    Run binary search to find the first bad commit.

    Assumes commit list is ordered from OLDEST to NEWEST (good -> bad).
    The goal is to find the first commit where the bug appears.
    """
    commit_list = load_commit_list(COMMITS_FILE)
    if not commit_list:
        print(f"No commits found in {COMMITS_FILE}")
        sys.exit(1)

    # Reverse if needed - we want oldest first (index 0 = oldest = likely good)
    # The file appears to have newest first, so reverse it
    commit_list = list(reversed(commit_list))

    print(f"Found {len(commit_list)} commits to bisect")
    print(f"  Oldest (likely good): {commit_list[0]}")
    print(f"  Newest (likely bad):  {commit_list[-1]}")

    LOG_DIR.mkdir(exist_ok=True)

    existing_results = load_existing_results()
    print(f"Loaded {len(existing_results)} existing results from {RESULTS_FILE}")

    original_commit = get_current_commit()

    # Binary search: find first bad commit
    # Invariant: all commits before 'left' are good, all commits at/after 'right' are bad
    left = 0
    right = len(commit_list)

    # First, verify boundary conditions if not cached
    print(f"\n{'='*60}")
    print("Verifying boundary: oldest commit (should be good)")
    print(f"{'='*60}")
    oldest_status = test_commit(commit_list[0], existing_results)
    if oldest_status == "bad":
        print(f"\n*** Oldest commit is already bad! Bug predates this range. ***")
        print(f"First bad commit: {commit_list[0]}")
        subprocess.run(["git", "checkout", original_commit], capture_output=True)
        return

    print(f"\n{'='*60}")
    print("Verifying boundary: newest commit (should be bad)")
    print(f"{'='*60}")
    newest_status = test_commit(commit_list[-1], existing_results)
    if newest_status == "good":
        print(f"\n*** Newest commit is good! Bug not in this range. ***")
        subprocess.run(["git", "checkout", original_commit], capture_output=True)
        return

    # Binary search
    left = 0  # Known good
    right = len(commit_list) - 1  # Known bad

    iteration = 0
    while left + 1 < right:
        iteration += 1
        mid = (left + right) // 2
        remaining = right - left - 1

        print(f"\n{'='*60}")
        print(f"Bisect iteration {iteration}: testing index {mid}/{len(commit_list)-1}")
        print(f"  Range: [{left}..{right}], {remaining} commits remaining to search")
        print(f"  Testing: {commit_list[mid]}")
        print(f"{'='*60}")

        status = test_commit(commit_list[mid], existing_results)

        if status == "good":
            left = mid
            print(f"  -> Commit is good, searching newer commits")
        elif status == "bad":
            right = mid
            print(f"  -> Commit is bad, searching older commits")
        else:  # skip
            # Can't determine, try adjacent commit
            print(f"  -> Commit skipped, trying next commit")
            # Move towards right (newer) as a heuristic
            found_testable = False
            for offset in range(1, right - mid):
                if mid + offset < right:
                    test_idx = mid + offset
                    print(f"  Trying offset +{offset}: {commit_list[test_idx]}")
                    alt_status = test_commit(commit_list[test_idx], existing_results)
                    if alt_status == "good":
                        left = test_idx
                        found_testable = True
                        break
                    elif alt_status == "bad":
                        right = test_idx
                        found_testable = True
                        break
            if not found_testable:
                print(f"  Could not find testable commit in range, narrowing from left")
                left = mid

    # Found the boundary
    first_bad = commit_list[right]
    last_good = commit_list[left]

    print(f"\n{'='*60}")
    print("BISECT COMPLETE")
    print(f"{'='*60}")
    print(f"Last good commit:  {last_good}")
    print(f"First bad commit:  {first_bad}")
    print(f"Iterations: {iteration}")
    print(f"\nThe bug was likely introduced in: {first_bad}")

    # Return to original commit
    print(f"\nReturning to original commit: {original_commit[:9]}")
    subprocess.run(["git", "checkout", original_commit], capture_output=True)
    print(f"\nResults saved to {RESULTS_FILE}")


def main():
    global DISABLE_MM_CACHE, USE_DEPRECATED_MM_FLAG, REQUIRED_SUCCESSFUL_RUNS

    parser = argparse.ArgumentParser(description="vLLM bisect verification")
    parser.add_argument("--commit", help="Specific commit to test (default: HEAD)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run verification on all commits in target_commits.csv (linear)")
    parser.add_argument("--bisect", action="store_true",
                        help="Run binary search to find first bad commit in target_commits.csv")
    parser.add_argument("--skip-unlisted", action="store_true",
                        help="Skip (exit 125) if commit not in target_commits.csv (for git bisect)")
    parser.add_argument("--disable-mm-cache", action="store_true",
                        help="Disable multimodal processor cache (--mm-processor-cache-gb 0)")
    parser.add_argument("--disable-mm-preprocessor-cache", action="store_true",
                        help="Disable multimodal processor cache using deprecated flag (--disable-mm-preprocessor-cache)")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_REQUIRED_RUNS,
                        help=f"Number of benchmark runs required (default: {DEFAULT_REQUIRED_RUNS})")
    args = parser.parse_args()

    # Set global flags for mm cache
    DISABLE_MM_CACHE = args.disable_mm_cache or args.disable_mm_preprocessor_cache
    USE_DEPRECATED_MM_FLAG = args.disable_mm_preprocessor_cache
    REQUIRED_SUCCESSFUL_RUNS = args.num_runs

    LOG_DIR.mkdir(exist_ok=True)

    # Binary search mode
    if args.bisect:
        run_bisect()
        return

    # Run all commits mode (linear)
    if args.run_all:
        run_all_commits()
        return

    commit = args.commit or get_current_commit()

    # Check if commit should be skipped (for git bisect with filtered list)
    if args.skip_unlisted:
        commit_list = load_commit_list(COMMITS_FILE)
        if commit_list and not is_commit_in_list(commit, commit_list):
            print(f"Skipping commit {commit[:9]} (not in filtered list)")
            sys.exit(125)

    print(f"Verifying commit: {commit}")

    results = verify_commit(commit)
    append_results_to_csv(results)

    print(f"Results: status={results['status']}, successful_runs={results['successful_runs']}")

    # Exit codes for git bisect
    if results["status"] == "good":
        sys.exit(0)
    elif results["status"] == "skip":
        sys.exit(125)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
