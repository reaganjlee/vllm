#!/usr/bin/env python3
"""Analyze bisect results sorted by commit order."""

import pandas as pd
from pathlib import Path
import csv

# Commit range to analyze (set to None to use all)
RANGE_START = "ad430a67c"  # Newer commit
RANGE_END = "d100d78eb"    # Older commit

# Load commit order from target_commits.csv
commits_file = Path("target_commits.csv")
commit_order = []
commit_messages = {}

with open(commits_file) as f:
    first_line = f.readline().strip()
    f.seek(0)

    if first_line.startswith('commit_hash'):
        reader = csv.DictReader(f)
        for row in reader:
            commit_hash = row.get('commit_hash', '').strip()
            message = row.get('message', '').strip()
            if commit_hash:
                commit_order.append(commit_hash)
                commit_messages[commit_hash] = message
    else:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                short_hash = parts[0]
                message = parts[1] if len(parts) > 1 else ""
                commit_order.append(short_hash)
                commit_messages[short_hash] = message

# Filter to specified range
if RANGE_START or RANGE_END:
    start_idx = 0
    end_idx = len(commit_order)

    for i, commit in enumerate(commit_order):
        if RANGE_START and commit.startswith(RANGE_START):
            start_idx = i
        if RANGE_END and commit.startswith(RANGE_END):
            end_idx = i + 1

    commit_order = commit_order[start_idx:end_idx]
    # Update messages dict to use short hashes for matching
    new_messages = {}
    for commit in commit_order:
        new_messages[commit[:8]] = commit_messages.get(commit, '')
    commit_messages = new_messages

# Load results
df = pd.read_csv("bisect_results.csv")

# Filter out server_startup errors (these aren't meaningful for memory leak detection)
df = df[df['error_type'] != 'server_startup']

# Drop duplicates, keeping the last result for each commit
df = df.drop_duplicates(subset='short_hash', keep='last')

# Create ordering index based on python_only_commits.py
# Match using first 8 chars since commit_order has 9-char hashes
def get_order(short_hash):
    for i, commit in enumerate(commit_order):
        if commit.startswith(short_hash) or short_hash.startswith(commit[:8]):
            return i
    return len(commit_order)

df['order'] = df['short_hash'].apply(get_order)

# Sort by commit order (ascending = newest first, as listed in file)
df = df.sort_values('order')

# Select columns for display
display_cols = [
    'short_hash', 'status', 'error_type', 'successful_runs',
    'ram_idle_mb', 'ram_run1_mb', 'ram_run2_mb', 'ram_run3_mb',
    'ram_run4_mb', 'ram_run5_mb', 'ram_run6_mb'
]

# Filter to existing columns
display_cols = [c for c in display_cols if c in df.columns]

print("=" * 80)
print("BISECT RESULTS (ordered newest -> oldest)")
print("(excluding server_startup errors)")
print("=" * 80)
print()

# Show all commits from the filtered range, marking tested ones
print(f"Commits in range ({RANGE_START} to {RANGE_END}):")
print("-" * 80)
for i, commit in enumerate(commit_order):
    short_hash = commit[:8]

    # Check if we have results for this commit
    match = df[df['short_hash'] == short_hash]

    if len(match) > 0:
        row = match.iloc[0]
        status = row['status']
        error_type = row.get('error_type', '') or '-'
        successful_runs = int(row.get('successful_runs', 0))

        if status == 'good':
            indicator = 'GOOD'
        elif status == 'bad':
            indicator = 'BAD '
        else:
            indicator = 'SKIP'

        runs_str = f"runs={successful_runs}/6"
        error_str = f"{error_type:<10}"
    else:
        indicator = '    '
        runs_str = "          "
        error_str = "          "

    message = commit_messages.get(short_hash, '')[:50]
    print(f"{i+1:2}. [{indicator}] {short_hash} | {runs_str} | {error_str} | {message}")

# Show any tested commits that are OUTSIDE the current range
print()
print("-" * 80)
print("Other tested commits (outside current range):")
print("-" * 80)

range_short_hashes = set(c[:8] for c in commit_order)
outside_range = df[~df['short_hash'].isin(range_short_hashes)]

if len(outside_range) > 0:
    for _, row in outside_range.iterrows():
        short_hash = row['short_hash']
        status = row['status']
        error_type = row.get('error_type', '') or '-'
        successful_runs = int(row.get('successful_runs', 0))
        message = row.get('message', '')[:50]

        if status == 'good':
            indicator = 'GOOD'
        elif status == 'bad':
            indicator = 'BAD '
        else:
            indicator = 'SKIP'

        print(f"    [{indicator}] {short_hash} | runs={successful_runs}/6 | {error_type:<10} | {message}")
else:
    print("    (none)")

print()
print("=" * 80)
print("MEMORY PROGRESSION (for commits that ran benchmarks)")
print("=" * 80)
print()

# Show memory progression for commits that actually ran
df_with_runs = df[df['successful_runs'] > 0].copy()

if not df_with_runs.empty:
    ram_cols = ['ram_idle_mb', 'ram_run1_mb', 'ram_run2_mb', 'ram_run3_mb',
                'ram_run4_mb', 'ram_run5_mb', 'ram_run6_mb']
    ram_cols = [c for c in ram_cols if c in df.columns]

    for _, row in df_with_runs.iterrows():
        short_hash = row['short_hash']
        runs = int(row['successful_runs'])
        print(f"\n{short_hash} ({runs} successful runs):")

        # Show RAM progression
        ram_values = []
        for col in ram_cols:
            val = row.get(col)
            if pd.notna(val):
                ram_values.append(f"{val/1024:.1f}GB")
            else:
                ram_values.append("-")

        print(f"  RAM: idle -> run1 -> run2 -> run3 -> run4 -> run5 -> run6")
        print(f"       {' -> '.join(ram_values)}")

        # Calculate RAM growth
        idle = row.get('ram_idle_mb')
        last_run_col = f'ram_run{runs}_mb'
        last_run = row.get(last_run_col)
        if pd.notna(idle) and pd.notna(last_run):
            growth = last_run - idle
            print(f"  Growth: +{growth/1024:.1f}GB over {runs} runs ({growth/runs/1024:.2f}GB per run)")
else:
    print("No commits completed any benchmark runs.")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

total = len(df)
good = len(df[df['status'] == 'good'])
bad = len(df[df['status'] == 'bad'])
skip = len(df[df['status'] == 'skip'])

print(f"Total commits tested: {total}/{len(commit_order)} (excluding server_startup errors)")
print(f"  Good: {good}")
print(f"  Bad:  {bad}")
print(f"  Skip: {skip}")

# Find transition point (going through commits newest to oldest)
last_good = None
first_bad = None

for commit in commit_order:
    short_hash = commit[:8]
    match = df[df['short_hash'] == short_hash]
    if len(match) > 0:
        status = match.iloc[0]['status']
        if status == 'good':
            last_good = commit
        elif status == 'bad' and first_bad is None:
            first_bad = commit

if first_bad:
    print(f"\nFirst bad commit (newest->oldest): {first_bad}")
    if first_bad in commit_messages:
        print(f"  Message: {commit_messages[first_bad]}")
if last_good:
    print(f"Last good commit: {last_good}")
    if last_good in commit_messages:
        print(f"  Message: {commit_messages[last_good]}")
