"""Multiprocessing distillation-label launcher (Stage-1 :Process-separated-labeling:).

Spec: teacher-distillation.spec.md. Spawns N labeler_sf.py worker PROCESSES (separate OS
processes, never threads sharing the trainer GIL), each writing its own shard for T
seconds, then MERGES the shards into the :Cumulative-dataset: (data/distill_sf.jsonl) with
exact-FEN dedup. The merge is a cumulative append that never regresses: existing rows are
retained, only FENs not already present are added.

This replaces distill_sf.py's retired in-process threaded labellers (the measured 58
labels/s GIL violation). Distillation TRAINING still reads the finished corpus separately.

Usage:
    python gen_labels.py [seconds] [depth] [n_workers] [out_path]
Defaults: 300s, depth 8, 4 workers, data/distill_sf.jsonl.
"""

import os
import sys
import json
import time
import glob
import subprocess

DATA = "data/distill_sf.jsonl"
SHARD_DIR = "data"


def row_fen(obj):
    """Extract the FEN from either the new dict schema or a legacy [fen, ...] list row."""
    if isinstance(obj, dict):
        return obj.get("fen")
    if isinstance(obj, list) and obj:
        return obj[0]
    return None


def load_existing_fens(path):
    """Set of FENs already in the cumulative dataset (tolerant of legacy + dict schemas)."""
    fens = set()
    if not os.path.exists(path):
        return fens
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fen = row_fen(json.loads(line))
            if fen is not None:
                fens.add(fen)
    return fens


def merge_shards(shard_paths, data_path):
    """Append novel rows from shards into data_path (exact-FEN dedup, cumulative).

    Returns (generated, added): total rows produced across shards, and rows actually
    appended after dedup against the existing dataset and within-run duplicates.
    """
    seen = load_existing_fens(data_path)
    generated = 0
    added = 0
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    with open(data_path, "a") as out:
        for sp in shard_paths:
            if not os.path.exists(sp):
                continue
            with open(sp) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    generated += 1
                    fen = row_fen(obj)
                    if fen is None or fen in seen:
                        continue
                    seen.add(fen)
                    out.write(json.dumps(obj) + "\n")
                    added += 1
    return generated, added


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    data_path = sys.argv[4] if len(sys.argv) > 4 else DATA

    os.makedirs(SHARD_DIR, exist_ok=True)
    stamp = int(time.time())
    shard_paths = [os.path.join(SHARD_DIR, f"_shard_{stamp}_{i}.jsonl") for i in range(n_workers)]

    print(f"Launching {n_workers} labeler processes, depth {depth}, {seconds:.0f}s each...")
    start = time.time()
    procs = []
    for i, sp in enumerate(shard_paths):
        # Fresh shard per run; each worker seeds a distinct RNG (seed = stamp+i).
        if os.path.exists(sp):
            os.remove(sp)
        p = subprocess.Popen(
            [sys.executable, "labeler_sf.py", sp, str(seconds), str(depth), str(stamp + i)]
        )
        procs.append(p)
    for p in procs:
        p.wait()
    elapsed = time.time() - start

    generated, added = merge_shards(shard_paths, data_path)
    for sp in shard_paths:
        if os.path.exists(sp):
            os.remove(sp)

    total = sum(1 for _ in open(data_path)) if os.path.exists(data_path) else 0
    print(
        f"Generated {generated} labels in {elapsed:.0f}s "
        f"({generated/elapsed:.1f} labels/s across {n_workers} workers). "
        f"Added {added} new (dedup). Cumulative dataset: {total} rows."
    )


if __name__ == "__main__":
    main()
