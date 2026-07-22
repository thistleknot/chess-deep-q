import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Autonomous expert-iteration climber for :Search-teacher: (spec/search-distill.spec.md).

Each round: relabel the cached position set with the PREVIOUS round's eval + d-minimax search
(the teacher strictly exceeds the static eval, so the eval absorbs ~one lookahead/round), refit the
amap-897 linear, then native-d9 anchor-ladder it vs Stockfish for a deployment-truth MLE rating.
Continues while the point estimate climbs; STOPS at the search-consistent plateau (PATIENCE rounds
with no new best) or MAX_ROUNDS. Ladder is the truth ruler (BARE ckpt native d9 — NOT enc: d2-beam).

Empirical climb so far: champion 1878 -> distillA2 1917 -> distillA3 2035 (iteration compounds).

Usage: python distill_climb.py <seed_ckpt> <start_round> [max_rounds=8] [label_depth=5]
                               [ladder_games=20] [threads=4]
Env DISTILL_CACHE (position set). Writes models/champion_distillA<k>.pt + appends round lines here.
"""
import os
import re
import sys
import subprocess

import numpy as np
import torch

from chessdq.corpus_gen import raw_weights
from experiments.distill_linear import label, featurize, fit_ridge, save_ckpt

CACHE = os.environ.get("DISTILL_CACHE", "models/distillA_labels.npz")
_MLE = re.compile(r"MLE rating\s+(-?\d+)\s+\(95%\s+(-?\d+)\.\.(-?\d+)")
PATIENCE = 2


def relabel(prev_ckpt, out_ckpt, ld, threads):
    z = np.load(CACHE, allow_pickle=True)
    fens = [str(f) for f in z["fens"]]
    w, b = raw_weights(prev_ckpt)
    y = label(fens, w, b, ld, threads)
    X = featurize(fens)
    w_new, b_new, rms = fit_ridge(X, y, 100.0)
    save_ckpt(w_new, b_new, out_ckpt, torch.load(prev_ckpt, map_location="cpu"))
    return rms


def ladder(ckpt, games, name):
    out = subprocess.run(
        [sys.executable, "experiments/anchor_ladder.py", ckpt, "9", str(games), name],
        capture_output=True, text=True, timeout=5400)
    line = next((l for l in out.stdout.splitlines() if "MLE rating" in l), "")
    m = _MLE.search(line)
    return (int(m.group(1)) if m else None), line.strip()


def main():
    seed        = sys.argv[1]
    k           = int(sys.argv[2])
    max_rounds  = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ld          = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    lg          = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    threads     = int(sys.argv[6]) if len(sys.argv) > 6 else 4

    prev = seed
    best_r, best_ckpt, stale = None, seed, 0
    while k <= max_rounds:
        out = f"models/champion_distillA{k}.pt"
        print(f"\n=== round {k}: relabel from {os.path.basename(prev)} (d{ld}) ===", flush=True)
        rms = relabel(prev, out, ld, threads)
        r, line = ladder(out, lg, f"distillA{k}")
        print(f"round {k}: fit-RMS {rms:.3f} | {line}", flush=True)
        if r is None:
            print(f"round {k}: NO MLE PARSED -> stop", flush=True)
            break
        if best_r is None or r > best_r:
            best_r, best_ckpt, stale = r, out, 0
        else:
            stale += 1
            print(f"round {k}: no new best ({r} <= {best_r}); stale {stale}/{PATIENCE}", flush=True)
            if stale >= PATIENCE:
                print(f"\nPLATEAU at round {k}. Best = {os.path.basename(best_ckpt)} @ {best_r}", flush=True)
                break
        prev = out
        k += 1
    print(f"\nCLIMB_BEST_CKPT={best_ckpt}", flush=True)
    print(f"CLIMB_BEST_MLE={best_r}", flush=True)


if __name__ == "__main__":
    main()
