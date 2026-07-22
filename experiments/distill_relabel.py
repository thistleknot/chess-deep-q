import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Deeper-teacher single-variable test for :Search-teacher: (spec/search-distill.spec.md).

Reuses the EXACT cached position set (models/distillA_labels.npz fens — same states, no generation
variance) and RE-labels them at a deeper search depth, then refits the amap-897 linear. Isolates one
variable: does a stronger minimax TEACHER (deeper labels) yield a stronger distilled eval? Emits a
champion-format ckpt for the native-d9 ladder / native duel.

Usage: python distill_relabel.py <label_depth> [out=models/champion_distill_d<depth>.pt] [threads=6] [ridge=100]
"""
import os
import sys

import numpy as np
import torch

from chessdq.corpus_gen import raw_weights
from experiments.distill_linear import label, featurize, fit_ridge, save_ckpt

CACHE = os.environ.get("DISTILL_CACHE", "models/distillA_labels.npz")
SRC   = os.environ.get("DISTILL_SRC", "models/champion.pt")   # teacher eval whose d-search labels


def main():
    ld      = int(sys.argv[1])
    out     = sys.argv[2] if len(sys.argv) > 2 else f"models/champion_distill_d{ld}.pt"
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    ridge   = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0

    z = np.load(CACHE, allow_pickle=True)
    fens = [str(f) for f in z["fens"]]
    print(f"relabeling {len(fens)} cached positions at d{ld} (teacher={SRC})...", flush=True)
    w, b = raw_weights(SRC)
    y = label(fens, w, b, ld, threads)
    X = featurize(fens)
    w_new, b_new, rms = fit_ridge(X, y, ridge)
    save_ckpt(w_new, b_new, out, torch.load(SRC, map_location="cpu"))
    print(f"saved {out}  fit atanh-RMS {rms:.3f}  ||w||={np.linalg.norm(w_new):.3f}", flush=True)
    print(f"LADDER: python experiments/anchor_ladder.py {out} 9 20 {os.path.basename(out)[:-3]}", flush=True)


if __name__ == "__main__":
    main()
