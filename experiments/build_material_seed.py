import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Merge 11 fallback rung — MATERIAL-ONLY seed (the donor paper's own from-scratch start).

KnightCap's main experiment (cs/9901002) initialized ONLY material to standard computer
values (P=1, N=4, B=4, R=6, Q=12; everything else zero) and climbed 1650->2150. This seed is
that start in our feature space — still zero trained/distilled weights — as a CONTROLLED
variant of the zero seed: identical noise + mover-optimism, the single changed variable is
material knowledge (piece-plane weights = piece value, White +v / Black −v, kings 0).

Preconditions: models/zca.npz. Failure modes: none silent — asserts check mover-optimism at
the start position (material-balanced => V = ±c) and the whitening round-trip.

Usage: python build_material_seed.py [out=models/qlearn_mat_seed.pt] [c=0.25]
"""
import sys

import numpy as np
import torch

NFEAT, STM = 809, 768
PAWN = 0.0141                                   # champion pawn unit (build_kc7_seed lineage)
VALS = [1.0, 4.0, 4.0, 6.0, 12.0, 0.0]          # KnightCap computer values: P N B R Q K

out = sys.argv[1] if len(sys.argv) > 1 else "models/qlearn_mat_seed.pt"
c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

z = np.load("models/zca.npz")
Z, mu = z["Z"].astype(np.float64), z["mu"].astype(np.float64)

rng = np.random.default_rng(11)                 # same noise stream as the zero seed (control)
w = rng.normal(0.0, 1e-3, NFEAT)
for side, sign in ((0, +1.0), (1, -1.0)):       # planes: White P..K then Black P..K
    for pt, v in enumerate(VALS):
        w[(side * 6 + pt) * 64:(side * 6 + pt) * 64 + 64] += sign * v * PAWN
w[STM] += 2.0 * c                               # mover optimism, identical to the zero seed
b = -c

wp = np.linalg.solve(Z, w)
bp = b + float(w @ mu)

import chess
from chessdq.cem_loop import encode_features
x0 = encode_features(chess.Board()).astype(np.float64)          # startpos, White to move
raw = float(w @ x0) + b
whitened = float(wp @ (Z @ (x0 - mu))) + bp
assert abs(raw - c) < 2e-2, ("startpos should be ±c (material balanced)", raw)
assert abs(raw - whitened) < 1e-6, (raw, whitened)

sd = {"head.weight": torch.tensor(wp, dtype=torch.float32).reshape(1, NFEAT),
      "head.bias": torch.tensor([bp], dtype=torch.float32)}
torch.save({"state_dict": sd, "arch": "linear", "enc": "kc", "zca": True, "cum_games": 0}, out)
print(f"{out}: material-only seed (P={PAWN}, computer values, optimism c={c}) — verified")
