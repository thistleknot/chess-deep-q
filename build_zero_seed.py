"""Merge 11 :Zero-seed: — optimistic zero-knowledge seed in whitened space.

Purpose: the from-scratch starting point for the pathfinding population — NO chess knowledge
(no donor coefficients, no material values), only symmetry-breaking noise plus side-to-move
optimism (S&B §2.6 made zero-sum-symmetric: w_stm=2c, b=−c ⇒ the mover always believes +c,
so both sides press instead of shuffling; a global positive bias would be White-optimism/
Black-pessimism under a White-absolute V).

Preconditions: models/zca.npz exists (Z symmetric, mu). Failure mode: none silent — asserts
verify the mover-optimism identity and the whitening round-trip before saving.

Usage: python build_zero_seed.py [out=models/qlearn_zero_seed.pt] [c=0.25]
"""
import sys

import numpy as np
import torch

NFEAT, STM = 809, 768                      # stm plane index (769 raw planes: 0..767 pieces, 768 stm)

out = sys.argv[1] if len(sys.argv) > 1 else "models/qlearn_zero_seed.pt"
c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

z = np.load("models/zca.npz")
Z, mu = z["Z"].astype(np.float64), z["mu"].astype(np.float64)

rng = np.random.default_rng(11)
w = rng.normal(0.0, 1e-3, NFEAT)           # symmetry-breaking noise, zero knowledge
w[STM] += 2.0 * c                          # mover optimism: V=+c white-to-move, −c black-to-move
b = -c

# raw -> whitened (Z symmetric): w'·(Z(x−mu)) + b' == w·x + b
wp = np.linalg.solve(Z, w)
bp = b + float(w @ mu)

x_white = np.zeros(NFEAT); x_white[STM] = 1.0          # empty-board proxy: only stm set
x_black = np.zeros(NFEAT)
for x, want in ((x_white, +c), (x_black, -c)):
    raw = float(w @ x) + b
    whitened = float(wp @ (Z @ (x - mu))) + bp
    assert abs(raw - want) < 1e-2, (raw, want)
    assert abs(raw - whitened) < 1e-6, (raw, whitened)

sd = {"head.weight": torch.tensor(wp, dtype=torch.float32).reshape(1, NFEAT),
      "head.bias": torch.tensor([bp], dtype=torch.float32)}
torch.save({"state_dict": sd, "arch": "linear", "enc": "kc", "zca": True, "cum_games": 0}, out)
print(f"{out}: optimistic zero seed (c={c}) — mover sees +{c}, whitening verified")
