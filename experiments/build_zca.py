import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""E3 prep — ZCA whitening transform for the 809-feature space (spec: intervention-queue
purist capacity #1). Corpus from native parallel self-play; Z = (Sigma + eps I)^(-1/2), ZCA
form (symmetric, decorrelates while staying close to the original axes).

Outputs models/zca.npz (Z, mu) and a WHITENED-SPACE seed lineage: for a raw-space seed w,b
the equivalent whitened weights are w' = Z^{-1} w, b' = b + w·mu  (V = w'·Z(x-mu) + b').

Usage: python build_zca.py <seed_lineage_in> <lineage_out> [games]
"""
import sys

import numpy as np
import torch
import chess

from chessdq.cem_loop import encode_features
from rsearch2 import play_games

SEED_IN = sys.argv[1] if len(sys.argv) > 1 else "kc7f"
LIN_OUT = sys.argv[2] if len(sys.argv) > 2 else "kc7g"
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 800


def main():
    ck = torch.load(f"models/qlearn_{SEED_IN}_best.pt", map_location="cpu")
    w = ck["state_dict"]["head.weight"].reshape(-1).double().numpy()
    b = float(ck["state_dict"]["head.bias"].reshape(-1)[0])

    print(f"corpus: {GAMES} native d1 self-play games (eps .3 for coverage)...", flush=True)
    games = play_games(list(w), b, list(w), b, 1, GAMES, 12, 1, 0.3, 160, 7)
    X = np.stack([encode_features(chess.Board(f))
                  for _z, _aw, recs in games for f, _v, _p in recs]).astype(np.float64)
    print(f"positions: {X.shape[0]}", flush=True)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / len(Xc)
    eps = 1e-3
    evals, evecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0]))
    Z = evecs @ np.diag(evals ** -0.5) @ evecs.T            # symmetric ZCA
    np.savez("models/zca.npz", Z=Z, mu=mu)
    cond_before = np.linalg.cond(cov + eps * np.eye(cov.shape[0]))
    print(f"saved models/zca.npz | covariance condition number {cond_before:.1e} -> ~1", flush=True)

    # whitened-space seed: identical V as the raw seed at every position
    w_prime = np.linalg.solve(Z, w)
    b_prime = b + float(w @ mu)
    sd = {"head.weight": torch.tensor(w_prime, dtype=torch.float32).reshape(1, -1),
          "head.bias": torch.tensor([b_prime], dtype=torch.float32)}
    out = {"state_dict": sd, "arch": "linear", "enc": "kc", "zca": True,
           "cum_games": int(ck.get("cum_games", 4200))}
    for p in (f"models/qlearn_{LIN_OUT}.pt", f"models/qlearn_{LIN_OUT}_best.pt"):
        torch.save(out, p)
    # equivalence check on one position
    x = encode_features(chess.Board()).astype(np.float64)
    v_raw = float(np.tanh(w @ x + b))
    v_wht = float(np.tanh(w_prime @ (Z @ (x - mu)) + b_prime))
    print(f"seed -> {LIN_OUT} (whitened space) | equivalence: raw {v_raw:+.6f} vs wht {v_wht:+.6f}")
    assert abs(v_raw - v_wht) < 1e-6, "whitened seed not equivalent"


if __name__ == "__main__":
    main()
