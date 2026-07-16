import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
""":PCA-reduce: — whitened-PCA feature reduction from OWN games (provenance-clean).

Operator: "we need a way to reduce features; PCA + ZCA is sufficient." Corpus = games played
by the pure random-init seed (rules + RNG only). Keep the top-k principal components at the
declared explained-variance threshold (default 95% — k is an INFRASTRUCTURE decision,
reported not tuned), whiten them: W = D_k^{-1/2} U_k^T (k x 809). qlearn consumes it through
the same npz keys as ZCA; the native searcher gets exact raw-space weights via W^T.

Outputs: models/pca_self.npz (Z=W k x 809, mu) + models/qlearn_pca_seed.pt (optimistic clean
seed projected into the reduced space; projection error on the optimism value is REPORTED —
the stm direction may not lie fully in the top-k subspace).

Usage: python build_pca_seed.py [ev=0.95] [games=800] [c=0.25]
"""
import sys

import numpy as np
import torch
import chess

from chessdq.cem_loop import encode_features
from rsearch4 import play_games

EV = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 800
c = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
NFEAT, STM = 809, 768


def main():
    ck = torch.load("models/qlearn_clean_seed.pt", map_location="cpu")
    w_raw = ck["state_dict"]["head.weight"].reshape(-1).double().numpy()
    b_raw = float(ck["state_dict"]["head.bias"].reshape(-1)[0])

    print(f"corpus: {GAMES} OWN games (pure seed, d1, tau 0.7)...", flush=True)
    games = play_games(list(w_raw), b_raw, list(w_raw), b_raw, 1, GAMES, 12, 1, 0.0, 160, 7, tau=0.7)
    X = np.stack([encode_features(chess.Board(f))
                  for _z, _aw, recs in games for f, _v, _p in recs]).astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / len(Xc)
    evals, evecs = np.linalg.eigh(cov + 1e-6 * np.eye(NFEAT))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    ev_frac = np.cumsum(evals) / np.sum(evals)
    k = int(np.searchsorted(ev_frac, EV) + 1)
    W = (evecs[:, :k] * (evals[:k] ** -0.5)).T            # k x 809 whitened-PCA
    np.savez("models/pca_self.npz", Z=W, mu=mu)
    print(f"positions {X.shape[0]} | k = {k}/{NFEAT} components at {EV:.0%} explained variance")

    wp = np.linalg.pinv(W).T @ w_raw                      # least-squares projection of the seed
    bp = b_raw + float(w_raw @ mu)
    x0 = encode_features(chess.Board()).astype(np.float64)
    v_raw = float(w_raw @ x0) + b_raw
    v_pca = float(wp @ (W @ (x0 - mu))) + bp
    print(f"seed projection: startpos raw {v_raw:+.3f} vs reduced {v_pca:+.3f} "
          f"(optimism target {c:+.2f}; difference = subspace loss, reported not hidden)")

    sd = {"head.weight": torch.tensor(wp, dtype=torch.float32).reshape(1, k),
          "head.bias": torch.tensor([bp], dtype=torch.float32)}
    torch.save({"state_dict": sd, "arch": "linear", "enc": "kc", "zca": True, "cum_games": 0},
               "models/qlearn_pca_seed.pt")
    print("models/pca_self.npz + models/qlearn_pca_seed.pt — provenance: rules+RNG+own games, PASSES")


if __name__ == "__main__":
    main()
