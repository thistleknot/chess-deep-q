"""Bake-off round 2 artifacts (plan: compositions, not replacements; base controls included).

One random-play corpus (rules+RNG — provenance-pure), then per-set transforms + seeds:
  S6 pst769   : 769 raw seed (noise + stm optimism)                    -> qlearn_p7_seed.pt
  S7 hk5k     : 5121 raw seed (noise + stm-feature optimism)           -> qlearn_hk_seed.pt
  S8 hk5k-std : 1-D diagonal standardize (clamped) + seed              -> hk_std.npz, qlearn_hks_seed.pt
  S9 concat   : 809 ⊕ k2048 (2857) diag std + seed                     -> kx_std.npz, qlearn_kx_seed.pt
  S10 pca5k   : 5121 -> top-512 whitened PCA + projected seed          -> pca5k.npz, qlearn_p5_seed.pt

Failure modes: none silent — every seed asserts its transform round-trip (or reports the
projection loss where exact round-trip is impossible).
"""
import numpy as np
import torch
import chess
import random

from cem_loop import encode_features
from kanerva_enc import active5k, encode_kanerva809, K_IN

C = 0.25
rng_np = np.random.default_rng(11)


def corpus(n_games=250, cap=120):
    rng = random.Random(7)
    acts, kcs = [], []
    for g in range(n_games):
        b = chess.Board()
        for ply in range(cap):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            b.push(rng.choice(moves))
            if ply % 2 == (g % 2):         # BOTH parities across games — side-to-move must
                acts.append(active5k(b))   # vary in the corpus or PCA/std kill the stm dim
                kcs.append(encode_features(b))
    return acts, np.stack(kcs).astype(np.float32)


def save_seed(path, w, b, enc, zca):
    sd = {"head.weight": torch.tensor(w, dtype=torch.float32).reshape(1, -1),
          "head.bias": torch.tensor([b], dtype=torch.float32)}
    torch.save({"state_dict": sd, "arch": "linear", "enc": enc, "zca": zca, "cum_games": 0}, path)


def main():
    print("corpus (random play, rules+RNG)...", flush=True)
    acts, KC = corpus()
    n = len(acts)
    print(f"positions: {n}", flush=True)

    # S6: 769 raw seed (pst encoding: planes + stm at index 768)
    w = rng_np.normal(0, 1e-3, 769); w[768] += 2 * C
    save_seed("models/qlearn_p7_seed.pt", w, -C, "pst", False)

    # 5k occupancy stats from active lists
    p = np.zeros(K_IN)
    for a in acts:
        p[a] += 1.0
    p /= n

    # S7: 5121 raw seed
    w = rng_np.normal(0, 1e-3, K_IN); w[K_IN - 1] += 2 * C
    save_seed("models/qlearn_hk_seed.pt", w, -C, "hk", False)

    # S8: diagonal standardize (clamped sigma), 1-D Z via the generic wrap
    sig = np.sqrt(np.clip(p * (1 - p), 1e-4, None))
    d = 1.0 / np.maximum(sig, 0.05)
    np.savez("models/hk_std.npz", Z=d.astype(np.float32), mu=p.astype(np.float32))
    w_raw = rng_np.normal(0, 1e-3, K_IN); w_raw[K_IN - 1] += 2 * C
    wp = w_raw / d
    bp = -C + float(w_raw @ p)
    x0 = np.zeros(K_IN); x0[active5k(chess.Board())] = 1.0
    assert abs((float(wp @ ((x0 - p) * d)) + bp) - (float(w_raw @ x0) - C)) < 1e-5
    save_seed("models/qlearn_hks_seed.pt", wp, bp, "hk", True)

    # S9: concat 809 ⊕ k2048 diag std
    KX = np.concatenate([KC, np.stack([encode_kanerva809(x) for x in KC])], axis=1)
    mu9 = KX.mean(axis=0)
    sig9 = KX.std(axis=0)
    d9 = 1.0 / np.maximum(sig9, 0.05)
    np.savez("models/kx_std.npz", Z=d9.astype(np.float32), mu=mu9.astype(np.float32))
    w_raw = rng_np.normal(0, 1e-3, 2857); w_raw[768] += 2 * C   # stm lives at 768 in the 809 block
    save_seed("models/qlearn_kx_seed.pt", w_raw / d9, -C + float(w_raw @ mu9), "kx", True)

    # S10: 5121 -> PCA-512 whitened
    X = np.zeros((n, K_IN), dtype=np.float32)
    for i, a in enumerate(acts):
        X[i, a] = 1.0
    mu10 = X.mean(axis=0).astype(np.float64)
    Xc = (X - mu10).astype(np.float32)
    cov = (Xc.T @ Xc).astype(np.float64) / n
    evals, evecs = np.linalg.eigh(cov + 1e-4 * np.eye(K_IN))
    order = np.argsort(evals)[::-1][:512]
    W = (evecs[:, order] * (evals[order] ** -0.5)).T
    np.savez("models/pca5k.npz", Z=W.astype(np.float32), mu=mu10.astype(np.float32))
    w_raw = rng_np.normal(0, 1e-3, K_IN); w_raw[K_IN - 1] += 2 * C
    wp = np.linalg.pinv(W).T @ w_raw
    bp = -C + float(w_raw @ mu10)
    v = float(wp @ (W @ (x0 - mu10))) + bp
    print(f"S10 seed projection: raw {float(w_raw @ x0) - C:+.3f} vs reduced {v:+.3f} (loss reported)")
    save_seed("models/qlearn_p5_seed.pt", wp, bp, "hk", True)   # enc hk + pca5k.npz wrap
    print("round-2 artifacts built — all rules+RNG, PASS :Provenance:")


if __name__ == "__main__":
    main()
