"""Linear afterstate-value RL (spec/linear-value-rl.spec.md) — TD-Gammon-style.

:Linear-value: V(s)=w·φ(s), TRUE linear (S&B §9.4 single-optimum). φ = gbdt_features (material+PST) or
rich_features (LINEAR_FEATURES=rich). :Feature-reduction: (LINEAR_REDUCE=pca) standardizes + PCA-
collapses the correlated φ to the variance elbow, warm-started by PROJECTING linear.npz into PCA space
(the fix for the 42-feature overfit). :Afterstate-target: is MC outcome z (default) or, with
LINEAR_TARGET=td, the semi-gradient TD(0)/PBE one-step bootstrap (lower variance, stable here: on-policy
+ linear + deterministic). :Parallel-selfplay: — MULTIPROCESSING alpha-beta self-play (CPU, no GPU).
:Linear-kill-check: — beat pst at equal depth.

Modes: LINEAR_FEATURES=rich|(material) · LINEAR_REDUCE=pca|(none) · LINEAR_TARGET=td|mc
Usage: python train_linear_rl.py [iters] [games_per_iter] [workers]
"""
import os, sys, time
from multiprocessing import Pool, cpu_count

import numpy as np
import chess

from engine import AlphaBetaEngine, pst_eval
from measure_ladder import random_mover, heuristic_mover, adj_pst, play, elo_diff

# Read at IMPORT time so spawned MP workers agree (φ set); reducer/target reach workers via args.
if os.environ.get("LINEAR_FEATURES", "").lower() == "rich":
    from rich_features import features
    _FEATSET = "rich"
else:
    from gbdt_features import features
    _FEATSET = "material+PST"
_REDUCE = os.environ.get("LINEAR_REDUCE", "").lower()          # "pca" | ""
_TARGET = os.environ.get("LINEAR_TARGET", "mc").lower()        # "td" | "mc"

SELFPLAY_DEPTH = 2
LINEAR_RANDOM_PLIES = 4
GAME_CAP = 120
LINEAR_LR = 0.01
GD_EPOCHS = 40
PCA_VAR_KEEP = 0.95
PCA_K_CAP = 15


def _phi(board, reducer):
    """Feature vector, PCA-reduced if a reducer (mu, sigma, comp) is given."""
    raw = features(board).astype(np.float64)
    if reducer is None:
        return raw
    mu, sigma, comp = reducer
    return comp @ ((raw - mu) / sigma)


def _eval_fn(w, reducer):
    """w·φ + bias → White-absolute score (same frame as pst_eval)."""
    wf, wb = w[:-1], w[-1]
    return lambda board: float(_phi(board, reducer) @ wf + wb)


def _play_one(args):
    """Worker: one alpha-beta self-play game. Returns (list φ, list target) — target = z (MC) or the
    TD(0) one-step bootstrap ev(s_next), z at terminal. All White-absolute."""
    w, reducer, target_mode, seed = args
    ev = _eval_fn(w, reducer)
    eng = AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=SELFPLAY_DEPTH)
    rng = np.random.RandomState(seed)
    board = chess.Board()
    for _ in range(LINEAR_RANDOM_PLIES):
        if board.is_game_over():
            break
        mvs = list(board.legal_moves); board.push(mvs[rng.randint(len(mvs))])
    seq, plies = [], 0
    while not board.is_game_over() and plies < GAME_CAP:
        mv = eng.search(board)[0]
        if mv is None or mv not in board.legal_moves:
            break
        seq.append(board.copy(stack=False))
        board.push(mv); plies += 1
    seq.append(board.copy(stack=False))                        # final state (terminal or cap)
    z = 1.0 if board.is_checkmate() and board.turn == chess.BLACK else \
        (-1.0 if board.is_checkmate() else 0.0)
    phis, targets = [], []
    for i in range(len(seq) - 1):
        phis.append(_phi(seq[i], reducer))
        if target_mode == "td" and i + 1 < len(seq) - 1:        # bootstrap; last transition uses z
            targets.append(ev(seq[i + 1]))
        else:
            targets.append(z)
    return phis, targets


def _fit_pca(X, var_keep=PCA_VAR_KEEP, k_cap=PCA_K_CAP):
    """Standardize + PCA (numpy SVD). Keep k comps at cumulative-variance >= var_keep."""
    mu = X.mean(0); sigma = X.std(0) + 1e-8
    _, S, Vt = np.linalg.svd((X - mu) / sigma, full_matrices=False)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    k = min(max(int(np.searchsorted(cum, var_keep)) + 1, 2), k_cap, Vt.shape[0])
    return (mu, sigma, Vt[:k]), k, float(cum[k - 1])


def _project_warmstart(reducer, coef, intercept):
    """Project the pst weights (linear.npz) into PCA space so iter 0 plays ~pst."""
    mu, sigma, comp = reducer
    c = np.zeros(len(mu)); c[:len(coef)] = coef; c /= 400.0     # raw-space pst weights
    w_pca = comp @ (c * sigma)                                 # onto the retained subspace
    b = float(c @ mu) + float(intercept) / 400.0
    return np.concatenate([w_pca, [b]])


def _ladder(w, reducer, depth, games):
    ev = _eval_fn(w, reducer)
    mv = lambda b: AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=depth).search(b)[0]
    out = {}
    for name, opp in (("random", random_mover), ("heuristic", heuristic_mover)):
        W, D, L = play(mv, opp, games, adj_pst)
        out[name] = (W + 0.5 * D) / (W + D + L)
    return out


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    games_per = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, cpu_count() - 1)
    d = np.load("models/linear.npz")
    coef = d["coef"].astype(np.float64); intercept = float(d["intercept"])
    D_raw = len(features(chess.Board()))

    with Pool(workers) as pool:
        # unreduced warm-start weights (pst-like) — used raw, or to seed the sample + projection
        w_raw = np.zeros(D_raw + 1); w_raw[:len(coef)] = coef / 400.0; w_raw[-1] = intercept / 400.0
        reducer = None
        if _REDUCE == "pca":
            rng0 = np.random.RandomState(1)
            sample = pool.map(_play_one, [(w_raw, None, "mc", int(s)) for s in rng0.randint(0, 2**31 - 1, 16)])
            X = np.array([p for ps, _ in sample for p in ps], dtype=np.float64)
            reducer, k, cvr = _fit_pca(X)
            w = _project_warmstart(reducer, coef, intercept)
            print(f"PCA: {D_raw}d -> {k}d (cum.var {cvr:.2f}) from {len(X)} sample positions", flush=True)
        else:
            w = w_raw

        m = np.zeros_like(w); v = np.zeros_like(w); t = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        rng = np.random.RandomState(0)
        tag = f"{_FEATSET}/{_REDUCE or 'raw'}/{_TARGET}"
        print(f"Linear-value RL [{tag}, phi={len(w)-1}d]: {iters}x{games_per} games, {workers} workers, "
              f"d{SELFPLAY_DEPTH} (CPU)\n", flush=True)
        out = f"models/linear_rl_{_REDUCE or 'raw'}_{_TARGET}.npz"

        for it in range(iters):
            t0 = time.time()
            seeds = rng.randint(0, 2**31 - 1, size=games_per)
            results = pool.map(_play_one, [(w, reducer, _TARGET, int(s)) for s in seeds])
            X = np.array([p for ps, _ in results for p in ps], dtype=np.float64)
            Y = np.array([y for _, ys in results for y in ys], dtype=np.float64)
            X = np.append(X, np.ones((len(X), 1)), axis=1)     # bias column
            for _ in range(GD_EPOCHS):                          # semi-gradient GD on MSE(w·φ, target)
                g = ((X @ w - Y)[:, None] * X).mean(0)
                t += 1
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
                w -= LINEAR_LR * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            mse = float(np.mean((X @ w - Y) ** 2))
            lad = _ladder(w, reducer, SELFPLAY_DEPTH, 10)
            print(f"iter {it}: mse={mse:.3f}  vs_random={lad['random']:.2f}  "
                  f"vs_heuristic={lad['heuristic']:.2f}  n={len(Y)}  [{time.time()-t0:.0f}s]", flush=True)
            save = dict(w=w, depth=SELFPLAY_DEPTH, featset=_FEATSET, target=_TARGET)
            if reducer is not None:
                save.update(pca_mu=reducer[0], pca_sigma=reducer[1], pca_comp=reducer[2])
            np.savez(out, **save)

    # :Linear-kill-check: head-to-head vs pst at equal depth
    ev = _eval_fn(w, reducer)
    lin = lambda b: AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=SELFPLAY_DEPTH).search(b)[0]
    pst = lambda b: AlphaBetaEngine(eval_fn=pst_eval, time_limit=1e9, max_depth=SELFPLAY_DEPTH).search(b)[0]
    W, D, L = play(lin, pst, 30, adj_pst)
    s = (W + 0.5 * D) / (W + D + L)
    print(f"\nKILL-CHECK [{_FEATSET}/{_REDUCE or 'raw'}/{_TARGET}] vs pst @d{SELFPLAY_DEPTH}: "
          f"{W}W-{D}D-{L}L  score {s:.2f}  elo_diff {elo_diff(s):+.0f}  "
          f"({'BEATS pst' if s > 0.5 else 'does NOT beat pst'})", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
