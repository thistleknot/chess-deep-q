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
from value_targets import lambda_return, to_white

# Read at IMPORT time so spawned MP workers agree (φ set); reducer/target reach workers via args.
if os.environ.get("LINEAR_FEATURES", "").lower() == "rich":
    from rich_features import features
    _FEATSET = "rich"
else:
    from gbdt_features import features
    _FEATSET = "material+PST"
_REDUCE = os.environ.get("LINEAR_REDUCE", "").lower()          # "pca" | ""
_TARGET = os.environ.get("LINEAR_TARGET", "mc").lower()        # "lambda" | "td" | "mc"
# :Two-dial-hybrid: — tau (exploration breadth) + lambda (value-target bias-variance).
_TAU = float(os.environ.get("LINEAR_TAU", "0") or 0)          # 0 = argmax; >0 = tempered top-K sampling
_LAMBDA = float(os.environ.get("LINEAR_LAMBDA", "0.7") or 0.7)
LINEAR_TEMP_PLIES = int(os.environ.get("LINEAR_TEMP_PLIES", "6") or 6)
LINEAR_TOPK = 5
KILL_GAMES = int(os.environ.get("KILL_GAMES", "30") or 30)     # kill-check n (raise to beat noise when tuning)

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


def _sample_move(board, eng, ev, rng):
    """tau-tempered softmax over the TOP-K moves by engine ordering, scored by afterstate value in
    the mover frame ('scoring temperature near max moves'). Ordering ranks captures/checks high, so
    the sample keeps refutations — sound exploration, not the old uniform beam."""
    legal = list(board.legal_moves)
    if len(legal) == 1:
        return legal[0]
    ordered = eng._order(board, legal, None, 0)[:LINEAR_TOPK]
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    scores = np.empty(len(ordered))
    for i, m in enumerate(ordered):
        board.push(m); scores[i] = sign * ev(board); board.pop()
    p = np.exp((scores - scores.max()) / max(_TAU, 1e-6)); p /= p.sum()
    return ordered[int(rng.choice(len(ordered), p=p))]


def _play_one(args):
    """Worker: one SOUND alpha-beta self-play game. Records the White-absolute sound-search value per
    state; target = lambda_return (mc=lam1 / td=lam0 / interior). tau>0 => tempered top-K sampling at
    opening plies (:Two-dial-hybrid:). All White-absolute."""
    w, reducer, target_mode, seed = args
    ev = _eval_fn(w, reducer)
    eng = AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=SELFPLAY_DEPTH)
    rng = np.random.RandomState(seed)
    board = chess.Board()
    for _ in range(LINEAR_RANDOM_PLIES):
        if board.is_game_over():
            break
        mvs = list(board.legal_moves); board.push(mvs[rng.randint(len(mvs))])
    phis, vsearch, plies = [], [], 0
    while not board.is_game_over() and plies < GAME_CAP:
        best, info = eng.search(board)
        if best is None or best not in board.legal_moves:
            break
        phis.append(_phi(board, reducer))
        # Bootstrap V(s) on the OUTCOME scale: clamp to [-1,1] so the engine's mate injection (+-1e6)
        # and large material scores don't blow up the lambda-return (the terminal reward z is +-1).
        vw = to_white(float(info.get("score_cp", 0.0)), board.turn == chess.WHITE)
        vsearch.append(max(-1.0, min(1.0, vw)))
        mv = _sample_move(board, eng, ev, rng) if (_TAU > 0 and plies < LINEAR_TEMP_PLIES) else best
        board.push(mv); plies += 1
    z = 1.0 if board.is_checkmate() and board.turn == chess.BLACK else \
        (-1.0 if board.is_checkmate() else 0.0)
    n = len(phis)
    if n == 0:
        return [], []
    lam = {"mc": 1.0, "td": 0.0}.get(target_mode, _LAMBDA)      # mc/td are lambda_return endpoints
    rewards = [0.0] * (n - 1) + [z]
    boot = vsearch[1:] + [0.0]                                  # bootstrap_values[t] = V(s_{t+1})
    dones = [False] * (n - 1) + [True]
    targets = lambda_return(rewards, boot, dones, lam)
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
        tag = f"{_FEATSET}/{_REDUCE or 'raw'}/{_TARGET}" + \
            (f"(lam={_LAMBDA},tau={_TAU})" if _TARGET == "lambda" else "")
        print(f"Linear-value RL [{tag}, phi={len(w)-1}d]: {iters}x{games_per} games, {workers} workers, "
              f"d{SELFPLAY_DEPTH} (CPU)\n", flush=True)
        out = f"models/linear_rl_{_TARGET}" + \
            (f"_l{str(_LAMBDA).replace('.', '')}" if _TARGET == "lambda" else "") + ".npz"

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
    W, D, L = play(lin, pst, KILL_GAMES, adj_pst)
    s = (W + 0.5 * D) / (W + D + L)
    print(f"\nKILL-CHECK [{_FEATSET}/{_REDUCE or 'raw'}/{_TARGET}] vs pst @d{SELFPLAY_DEPTH}: "
          f"{W}W-{D}D-{L}L  score {s:.2f}  elo_diff {elo_diff(s):+.0f}  "
          f"({'BEATS pst' if s > 0.5 else 'does NOT beat pst'})", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
