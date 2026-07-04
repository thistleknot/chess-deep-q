"""Linear afterstate-value RL (spec/linear-value-rl.spec.md) — TD-Gammon-style.

:Linear-value: V(s)=w·φ(s) — TRUE linear (no squash), the case S&B §9.4 shows has a SINGLE global
optimum / guaranteed convergence (a tanh would forfeit exactly that). φ = `gbdt_features.features`.
:Afterstate-target: — linear gradient-MC (S&B 9.5): GD regresses w toward the self-play OUTCOME z,
swapping the supervised SF label; semi-gradient TD(λ) is the bootstrapped extension. :Parallel-selfplay:
— MULTIPROCESSING alpha-beta self-play (µs eval, NO GPU) → minutes, not hours. Warm-started from
models/linear.npz. Adam handles the feature-scale spread (counts ~1 vs PST sums ~100s). Caveat (S&B
9.2): min-VE is a proxy, not the policy objective. :Linear-kill-check: — beat pst at equal depth.

Usage: python train_linear_rl.py [iters] [games_per_iter] [workers]
"""
import os, sys, time
from multiprocessing import Pool, cpu_count

import numpy as np
import chess

from engine import AlphaBetaEngine, pst_eval
from gbdt_features import features
from measure_ladder import random_mover, heuristic_mover, adj_pst, play, elo_diff

SELFPLAY_DEPTH = 2
LINEAR_RANDOM_PLIES = 4
GAME_CAP = 120
LINEAR_LR = 0.01          # Adam step
GD_EPOCHS = 40


def _eval_fn(w):
    """w·φ + bias (last w) → White-absolute score; monotone leaf eval for alpha-beta."""
    wf, wb = w[:-1], w[-1]
    return lambda board: float(features(board) @ wf + wb)


def _play_one(args):
    """Worker: one alpha-beta(:Linear-value:) self-play game. Returns (list of φ, outcome z)."""
    w, seed = args
    rng = np.random.RandomState(seed)
    eng = AlphaBetaEngine(eval_fn=_eval_fn(w), time_limit=1e9, max_depth=SELFPLAY_DEPTH)
    board = chess.Board()
    for _ in range(LINEAR_RANDOM_PLIES):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        board.push(moves[rng.randint(len(moves))])
    phis, plies = [], 0
    while not board.is_game_over() and plies < GAME_CAP:
        mv = eng.search(board)[0]
        if mv is None or mv not in board.legal_moves:
            break
        phis.append(features(board))
        board.push(mv); plies += 1
    z = 0.0
    if board.is_checkmate():
        z = 1.0 if board.turn == chess.BLACK else -1.0
    return phis, z


def _ladder(w, depth, games):
    """Cheap per-iter ladder: linear-RL engine vs random + heuristic-1ply (equal depth)."""
    ev = _eval_fn(w)
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

    d = np.load("models/linear.npz")                       # warm-start (pst-like), rescaled for tanh
    w = np.concatenate([d["coef"].astype(np.float64), [float(d["intercept"])]]) / 400.0
    m = np.zeros_like(w); v = np.zeros_like(w); t = 0
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.RandomState(0)
    print(f"Linear-value RL: {iters} iters x {games_per} games, {workers} workers, "
          f"depth {SELFPLAY_DEPTH} (CPU, no GPU)\n", flush=True)

    with Pool(workers) as pool:
        for it in range(iters):
            t0 = time.time()
            seeds = rng.randint(0, 2**31 - 1, size=games_per)
            results = pool.map(_play_one, [(w, int(s)) for s in seeds])
            X = np.array([phi for phis, _ in results for phi in phis], dtype=np.float64)
            Y = np.array([z for phis, z in results for _ in phis], dtype=np.float64)
            X = np.append(X, np.ones((len(X), 1)), axis=1)  # bias column
            for _ in range(GD_EPOCHS):                       # linear gradient-MC (S&B 9.4/9.5): v=w·φ
                pred = X @ w                                 # TRUE linear — keeps the single-optimum guarantee
                g = ((pred - Y)[:, None] * X).mean(0)        # ∇½(w·φ − z)² = (w·φ − z)φ
                t += 1
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                w -= LINEAR_LR * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            mse = float(np.mean((X @ w - Y) ** 2))
            lad = _ladder(w, SELFPLAY_DEPTH, 20)
            print(f"iter {it}: mse={mse:.3f}  vs_random={lad['random']:.2f}  "
                  f"vs_heuristic={lad['heuristic']:.2f}  n={len(Y)}  [{time.time()-t0:.0f}s]", flush=True)
            np.savez("models/linear_rl.npz", w=w, depth=SELFPLAY_DEPTH)

    # :Linear-kill-check: head-to-head vs pst at equal depth (the ship gate)
    ev = _eval_fn(w)
    lin = lambda b: AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=SELFPLAY_DEPTH).search(b)[0]
    pst = lambda b: AlphaBetaEngine(eval_fn=pst_eval, time_limit=1e9, max_depth=SELFPLAY_DEPTH).search(b)[0]
    W, D, L = play(lin, pst, 30, adj_pst)
    s = (W + 0.5 * D) / (W + D + L)
    print(f"\nKILL-CHECK linear-RL vs pst @d{SELFPLAY_DEPTH}: {W}W-{D}D-{L}L  score {s:.2f}  "
          f"elo_diff {elo_diff(s):+.0f}  ({'BEATS pst' if s > 0.5 else 'does NOT beat pst'})", flush=True)
    print("saved models/linear_rl.npz")


if __name__ == "__main__":
    main()
