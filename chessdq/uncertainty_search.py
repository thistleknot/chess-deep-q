"""Uncertainty-aware search depth allocation.

Same champion eval, same native minimax (rsearch4). The only change: positions
where the ensemble disagrees (high σ) get MORE search depth. Positions where
the model is confident get less. Same total time budget, better allocated.

This is the deployment agent — no training, no refit. Just smarter time management.
"""
import math
import time

import numpy as np
import chess

from chessdq.encoders import get
from chessdq.corpus_gen import raw_weights

_ENC_FN, _NIN = get("amap")


class UncertaintySearchAgent:
    """Champion eval + native minimax, with uncertainty-directed depth allocation.

    At each move:
      1. Compute σ_ens for the current position
      2. If σ > threshold: search at deep_depth (e.g. 11)
         Else: search at shallow_depth (e.g. 7)
      3. Return the move from the native alpha-beta search

    The ensemble is precomputed (K ridge heads over the existing corpus).
    The searcher is rsearch4.Searcher (native Rust alpha-beta, same as champion).
    """

    def __init__(self, W_ens, B_ens, searcher,
                 sigma_threshold=0.010,
                 deep_depth=11, shallow_depth=7):
        self.W = np.asarray(W_ens, dtype=np.float64)
        self.B = np.asarray(B_ens, dtype=np.float64)
        self.searcher = searcher
        self.sigma_threshold = sigma_threshold
        self.deep_depth = deep_depth
        self.shallow_depth = shallow_depth
        self._deep_count = 0
        self._shallow_count = 0

    def _sigma(self, board):
        """Ensemble disagreement for this position."""
        x = _ENC_FN(board).astype(np.float64)
        preds = x @ self.W.T + self.B  # shape (K,)
        return float(preds.std())

    def __call__(self, board):
        sigma = self._sigma(board)
        if sigma > self.sigma_threshold:
            depth = self.deep_depth
            self._deep_count += 1
        else:
            depth = self.shallow_depth
            self._shallow_count += 1
        move_uci = self.searcher.search(board.fen(), depth)[0]
        return chess.Move.from_uci(move_uci)

    def stats(self):
        total = self._deep_count + self._shallow_count
        if total == 0:
            return "no moves played"
        return (f"deep({self.deep_depth}): {self._deep_count} "
                f"({self._deep_count/total:.0%}), "
                f"shallow({self.shallow_depth}): {self._shallow_count} "
                f"({self._shallow_count/total:.0%})")


def build_agent(ckpt="models/champion.pt", K=16, ridge=100.0, seed=977,
                sigma_threshold=0.010, deep_depth=11, shallow_depth=7):
    """Build the full agent: ensemble + native searcher."""
    import importlib
    from experiments.distill_linear import CACHE, featurize, fit_ridge

    # Load champion weights for the searcher
    w, b = raw_weights(ckpt)
    rs = importlib.import_module("rsearch4")
    searcher = rs.Searcher(w, b)

    # Build ensemble
    z = np.load(CACHE, allow_pickle=True)
    fens, y = list(z["fens"]), z["y"]
    X = featurize(fens)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    W = np.empty((K, X.shape[1]), dtype=np.float64)
    B = np.empty(K, dtype=np.float64)
    for k in range(K):
        idx = rng.integers(0, n, size=n)
        w_k, b_k, _ = fit_ridge(X[idx], y[idx], ridge)
        W[k], B[k] = w_k, b_k

    return UncertaintySearchAgent(W, B, searcher,
                                   sigma_threshold=sigma_threshold,
                                   deep_depth=deep_depth,
                                   shallow_depth=shallow_depth)


def run_ladder(games=20, deep_depth=11, shallow_depth=7, sigma_threshold=0.010):
    """Measure uncertainty-search agent on the standard anchor ladder."""
    import importlib
    from experiments.anchor_ladder import mle_rating, STANDARD_ANCHORS
    import glob
    import chess.engine

    print("=" * 60)
    print("UNCERTAINTY-AWARE SEARCH — LADDER TEST")
    print(f"  deep={deep_depth} at σ>{sigma_threshold}, shallow={shallow_depth} elsewhere")
    print("=" * 60)

    t0 = time.time()
    print("Building agent (ensemble + native searcher)...", flush=True)
    agent = build_agent(deep_depth=deep_depth, shallow_depth=shallow_depth,
                        sigma_threshold=sigma_threshold)
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    # Load SF
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if not sfp:
        raise SystemExit("No stockfish under engines/")
    sf = chess.engine.SimpleEngine.popen_uci(sfp[0])
    sf_lim = chess.engine.Limit(time=0.05)

    # Bell-curve allocation scaled to games
    ratios = [10, 25, 30, 25, 10]
    ratio_sum = sum(ratios)
    games_per = [max(1, round(games * r / ratio_sum)) for r in ratios]
    diff = games - sum(games_per)
    games_per[2] += diff

    results = []
    per = {a: [0, 0, 0] for a in STANDARD_ANCHORS}
    PLY_CAP = 120

    for anchor, n_g in zip(STANDARD_ANCHORS, games_per):
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": int(anchor)})
        print(f"  vs SF@{anchor} ({n_g}g)...", end=" ", flush=True)
        for g in range(n_g):
            agent_white = (g % 2 == 0)
            board = chess.Board()
            plies = 0
            while not board.is_game_over() and plies < PLY_CAP:
                if (board.turn == chess.WHITE) == agent_white:
                    mv = agent(board)
                else:
                    mv = sf.play(board, sf_lim).move
                board.push(mv)
                plies += 1
            if board.is_checkmate():
                res = 1.0 if (board.turn == chess.BLACK) == agent_white else 0.0
            else:
                res = 0.5
            per[anchor][0 if res == 1.0 else 1 if res == 0.5 else 2] += 1
            results.append((anchor, res))
        W, D, L = per[anchor]
        s = (W + 0.5 * D) / n_g
        print(f"{W}W {D}D {L}L  score={s:.3f}", flush=True)

    sf.quit()

    r, se = mle_rating(results)
    lo, hi = r - 1.96 * se, r + 1.96 * se
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"MLE RATING: {r:.0f} (95% CI: {lo:.0f}..{hi:.0f})")
    print(f"Depth allocation: {agent.stats()}")
    print(f"Wall-clock: {elapsed:.0f}s ({elapsed/sum(games_per):.1f}s/game)")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    deep = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    shallow = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    run_ladder(games=games, deep_depth=deep, shallow_depth=shallow)
