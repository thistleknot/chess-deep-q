import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))
"""Measure the uncertainty-MCTS derived model on the standard multi-anchor Elo ladder.

Uses the Python AlphaBetaEngine (with champion_umcts.pt weights) as the agent,
Stockfish at UCI_Elo anchors as the opponent. Standard protocol: 100 games,
bell-curve allocation across 900/1200/1500/1800/2100.

The agent plays at whatever depth the time_limit allows (default 1.0s/move).
This measures the EVAL + Python-search strength, not native d9 deployment.

Usage: python -m experiments.measure_umcts [time_limit=1.0] [workers=4]
"""
import glob
import math
import os
import sys
import time

import chess
import chess.engine
import numpy as np

from chessdq.engine import AlphaBetaEngine
from chessdq.corpus_gen import raw_weights
from chessdq.encoders import get
from experiments.anchor_ladder import mle_rating, STANDARD_ANCHORS, STANDARD_GAMES_PER

CKPT = os.environ.get("MEASURE_CKPT", "models/champion_umcts.pt")
PLY_CAP = 120


def make_agent(ckpt, time_limit):
    """Build a Python AlphaBetaEngine mover from the checkpoint."""
    enc_fn, _nin = get("amap")
    w, b = raw_weights(ckpt)
    w = np.asarray(w, dtype=np.float64)
    b_val = float(b)

    def eval_fn(board):
        x = enc_fn(board).astype(np.float64)
        return 800.0 * (w @ x + b_val)

    engine = AlphaBetaEngine(eval_fn=eval_fn, time_limit=time_limit, max_depth=9)

    def mover(board):
        mv, _info = engine.search(board)
        return mv

    return mover


def play_one(mover, sf, anchor, agent_white, lim):
    """Play one game, return agent-perspective result."""
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": int(anchor)})
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < PLY_CAP:
        if (board.turn == chess.WHITE) == agent_white:
            mv = mover(board)
        else:
            mv = sf.play(board, lim).move
        if mv is None or mv not in board.legal_moves:
            break
        board.push(mv)
        plies += 1
    if board.is_checkmate():
        return 1.0 if (board.turn == chess.BLACK) == agent_white else 0.0
    return 0.5


def main():
    time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    total_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # Bell-curve allocation scaled to budget (same shape as standard 100g)
    # Standard ratios: 10:25:30:25:10 → scale to budget
    ratios = [10, 25, 30, 25, 10]
    ratio_sum = sum(ratios)
    raw = [total_budget * r / ratio_sum for r in ratios]
    # Round to ints summing to budget
    games_per = [max(1, round(x)) for x in raw]
    diff = total_budget - sum(games_per)
    # Adjust middle anchor to hit exact budget
    games_per[2] += diff

    print("=" * 60)
    print("STANDARD ELO MEASUREMENT — champion_umcts.pt")
    print("=" * 60)
    print(f"Agent: {CKPT} (Python AlphaBetaEngine, time_limit={time_limit}s)")
    print(f"Anchors: {STANDARD_ANCHORS}")
    print(f"Games per anchor: {games_per} (total {sum(games_per)})")
    print()

    # Load agent
    mover = make_agent(CKPT, time_limit)

    # Load Stockfish
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if not sfp:
        raise SystemExit("No stockfish found under engines/")
    sf = chess.engine.SimpleEngine.popen_uci(sfp[0])
    sf_lim = chess.engine.Limit(time=0.05)

    results = []  # (anchor, score)
    per = {a: [0, 0, 0] for a in STANDARD_ANCHORS}  # W, D, L
    t0 = time.time()
    total_games = sum(games_per)
    done = 0

    for anchor, n_games in zip(STANDARD_ANCHORS, games_per):
        print(f"  vs SF@{anchor} ({n_games} games)...", flush=True)
        for g in range(n_games):
            agent_white = (g % 2 == 0)
            res = play_one(mover, sf, anchor, agent_white, sf_lim)
            per[anchor][0 if res == 1.0 else 1 if res == 0.5 else 2] += 1
            results.append((anchor, res))
            done += 1
        W, D, L = per[anchor]
        s = (W + 0.5 * D) / n_games
        print(f"    {W}W {D}D {L}L  score={s:.3f}", flush=True)

    sf.quit()

    # MLE rating
    r, se = mle_rating(results)
    lo, hi = r - 1.96 * se, r + 1.96 * se

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"MLE RATING: {r:.0f} (95% CI: {lo:.0f} .. {hi:.0f})")
    print(f"{'=' * 60}")
    print(f"Total: {done} games in {elapsed:.0f}s ({elapsed/done:.1f}s/game)")
    print(f"Vehicle: Python AlphaBetaEngine @ {time_limit}s/move (NOT native d9)")
    print(f"Note: native d9 (rsearch4) would be stronger — this is a FLOOR estimate.")


if __name__ == "__main__":
    main()
