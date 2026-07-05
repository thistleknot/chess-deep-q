"""NNUE kill-check (spec/nnue-eval.spec.md :NNUE-kill-check:, equal-DEPTH gate).

Plays the trained NNUE eval inside alpha-beta against pst_eval at the SAME depth on the ladder, n games
with adjudication, and reports the score + Elo delta. This is the Stage-2 test: does the enriched-data
distilled critic move off the v1 wall (v1 was d2=808 / d3=-280 vs pst 1367/1672)? Equal-TIME is a
separate gate that needs the incremental accumulator (Stage 3) — this only judges eval QUALITY at fixed
depth. CPU eval on purpose: single-position torch is faster on CPU than per-call GPU launch overhead.

Usage: python measure_nnue.py [games] [depth]
"""
import sys

import chess
from engine import AlphaBetaEngine, pst_eval
from measure_ladder import play, adj_pst, elo_diff
from nnue_model import load_nnue, make_nnue_eval


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    import os
    net = load_nnue(os.environ.get("NNUE_MODEL", "models/nnue.pt"), "cpu")
    ev = make_nnue_eval(net, "cpu")
    nnue_mv = lambda b: AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=depth).search(b)[0]
    pst_mv = lambda b: AlphaBetaEngine(eval_fn=pst_eval, time_limit=1e9, max_depth=depth).search(b)[0]
    W, D, L = play(nnue_mv, pst_mv, games, adj_pst)
    score = (W + 0.5 * D) / max(W + D + L, 1)
    verdict = "BEATS pst" if score > 0.5 else "does NOT beat pst"
    print(f"NNUE vs pst @d{depth}: {W}W-{D}D-{L}L  score {score:.2f}  elo_diff {elo_diff(score):+.0f}  ({verdict})")


if __name__ == "__main__":
    main()
