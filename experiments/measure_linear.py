import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Measure the smooth linear learned eval in engine.py alpha-beta vs SF@1320.

Usage: python measure_linear.py [games] [depths]
"""
import sys
import numpy as np
import chess

from chessdq.engine import AlphaBetaEngine
from chessdq.gbdt_features import features
from chessdq.measure_gbdt import play_match, report


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    depths = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['2'])]
    d = np.load("models/linear.npz")
    coef, intercept = d["coef"], float(d["intercept"])

    def ev(board):
        return float(features(board) @ coef + intercept)   # smooth, µs-fast dot product

    for depth in depths:
        eng = AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=depth)
        print(f"[linear learned eval in alpha-beta, depth {depth}]")
        W, D, L = play_match(lambda b, e=eng: e.search(b)[0], games)
        report(f'linear-d{depth}', W, D, L)


if __name__ == "__main__":
    main()
