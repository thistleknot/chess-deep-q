"""Measure the hybrid eval = pst_eval + lambda*residual in engine.py alpha-beta vs SF@1320.

lambda=0 is the pure heuristic (baseline ~1367 d2 / 1672 d3). lambda>0 adds the learned correction.
If a lambda>0 beats lambda=0, the LEARNED component genuinely improves strength.

Usage: python measure_hybrid.py [lambda] [games] [depths]   e.g. python measure_hybrid.py 0.5 30 2
"""
import sys
import chess
import xgboost as xgb

from engine import pst_eval, AlphaBetaEngine
from gbdt_features import features
from measure_gbdt import play_match, report

RES = "models/gbdt_residual.json"


def main():
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    depths = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['2'])]

    if lam == 0.0:
        ev = pst_eval   # pure heuristic baseline
    else:
        bst = xgb.Booster(); bst.load_model(RES); bst.set_param({"device": "cpu"})
        def ev(board):
            return pst_eval(board) + lam * float(bst.inplace_predict(features(board)[None, :])[0])

    for d in depths:
        eng = AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=d)
        print(f"[hybrid lambda={lam} in alpha-beta, depth {d}]")
        W, D, L = play_match(lambda b, e=eng: e.search(b)[0], games)
        report(f'hybrid-lam{lam}-d{d}', W, D, L)


if __name__ == "__main__":
    main()
