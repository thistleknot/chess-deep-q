import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Time-controlled eval bake-off: which evaluator is strongest at a fixed thinking time?

The fixed-depth ladder (measure_gbdt/measure_hybrid) is unbounded at d4+ (~20s/game already at d3,
~5-6x per extra ply). This runs each eval in AlphaBetaEngine at a fixed WALL-CLOCK budget
(time_limit, default 0.3s/move) vs SF@1320 -- bounded (~17s/game) and the real playing condition:
a deeper-capable eval automatically banks its depth benefit within the budget. Settles spec Q2
(is search or eval the binding constraint) with binomial CIs.

Usage: python measure_timed.py [time_s] [games] [evals]
  e.g. python measure_timed.py 0.3 30 pst,gbdt,hybrid:0.5
Eval names: pst | gbdt | linear | hybrid:LAMBDA (pst + LAMBDA*gbdt_residual)
"""
import sys, time

from chessdq.engine import pst_eval, AlphaBetaEngine
from chessdq.measure_gbdt import play_match, report


def make_eval(name):
    """Return (label, eval_fn) for an eval spec. eval_fn(board) -> White-absolute centipawns."""
    if name == "pst":
        return "pst", pst_eval
    if name == "gbdt":
        from chessdq.measure_gbdt import make_eval as mk
        return "gbdt", mk("models/gbdt.json")
    if name == "linear":
        import numpy as np
        from chessdq.gbdt_features import features
        d = np.load("models/linear.npz")
        coef, intercept = d["coef"], float(d["intercept"])
        return "linear", (lambda board: float(features(board) @ coef + intercept))
    if name.startswith("hybrid"):
        import xgboost as xgb
        from chessdq.gbdt_features import features
        lam = float(name.split(":", 1)[1]) if ":" in name else 0.5
        bst = xgb.Booster(); bst.load_model("models/gbdt_residual.json"); bst.set_param({"device": "cpu"})
        def ev(board):
            return pst_eval(board) + lam * float(bst.inplace_predict(features(board)[None, :])[0])
        return f"hybrid-lam{lam}", ev
    if name == "nnue":
        # Per-node alpha-beta eval → CPU (launch-free); the kill-check measures time at equal budget.
        from chessdq.nnue_model import load_nnue, make_nnue_eval
        return "nnue", make_nnue_eval(load_nnue("models/nnue.pt", "cpu"), "cpu")
    raise SystemExit(f"unknown eval '{name}'")


def main():
    tsec = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else ["pst", "gbdt", "hybrid:0.5"]
    print(f"Time-control bake-off: {tsec}s/move, {games} games vs SF@1320, evals={names}\n")
    for name in names:
        label, ev = make_eval(name)
        eng = AlphaBetaEngine(eval_fn=ev, time_limit=tsec)   # max_depth default 64 => time-bounded
        print(f"[{label} @ {tsec}s/move]")
        t = time.time()
        W, D, L = play_match(lambda b, e=eng: e.search(b)[0], games)
        report(f"{label}-t{tsec}", W, D, L)
        print(f"  [wall {time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
