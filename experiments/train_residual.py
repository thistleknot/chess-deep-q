import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Train a GBDT to predict the RESIDUAL (SF_cp - pst_eval), for a hybrid eval = heuristic + lambda*residual.

Every learned-eval attempt failed because strong search exploits the eval's noise/holes. A hand
heuristic has none (it's smooth), which is why it reaches 1672. So instead of replacing it, LEARN A
CORRECTION on top of it: the smooth heuristic stays the base (search can't be exploited), and a
small learned residual nudges toward Stockfish. This is how NNUE was first deployed. If it beats the
pure heuristic (1367/1672), we have a LEARNED model above 1600.

Usage: python train_residual.py [rounds] [max_depth]
"""
import sys, json, time
import numpy as np
import chess
import xgboost as xgb

from chessdq.engine import pst_eval
from chessdq.gbdt_features import features

DATA = "data/distill_cp.jsonl"
OUT = "models/gbdt_residual.json"


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    X, y = [], []
    t = time.time()
    with open(DATA) as fh:
        for line in fh:
            try:
                fen, cp = json.loads(line)[:2]
            except Exception:
                continue
            b = chess.Board(fen)
            X.append(features(b))
            cp_c = max(-2000.0, min(2000.0, float(cp)))   # clip SF mate scores (±100000) before residual
            y.append(cp_c - pst_eval(b))   # residual: what the heuristic misses
    X = np.stack(X).astype(np.float32); y = np.array(y, dtype=np.float32)
    print(f"{len(y)} positions, residual range [{y.min():.0f},{y.max():.0f}] "
          f"mean|resid| {np.abs(y).mean():.0f}cp in {time.time()-t:.0f}s")

    idx = np.random.RandomState(0).permutation(len(y))
    nval = min(4000, len(y) // 10); val, tr = idx[:nval], idx[nval:]
    dtr = xgb.DMatrix(X[tr], label=y[tr]); dval = xgb.DMatrix(X[val], label=y[val])
    params = {"max_depth": max_depth, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
              "objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist"}
    try:
        import torch
        params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        params["device"] = "cpu"
    bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dval, "val")],
                    verbose_eval=200, early_stopping_rounds=50)
    pred = bst.predict(dval)
    rmse = float(np.sqrt(np.mean((pred - y[val]) ** 2)))
    print(f"residual val_rmse {rmse:.0f}cp (vs raw-cp eval rmse ~289 — smaller residual = heuristic already good)")
    bst.save_model(OUT); print(f"saved {OUT}")


if __name__ == "__main__":
    main()
