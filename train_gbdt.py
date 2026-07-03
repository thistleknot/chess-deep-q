"""Distill the Stockfish-labelled dataset into an XGBoost value eval (CPU µs/node inference).

Now on dense hand-features (material + PST + castling; gbdt_features) regressing RAW CENTIPAWNS
(recovered from the stored tanh label), so material is linearly scaled — the earlier version
undervalued a queen 2.5x, which alpha-beta happily exploited into 0/30.

Usage: python train_gbdt.py [rounds] [max_depth]
"""
import os, sys, json, time
import numpy as np
import chess
import xgboost as xgb

from gbdt_features import features, cp_from_tanh

DATA_CP = "data/distill_cp.jsonl"     # raw centipawn labels (preferred)
DATA_TANH = "data/distill_sf.jsonl"   # squashed tanh(cp/400) labels (fallback, lossy)
OUT = "models/gbdt.json"
CP_CLIP = 2000.0


def load_xy():
    raw = os.path.exists(DATA_CP)
    src = DATA_CP if raw else DATA_TANH
    print(f"loading {'RAW cp' if raw else 'tanh (recovered)'} labels from {src}")
    X, y = [], []
    with open(src) as fh:
        for line in fh:
            try:
                fen, v = json.loads(line)[:2]
            except Exception:
                continue
            X.append(features(chess.Board(fen)))
            cp = max(-CP_CLIP, min(CP_CLIP, float(v))) if raw else cp_from_tanh(v)
            y.append(cp)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    t = time.time()
    X, y = load_xy()
    print(f"{len(y)} positions, features {X.shape} in {time.time()-t:.0f}s; cp target range "
          f"[{y.min():.0f},{y.max():.0f}] mean|cp| {np.abs(y).mean():.0f}")

    idx = np.random.RandomState(0).permutation(len(y))
    nval = min(4000, len(y) // 10)
    val, tr = idx[:nval], idx[nval:]
    dtr = xgb.DMatrix(X[tr], label=y[tr]); dval = xgb.DMatrix(X[val], label=y[val])
    params = {"max_depth": max_depth, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
              "objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist"}
    try:
        import torch
        params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        params["device"] = "cpu"

    t = time.time()
    bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dval, "val")],
                    verbose_eval=100, early_stopping_rounds=50)
    pred = bst.predict(dval); yval = y[val]
    rmse = float(np.sqrt(np.mean((pred - yval) ** 2)))
    sign = float(np.mean((pred > 0) == (yval > 0)))
    print(f"trained {bst.num_boosted_rounds()} rounds in {time.time()-t:.0f}s  "
          f"val_rmse {rmse:.0f}cp  sign_acc {sign:.3f}")
    bst.save_model(OUT)
    print(f"saved {OUT}")

    # Sanity: a queen must read ~+900, not +377.
    from gbdt_features import features as F
    for name, fen in [("start", chess.STARTING_FEN),
                      ("white up a queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")]:
        cp = float(bst.inplace_predict(F(chess.Board(fen))[None, :])[0])
        print(f"  {name}: {cp:+.0f} cp")


if __name__ == "__main__":
    main()
