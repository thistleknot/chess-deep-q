"""Learn a SMOOTH linear eval (ridge regression) on the hand features -> Stockfish cp.

Every non-linear learned eval (net, GBDT trees) failed because strong search exploits its holes. A
linear model has NO holes — it is smooth by construction, exactly like the hand heuristic (which is
just a hand-weighted linear combination). Ridge learns the OPTIMAL weights from SF labels instead of
hand-tuning. Inference is a dot product (µs). This is classical eval tuning (Texel), and it is the
right way to get a learned eval that survives alpha-beta.

Usage: python train_linear.py [alpha]
"""
import sys, json, time
import numpy as np
import chess
from sklearn.linear_model import Ridge

from gbdt_features import features

DATA = "data/distill_cp.jsonl"
OUT = "models/linear.npz"


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    X, y = [], []
    t = time.time()
    with open(DATA) as fh:
        for line in fh:
            try:
                fen, cp = json.loads(line)[:2]
            except Exception:
                continue
            X.append(features(chess.Board(fen)))
            y.append(max(-2000.0, min(2000.0, float(cp))))
    X = np.stack(X).astype(np.float32); y = np.array(y, dtype=np.float32)
    idx = np.random.RandomState(0).permutation(len(y))
    nval = min(4000, len(y) // 10); val, tr = idx[:nval], idx[nval:]

    model = Ridge(alpha=alpha).fit(X[tr], y[tr])
    pred = model.predict(X[val])
    rmse = float(np.sqrt(np.mean((pred - y[val]) ** 2)))
    sign = float(np.mean((pred > 0) == (y[val] > 0)))
    print(f"{len(y)} positions in {time.time()-t:.0f}s  ridge(alpha={alpha})  "
          f"val_rmse {rmse:.0f}cp  sign_acc {sign:.3f}")
    np.savez(OUT, coef=model.coef_.astype(np.float32), intercept=np.float32(model.intercept_))
    print(f"saved {OUT}")

    for name, fen in [("start", chess.STARTING_FEN),
                      ("white up a queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                      ("black up a queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")]:
        b = chess.Board(fen)
        cp = float(features(b) @ model.coef_ + model.intercept_)
        print(f"  {name}: {cp:+.0f} cp")


if __name__ == "__main__":
    main()
