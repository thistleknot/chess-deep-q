import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""P5: Random-Bar / shadow feature audit — which of the 897 amap features carry signal ABOVE a
noise null, against the search-distilled targets (data/distill_own_d5.jsonl). A feature counts only
if it out-importances a random control (the same "band excludes the null" discipline, applied to
features). Diagnostic: a leaner eval / a priority list for a halfKP port — NOT a strength change.

Method: featurize positions with amap (897), append K shadow columns (shuffled copies of real
features — a Boruta-style null that matches feature distributions, more honest than one Gaussian bar),
fit a gradient-boosted regressor on atanh(search-value), read gain importances. The null bar = the
MAX shadow importance (Boruta's "beat the best shadow"). Features above it survive.

Correlation caveat (docs/LESSONS.md): tree gain-importance splits credit among correlated features,
so a SURVIVED feature is load-bearing, but a DROPPED one is "redundant given the others," not useless.

Usage: FEAT_AUDIT_N=8000 python feature_audit_randombar.py
"""
import os
import json

import numpy as np
import chess

from chessdq.encoders import get

N       = int(os.environ.get("FEAT_AUDIT_N", "8000"))
KSHADOW = int(os.environ.get("FEAT_AUDIT_SHADOW", "20"))
LABELS  = os.environ.get("FEAT_AUDIT_LABELS", "data/distill_own_d5.jsonl")


def main():
    enc, nin = get("amap")
    rows = []
    with open(LABELS) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = np.random.RandomState(0)
    if len(rows) > N:
        rows = [rows[i] for i in rng.choice(len(rows), N, replace=False)]
    X = np.empty((len(rows), nin), dtype=np.float32)
    y = np.empty(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        X[i] = enc(chess.Board(r["fen"]))
        y[i] = np.arctanh(np.clip(float(r["value"]), -0.999, 0.999))
    print(f"loaded {len(rows)} positions, {nin} amap features", flush=True)

    # shadow columns: shuffled copies of randomly-chosen real features (distribution-matched null)
    sh_src = rng.choice(nin, KSHADOW, replace=True)
    SH = np.empty((len(rows), KSHADOW), dtype=np.float32)
    for j, src in enumerate(sh_src):
        SH[:, j] = X[rng.permutation(len(rows)), src]
    Xa = np.hstack([X, SH])

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=200, max_depth=6, n_jobs=2, subsample=0.8,
                             tree_method="hist", objective="reg:squarederror")
        model.fit(Xa, y)
        imp = model.feature_importances_
        which = "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        model = HistGradientBoostingRegressor(max_iter=200, max_depth=6)
        model.fit(Xa, y)
        r = permutation_importance(model, Xa, y, n_repeats=3, n_jobs=2, random_state=0)
        imp = r.importances_mean
        which = "sklearn-permutation"

    real_imp, shadow_imp = imp[:nin], imp[nin:]
    bar = float(shadow_imp.max())                      # Boruta null: beat the best shadow
    survivors = int((real_imp > bar).sum())
    order = np.argsort(real_imp)[::-1]
    print(f"\nP5 feature audit ({which}): null bar (max shadow importance) = {bar:.5f}", flush=True)
    print(f"  {survivors}/{nin} amap features beat the null; "
          f"{nin - survivors} indistinguishable from noise (redundant-or-dead).", flush=True)
    print(f"  top-15 features by importance: {[int(i) for i in order[:15]]}", flush=True)
    print(f"  amap layout: [0:769]=pst planes, [769:897]=attack-coverage map (128).", flush=True)
    top_in_cov = int((order[:survivors] >= 769).sum()) if survivors else 0
    print(f"  of {survivors} survivors, {top_in_cov} are coverage-map features (>=769), "
          f"{survivors - top_in_cov} are pst-plane features.", flush=True)


if __name__ == "__main__":
    main()
