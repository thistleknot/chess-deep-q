import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Phase A of :Search-teacher: (spec/search-distill.spec.md) — the :Distill-control:.

Expert-iteration control arm: refit the SAME amap-897 linear eval against its OWN deep-search
labels, and test whether one step of approximate policy iteration on the eval buys Elo. Purity-
compliant (own net, own search, own games — no external label source; spec/pathfind-population).

Pipeline (all checkpointed load-if-exists; heavy steps are native + threaded):
  1. GENERATE  rsearch4.play_games from champion weights, both sides, d=gen_depth, eps/tau
     exploration -> PV-leaf FENs (quiescence-resolved TDLeaf targets).
  2. LABEL     Searcher.search(fen, label_depth) -> White-tanh backed value (the DEEPER teacher;
     label_depth > gen_depth is the whole point). Cached to an npz keyed by (fen).
  3. FIT       ridge regression of amap-897 features -> atanh(label) (native eval is linear
     pre-tanh, so atanh-space is the correct target). Emits a champion-format ckpt.

A --probe run times labeling a small sample at d in {5,7} and extrapolates full-pass wall-clock
so label_depth can be chosen under the <=30 min budget BEFORE committing to a full labeling pass.

Usage:
  python distill_linear.py probe [n_games=200] [threads=6]
  python distill_linear.py run   [n_games=8000] [label_depth=5] [gen_depth=2] [threads=6] [ridge=1.0]

Output: models/champion_distillA.pt  (enc=amap, arch=linear; drop-in native mover).
Measure: python -m chessdq.head2head enc:amap:models/champion_distillA.pt enc:amap:models/champion.pt 50
Failure modes: champion.pt missing -> SystemExit; non-amap champion -> raw_weights asserts.
"""
import os
import sys
import time

import numpy as np
import torch

from chessdq.corpus_gen import raw_weights
from chessdq.encoders import get

CHAMPION = os.environ.get("DISTILL_SRC", "models/champion.pt")
OUT_CKPT = os.environ.get("DISTILL_OUT", "models/champion_distillA.pt")
CACHE    = os.environ.get("DISTILL_CACHE", "models/distillA_labels.npz")
PLY_CAP  = 160
SEED0    = 977
_ENC_FN, _NIN = get("amap")           # 897-dim; self-contained (pst planes + coverage map)


def _searcher(w, b):
    import importlib
    return importlib.import_module("rsearch4").Searcher(w, b)


def generate_leaves(w, b, n_games, gen_depth, threads, eps=0.15, tau=0.10):
    """Native threaded self-play (champion vs champion); return de-duplicated PV-leaf FENs.
    Terminal positions (no legal moves) are dropped: the searcher can't score them and the
    eval is never consulted at terminal nodes."""
    import importlib
    import chess
    rs = importlib.import_module("rsearch4")
    games = rs.play_games(w, b, w, b, gen_depth, n_games, threads, gen_depth,
                          eps, PLY_CAP, SEED0, tau)
    seen, fens = set(), []
    for _z, _agent_white, moves in games:
        for leaf_fen, _v, _pred in moves:
            if leaf_fen not in seen:
                seen.add(leaf_fen)
                if any(True for _ in chess.Board(leaf_fen).legal_moves):
                    fens.append(leaf_fen)
    return fens


_LBL = {}   # per-worker Searcher cache (weights are process-global via initializer)


def _lbl_init(w, b, d):
    _LBL["srch"], _LBL["d"] = _searcher(w, b), d


def _lbl_one(fen):
    return _LBL["srch"].search(fen, _LBL["d"])[1]     # white_value_tanh


def label(fens, w, b, label_depth, threads):
    """White-tanh backed value at label_depth for each FEN (the deeper teacher signal).
    Labeling is embarrassingly parallel — one Searcher per process (search releases the GIL
    natively, but a process pool sidesteps any Python-side per-call overhead)."""
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(threads, initializer=_lbl_init, initargs=(w, b, label_depth)) as pool:
        ys = np.array(pool.map(_lbl_one, fens, chunksize=64), dtype=np.float64)
    print(f"  labeled {len(fens)} at d{label_depth} in {(time.time()-t0)/60:.1f} min "
          f"({len(fens)/(time.time()-t0):.0f} pos/s, {threads} procs)", flush=True)
    return ys


def featurize(fens):
    """amap-897 feature matrix for a list of FENs (White-absolute, matches native eval space)."""
    import chess
    X = np.empty((len(fens), _NIN), dtype=np.float64)
    for i, fen in enumerate(fens):
        X[i] = _ENC_FN(chess.Board(fen)).astype(np.float64)
    return X


SAT = 0.90   # label saturation cut: self-play explored leaves include many won/lost (|value|->1)
             # positions; atanh(±1)->±3.8 dominates the L2 loss and blows the fitted eval scale
             # ~16x (norm 10.3 vs champion 0.65, corr +0.06 -> a broken, native-unplayable eval).
             # Dropping |label|>SAT keeps positional gradient only (data/h2h_verdict_distillA.md).


def fit_ridge(X, y, ridge):
    """Ridge solve w' = (X^T X + ridge*I)^-1 X^T y in atanh space; bias is the intercept column.
    Saturated labels (|y|>SAT) are dropped first — they carry no positional signal and break the
    eval scale. ridge default 100 lands the fitted norm near champion's (~0.65..0.78), which the
    native searcher's aspiration windows require."""
    keep = np.abs(y) <= SAT
    Xk, yk = X[keep], y[keep]
    yv = np.arctanh(np.clip(yk, -SAT, SAT))
    Xb = np.hstack([Xk, np.ones((Xk.shape[0], 1))])
    A = Xb.T @ Xb
    A[np.diag_indices_from(A)] += ridge
    coef = np.linalg.solve(A, Xb.T @ yv)
    w_new, b_new = coef[:-1], float(coef[-1])
    rms = float(np.sqrt(np.mean((Xb @ coef - yv) ** 2)))
    print(f"  dropped {(~keep).sum()}/{len(y)} saturated labels (|y|>{SAT})", flush=True)
    return w_new, b_new, rms


def save_ckpt(w_new, b_new, path, src_meta):
    sd = {"head.weight": torch.tensor(w_new, dtype=torch.float32).reshape(1, -1),
          "head.bias":   torch.tensor([b_new], dtype=torch.float32)}
    ck = {"state_dict": sd, "arch": "linear", "enc": "amap", "zca": False,
          "cum_games": int(src_meta.get("cum_games", 0)), "strength": 0.0,
          "rating": 0.0, "ts": 0, "distill": "search-teacher-A"}
    torch.save(ck, path)


def cmd_probe(argv):
    n_games = int(argv[0]) if argv else 200
    threads = int(argv[1]) if len(argv) > 1 else 6
    w, b = raw_weights(CHAMPION)
    print(f"probe: generating {n_games} games (d2) for leaf sample...", flush=True)
    fens = generate_leaves(w, b, n_games, 2, threads)
    sample = fens[:min(400, len(fens))]
    print(f"probe: {len(fens)} unique leaves; timing {len(sample)} labels per depth", flush=True)
    for d in (5, 7):
        srch = _searcher(w, b)
        t0 = time.time()
        for fen in sample:
            srch.search(fen, d)
        dt = time.time() - t0
        rate = len(sample) / dt
        # extrapolate an 8k-game full pass; leaves/game ~ len(fens)/n_games
        est_leaves = len(fens) / n_games * 8000
        print(f"  d{d}: {rate:.0f} pos/s single-thread -> ~{est_leaves/rate/60:.1f} min "
              f"single-thread for ~{est_leaves:.0f} leaves (labeling is single-threaded here)", flush=True)


def cmd_run(argv):
    n_games     = int(argv[0]) if argv else 8000
    label_depth = int(argv[1]) if len(argv) > 1 else 5
    gen_depth   = int(argv[2]) if len(argv) > 2 else 2
    threads     = int(argv[3]) if len(argv) > 3 else 6
    ridge       = float(argv[4]) if len(argv) > 4 else 100.0

    w, b = raw_weights(CHAMPION)
    src_meta = torch.load(CHAMPION, map_location="cpu")

    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        fens, y = list(z["fens"]), z["y"]
        print(f"loaded {len(fens)} cached labels from {CACHE}", flush=True)
    else:
        print(f"generating {n_games} games (d{gen_depth})...", flush=True)
        fens = generate_leaves(w, b, n_games, gen_depth, threads)
        print(f"{len(fens)} unique leaves; labeling at d{label_depth}...", flush=True)
        y = label(fens, w, b, label_depth, threads)
        np.savez_compressed(CACHE, fens=np.array(fens, dtype=object), y=y)
        print(f"cached labels -> {CACHE}", flush=True)

    print("featurizing + ridge fit...", flush=True)
    X = featurize(fens)
    w_new, b_new, rms = fit_ridge(X, y, ridge)
    print(f"fit RMS (atanh space) = {rms:.4f}  ||w_new||={np.linalg.norm(w_new):.3f}", flush=True)
    save_ckpt(w_new, b_new, OUT_CKPT, src_meta)
    print(f"saved {OUT_CKPT}", flush=True)
    print(f"measure: python -m chessdq.head2head enc:amap:{OUT_CKPT} enc:amap:{CHAMPION} 50", flush=True)


def main():
    if not os.path.exists(CHAMPION):
        raise SystemExit(f"{CHAMPION} not found")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    (cmd_probe if cmd == "probe" else cmd_run)(sys.argv[2:])


if __name__ == "__main__":
    main()
