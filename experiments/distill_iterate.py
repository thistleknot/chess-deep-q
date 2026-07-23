import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Expert-iteration loop for :Search-teacher: Phase A (spec/search-distill.spec.md).

Approximate policy iteration on the eval: each round labels fresh self-play positions with the
PREVIOUS round's eval + depth-d native minimax, refits the amap-897 linear, and duels the new eval
vs the previous one on the CONTROLLED head2head ruler (the native play_games generator is NOT a fair
duel — it handicaps the exploring side; data/h2h_verdict_distillA.md). The loop stops at the fixed
point: the first round whose gain-band no longer excludes zero (eval@d ~= eval, no more to distill).

Round k: generate from ckpt_{k-1} -> label at label_depth -> fit (drop saturated + ridge) -> ckpt_k
         -> head2head(ckpt_k vs ckpt_{k-1}, 50g). CONTINUE iff A-score band excludes zero AND A wins.

Usage: python distill_iterate.py [seed_ckpt=models/champion_distillA2.pt] [rounds=4]
                                  [n_games=4000] [label_depth=5] [gen_depth=2] [threads=6]
Emits models/champion_distill_r<k>.pt each round; the last confirmed winner is the fixed-point champion.
"""
import importlib
import random
import sys
from math import sqrt

import chess
import numpy as np
import torch

from chessdq.corpus_gen import raw_weights
from experiments.distill_linear import generate_leaves, label, featurize, fit_ridge, save_ckpt

DUEL_DEPTH = int(__import__("os").environ.get("DISTILL_DUEL_DEPTH", "6"))
DUEL_N     = int(__import__("os").environ.get("DISTILL_DUEL_N", "30"))


def _elo(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return -400 * np.log10(1 / x - 1)


from experiments._duelcore import duel_init as _duel_init, duel_game as _duel_game  # torch-free workers


def duel(a_ckpt, b_ckpt, games, tag, threads=6):
    """Deployment-faithful PARALLEL native duel: A vs B, each side its OWN rsearch4.Searcher at
    DUEL_DEPTH, DETERMINISTIC play from `games` diverse 4-ply openings, colors alternated. The ONLY
    fair native ruler — head2head is a d2-beam lens that does NOT track native-d9 deployment
    (distillA2 +798 d2-beam yet 1917 vs 1878 champion native-d9, a tie), and play_games handicaps
    the exploring side. Return (a_wins_and_band_excludes_0.5, verdict_line)."""
    import multiprocessing as mp
    wa, ba = raw_weights(a_ckpt); wb, bb = raw_weights(b_ckpt)   # parent loads weights ONCE (torch here only)
    wa, wb = list(wa), list(wb)
    N = games
    jobs = [(i, i % 2 == 0) for i in range(N)]
    with mp.Pool(threads, initializer=_duel_init, initargs=(wa, ba, wb, bb, DUEL_DEPTH)) as pool:
        res = pool.map(_duel_game, jobs)
    p = sum(res) / N
    zc = 1.96; den = 1 + zc * zc / N
    c = (p + zc * zc / 2 / N) / den
    h = zc * sqrt(p * (1 - p) / N + zc * zc / 4 / N / N) / den
    lo, hi = c - h, c + h
    excl = lo > 0.5 or hi < 0.5
    line = (f"native-d{DUEL_DEPTH} {tag}: A={a_ckpt.split('/')[-1]} score {p:.3f} "
            f"[{lo:.3f}..{hi:.3f}] Elo {_elo(p):+.0f} band-excludes-0.5={excl}")
    return (p > 0.5 and excl), line


def main():
    seed   = sys.argv[1] if len(sys.argv) > 1 else "models/champion_distillA2.pt"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ngames = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    ld     = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    gd     = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    threads= int(sys.argv[6]) if len(sys.argv) > 6 else 6

    prev = seed
    src_meta = torch.load(seed, map_location="cpu")
    for k in range(2, 2 + rounds):
        w, b = raw_weights(prev)
        print(f"\n=== round {k}: generate from {prev} (d{gd}) ===", flush=True)
        fens = generate_leaves(w, b, ngames, gd, threads)
        print(f"round {k}: {len(fens)} leaves; labeling d{ld}...", flush=True)
        y = label(fens, w, b, ld, threads)
        X = featurize(fens)
        w_new, b_new, rms = fit_ridge(X, y, 100.0)
        out = f"models/champion_distill_r{k}.pt"
        save_ckpt(w_new, b_new, out, src_meta)
        print(f"round {k}: saved {out} (fit atanh-RMS {rms:.3f}, "
              f"||w||={np.linalg.norm(w_new):.3f})", flush=True)
        gained, line = duel(out, prev, DUEL_N, f"distill_r{k}")
        print(f"round {k} DUEL vs {os.path.basename(prev)}: {line}", flush=True)
        if not gained:
            print(f"\nFIXED POINT at round {k}: gain-band does not exclude zero -> "
                  f"champion is {os.path.basename(prev)}", flush=True)
            print(f"FINAL_CHAMPION={prev}", flush=True)
            return
        prev = out
    print(f"\nrounds exhausted; still gaining -> champion is {os.path.basename(prev)}", flush=True)
    print(f"FINAL_CHAMPION={prev}", flush=True)


if __name__ == "__main__":
    main()
