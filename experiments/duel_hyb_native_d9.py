import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Budget-matched native head-to-head: hyb (rsearch4.HybSearcher, lb=0.5/leaf=ab2, sims
calibrated to match the champion's own d9 per-move wall-clock) vs the champion's own native
d9 mover (rsearch4.Searcher.search(fen, 9)) — the deployment-truth ruler per
experiments/distill_iterate.py's `duel()` docstring, extended to the search-arms hyb question
(spec/search-arms.spec.md :Tournament-and-verdict:, dispositioned.md 2026-07-25 native-hyb entry).

Usage: python -m experiments.duel_hyb_native_d9 [n_games] [sims] [threads]
"""
import os
import sys
from math import sqrt

import numpy as np

from chessdq.corpus_gen import raw_weights


def _elo(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return -400 * np.log10(1 / x - 1)


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    import multiprocessing as mp
    from experiments._duelcore_hyb import duel_init, duel_game

    w, b = raw_weights("models/champion.pt")
    w = list(w)
    jobs = [(i, i % 2 == 0) for i in range(n_games)]
    with mp.Pool(threads, initializer=duel_init,
                 initargs=(w, b, sims, 0.5, 1.5, 0.2, 2, 0.0, 9)) as pool:
        res = pool.map(duel_game, jobs)
    p = sum(res) / n_games
    zc = 1.96
    den = 1 + zc * zc / n_games
    c = (p + zc * zc / 2 / n_games) / den
    h = zc * sqrt(p * (1 - p) / n_games + zc * zc / 4 / n_games / n_games) / den
    lo, hi = c - h, c + h
    excl = lo > 0.5 or hi < 0.5
    print(f"budget-matched native hyb(sims={sims},lb=0.5,ab2) vs champion-native-d9 "
          f"({n_games}g): hyb score {p:.3f} [{lo:.3f}..{hi:.3f}] Elo {_elo(p):+.0f} "
          f"band-excludes-0.5={excl}", flush=True)


if __name__ == "__main__":
    main()
