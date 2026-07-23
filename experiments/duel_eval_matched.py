import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""P2: matched-search-depth eval duel — halfKP NNUE vs the linear champion in the SAME Python
AlphaBetaEngine at EQUAL max_depth, isolating EVAL QUALITY (search is identical, only the eval_fn
differs). Deterministic diverse-opening games, colors alternated. Answers whether nonlinear capacity
extracts more from the search-distilled signal than the linear eval — independent of label depth (P1).

Usage: EVAL_DUEL_DEPTH=4 EVAL_DUEL_N=30 python duel_eval_matched.py
"""
import os
import random
from math import sqrt

import numpy as np
import chess
import torch

from chessdq.corpus_gen import raw_weights
from chessdq.encoders import get
from chessdq.engine import AlphaBetaEngine
from chessdq.nnue_model import load_nnue, make_incremental_nnue_eval

DEPTH = int(os.environ.get("EVAL_DUEL_DEPTH", "4"))
N     = int(os.environ.get("EVAL_DUEL_N", "30"))


def _elo(x):
    x = min(max(x, 1e-6), 1 - 1e-6)
    return -400 * np.log10(1 / x - 1)


def main():
    dev = torch.device("cpu")
    w, b = raw_weights("models/champion.pt")
    w = np.asarray(w, dtype=np.float64)
    enc = get("amap")[0]

    def lin_eval(board):
        return 800.0 * float(w @ enc(board).astype(np.float64) + b)   # centipawn-ish, white-absolute

    net = load_nnue("models/nnue_own.pt", dev)
    nnue_eval = make_incremental_nnue_eval(net, dev)                  # white-absolute cp, fast numpy head

    eng_lin = AlphaBetaEngine(eval_fn=lin_eval, time_limit=600, max_depth=DEPTH)   # depth-bounded, no clock
    eng_nn  = AlphaBetaEngine(eval_fn=nnue_eval, time_limit=600, max_depth=DEPTH)

    def play(opens, nnue_white):
        bd = chess.Board()
        for m in opens:
            bd.push(m)
        while not bd.is_game_over(claim_draw=True) and bd.fullmove_number < 120:
            eng = eng_nn if ((bd.turn == chess.WHITE) == nnue_white) else eng_lin
            bd.push(eng.search(bd)[0])
        r = bd.result(claim_draw=True)
        wv = 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
        return wv if nnue_white else 1 - wv

    rng = random.Random(77)
    s = 0.0
    for i in range(N):
        bb = chess.Board(); opens = []
        for _ in range(4):
            ms = list(bb.legal_moves)
            if not ms:
                break
            mv = rng.choice(ms); opens.append(mv); bb.push(mv)
        s += play(opens, i % 2 == 0)
        print(f"  game {i+1}/{N}: NNUE running {s:.1f}", flush=True)
    p = s / N
    zc = 1.96; den = 1 + zc * zc / N
    c = (p + zc * zc / 2 / N) / den
    h = zc * sqrt(p * (1 - p) / N + zc * zc / 4 / N / N) / den
    lo, hi = c - h, c + h
    print(f"\nP2 matched-d{DEPTH} ({N}g): NNUE(halfKP) vs linear champion = {p:.3f} "
          f"[{lo:.3f}..{hi:.3f}] Elo {_elo(p):+.0f} band-excludes-0.5={lo>0.5 or hi<0.5}", flush=True)


if __name__ == "__main__":
    main()
