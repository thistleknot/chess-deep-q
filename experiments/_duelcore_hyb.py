"""Torch-free pool-worker core for the native hyb-vs-d9 budget-matched duel (mirrors
_duelcore.py's pattern exactly — spawn re-imports ONLY this module, no torch in workers)."""
import importlib
import random

import chess

_DUEL = {}


def duel_init(w, b, sims, lb, c_puct, t_prior, leaf_depth, tau, d9_depth):
    rs = importlib.import_module("rsearch4")
    _DUEL["HYB"] = rs.HybSearcher(list(w), b, 0)
    _DUEL["D9"] = rs.Searcher(list(w), b)
    _DUEL["sims"], _DUEL["lb"], _DUEL["c"], _DUEL["tp"] = sims, lb, c_puct, t_prior
    _DUEL["leaf_depth"], _DUEL["tau"], _DUEL["d9_depth"] = leaf_depth, tau, d9_depth


def duel_game(args):
    """One deterministic game from a seeded 4-ply opening; hyb-perspective result (1/0.5/0)."""
    i, hyb_white = args
    d = _DUEL
    rng = random.Random(1000 + i)
    b = chess.Board()
    for _ in range(4):
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(rng.choice(ms))
    while not b.is_game_over(claim_draw=True) and b.fullmove_number < 120:
        if (b.turn == chess.WHITE) == hyb_white:
            mv = d["HYB"].choose(b.fen(), d["sims"], d["lb"], d["c"], d["tp"],
                                  d["leaf_depth"], d["tau"])[0]
        else:
            mv = d["D9"].search(b.fen(), d["d9_depth"])[0]
        b.push(chess.Move.from_uci(mv))
    r = b.result(claim_draw=True)
    w = 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
    return w if hyb_white else 1 - w
