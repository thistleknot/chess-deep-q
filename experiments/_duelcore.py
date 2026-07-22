"""Torch-free pool-worker core for native duels (spawn re-imports ONLY this module for the
workers, so no torch/CUDA DLLs load in workers -> no Windows paging exhaustion, WinError 1455).
Imports limited to rsearch4/chess/random."""
import importlib
import random

import chess

_DUEL = {}


def duel_init(wa, ba, wb, bb, depth):
    rs = importlib.import_module("rsearch4")
    _DUEL["SA"], _DUEL["SB"], _DUEL["d"] = rs.Searcher(wa, ba), rs.Searcher(wb, bb), depth


def duel_game(args):
    """One deterministic game from a seeded 4-ply opening; A-perspective result (1/0.5/0)."""
    i, a_white = args
    SA, SB, d = _DUEL["SA"], _DUEL["SB"], _DUEL["d"]
    rng = random.Random(1000 + i)
    b = chess.Board()
    for _ in range(4):
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(rng.choice(ms))
    while not b.is_game_over(claim_draw=True) and b.fullmove_number < 120:
        S = SA if ((b.turn == chess.WHITE) == a_white) else SB
        b.push(chess.Move.from_uci(S.search(b.fen(), d)[0]))
    r = b.result(claim_draw=True)
    w = 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
    return w if a_white else 1 - w
