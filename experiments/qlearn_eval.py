import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Strength measurement for the Merge 2 Q-learning agent (spec/q-learning.spec.md :Proxy-strength: /
:Calibrated-elo:). Thin wrapper over measure_ladder — no new Elo math.

  proxy_strength(move_fn, games) -> (wr_random, wr_pst)   cheap, every iteration
  calibrated_elo(move_fn, games) -> elo or None           slow (needs Stockfish), periodic

move_fn is any board->move callable (the greedy Q-agent). The calibrated anchor is Stockfish limited to
UCI_Elo=1320; beating it with score s gives 1320 + elo_diff(s), elo_diff(s)=400*log10(s/(1-s)).
"""
import glob

import chess
import chess.engine

from chessdq.measure_ladder import play, elo_diff, random_mover, heuristic_mover, adj_pst


def _score(wdl):
    w, d, l = wdl
    n = w + d + l
    return (w + 0.5 * d) / n if n else 0.0


def proxy_strength(move_fn, games=30):
    """Greedy-agent win-rate vs uniform-random and vs the 1-ply PST mover. Instant, no Stockfish."""
    wr_random = _score(play(move_fn, random_mover, games, adj_pst))
    wr_pst = _score(play(move_fn, heuristic_mover, games, adj_pst))
    return wr_random, wr_pst


def _find_sf():
    hits = glob.glob("engines/**/stockfish*.exe", recursive=True)
    return hits[0] if hits else None


def calibrated_elo(move_fn, games=20, anchor=1320):
    """Calibrated Elo = anchor + elo_diff(score vs Stockfish@anchor). None if no Stockfish binary."""
    sfp = _find_sf()
    if sfp is None:
        return None
    sf = chess.engine.SimpleEngine.popen_uci(sfp[0] if isinstance(sfp, list) else sfp)
    try:
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": anchor})
        lim = chess.engine.Limit(time=0.05)

        def sf_move(b):
            return sf.play(b, lim).move

        def adj_sf(b, nw):
            cp = sf.analyse(b, lim)["score"].white().score(mate_score=100000)
            return 0.5 if (cp is None or abs(cp) <= 150) else (1.0 if (cp > 150) == nw else 0.0)

        s = _score(play(move_fn, sf_move, games, adj_sf))
        return anchor + elo_diff(s)
    finally:
        sf.quit()
