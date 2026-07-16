import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Does the RICHER hand eval (fast_evaluate_position: mobility/king-safety/structure) beat the
simple pst_eval (material+PST, ~1367/1672) in the same alpha-beta? If yes, feature richness is the
lever a learned eval was missing — and beating the heuristic is hours (richer features), not weeks.

Usage: python measure_richheur.py [games] [depths]
"""
import sys
import chess

from chessdq.engine import AlphaBetaEngine
from chessdq.evaluation import fast_evaluate_position
from chessdq.measure_gbdt import play_match, report


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    depths = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['2', '3'])]

    ev = lambda b: fast_evaluate_position(b) * 100.0   # rich hand eval -> centipawns, White-absolute

    for d in depths:
        eng = AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=d)
        print(f"[rich hand eval (fast_evaluate_position) in alpha-beta, depth {d}]")
        W, D, L = play_match(lambda b, e=eng: e.search(b)[0], games)
        report(f'richheur-d{d}', W, D, L)


if __name__ == "__main__":
    main()
