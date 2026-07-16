import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Contract tests for evaluate_by_player (spec/terminal-interface.spec.md test reqs).

Pins the :Player-score-breakdown: invariants:
  1. material + position == total, per side.
  2. white['total'] - black['total'] == fast_evaluate_position(board), for
     positions not in check (the breakdown omits the turn-dependent check bonus).

Run: python test_player_scores.py
"""

import chess
from chessdq.evaluation import evaluate_by_player, fast_evaluate_position

# Positions not in check: startpos and a couple of mid-game positions.
POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
    "r3k2r/pp1n1ppp/2p5/8/8/2P2N2/PP3PPP/R3K2R w KQkq - 0 1",
    "8/5k2/8/8/3Q4/8/5K2/8 w - - 0 1",
]


def approx(a, b, eps=1e-6):
    return abs(a - b) < eps


def main():
    for fen in POSITIONS:
        board = chess.Board(fen)
        assert not board.is_check(), f"test position must not be in check: {fen}"
        scores = evaluate_by_player(board)

        # 1. per-side material + position == total
        for side in ("white", "black"):
            s = scores[side]
            assert approx(s["material"] + s["position"], s["total"]), \
                f"{side} material+position != total for {fen}"

        # 2. white total - black total reconstructs the fast evaluator
        diff = scores["white"]["total"] - scores["black"]["total"]
        ev = fast_evaluate_position(board)
        assert approx(diff, ev), \
            f"per-player diff {diff:.6f} != eval {ev:.6f} for {fen}"

    print(f"OK: {len(POSITIONS)} positions pass the player-score-breakdown invariants.")


if __name__ == "__main__":
    main()
