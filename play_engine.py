"""Play a human vs the alpha-beta engine in the terminal. KISS, no GPU needed.

Difficulty is search time per move -- the reliable strength knob for a search engine (far more
predictable than net temperature). The ELO labels are ROUGH estimates for a pure-Python
alpha-beta with PST eval + quiescence + null-move + LMR; treat them as ballpark, not measured.
"""

import sys
import chess

from engine import AlphaBetaEngine

# Rough, honest ballpark: search time -> approximate strength for this engine.
LEVELS = {
    "1": (0.1, "~1200 (very fast)"),
    "2": (0.5, "~1500"),
    "3": (1.5, "~1750"),
    "4": (3.0, "~1900"),
    "5": (6.0, "~2050 (slow)"),
}


def main():
    print("Human vs alpha-beta engine (pure-Python, laptop-friendly).")
    for k, (t, label) in LEVELS.items():
        print(f"  {k}. {t:>4}s/move  {label} ELO")
    lvl = input("Difficulty (1-5, default 3): ").strip() or "3"
    time_limit = LEVELS.get(lvl, LEVELS["3"])[0]
    human_white = (input("Play as white? (y/n, default y): ").strip().lower() or "y") == "y"

    board = chess.Board()
    engine = AlphaBetaEngine(time_limit=time_limit)

    while not board.is_game_over():
        print("\n" + str(board))
        if (board.turn == chess.WHITE) == human_white:
            raw = input("Your move (SAN or UCI, e.g. Nf3 / g1f3, 'quit'): ").strip()
            if raw in ("quit", "q"):
                return
            try:
                move = board.parse_san(raw)
            except ValueError:
                try:
                    move = chess.Move.from_uci(raw)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Illegal / unparseable move; try again.")
                    continue
            board.push(move)
        else:
            print("Engine thinking...")
            move, info = engine.search(board)
            print(f"Engine: {board.san(move)}  (depth {info['depth']}, "
                  f"score {info['score_cp']/100:+.2f}, {info['nps']:,.0f} nps)")
            board.push(move)

    print("\n" + str(board))
    print("Result:", board.result(), "-", board.outcome().termination.name)


if __name__ == "__main__":
    main()
