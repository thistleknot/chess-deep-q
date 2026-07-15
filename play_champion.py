"""Play a human vs the CANONICAL RL champion — the self-play-trained graduated net
(models/champion.pt, kc-809 linear + ZCA lineage) inside the native rsearch alpha-beta.

This is the net whose measured ladder reads: crown 40.83, d7 60-game rung 1561
(1466..1704) vs SF@1320, 200-game claims run on the ledger. Difficulty = search depth
(the measured strength knob): the claims numbers are at depth 7.

Usage: python play_champion.py [depth]   (default 9 — the 1670-claims engine, ~1.5s/move;
pass 7 for a faster ~1572-class opponent)
Failure modes: missing/incompatible ckpt or ZCA identity failure -> SystemExit
(corpus_gen.raw_weights refuses an unsafe whitening conversion).
"""
import importlib
import sys

import chess

from corpus_gen import raw_weights

CKPT = "models/champion.pt"


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    w, b = raw_weights(CKPT)
    srch = importlib.import_module("rsearch4").Searcher(w, b)
    human_white = (input("Play as white? (y/n, default y): ").strip().lower() or "y") == "y"
    print(f"champion (self-play RL net + native alpha-beta d{depth}). "
          f"Moves as SAN (Nf3) or UCI (g1f3); 'quit' to exit.")
    board = chess.Board()
    while not board.is_game_over():
        print("\n" + str(board))
        if (board.turn == chess.WHITE) == human_white:
            raw = input("Your move: ").strip()
            if raw in ("quit", "q"):
                return
            try:
                mv = board.parse_san(raw)
            except ValueError:
                try:
                    mv = chess.Move.from_uci(raw)
                    if mv not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Illegal / unparseable; try again.")
                    continue
        else:
            uci, val, _leaf, _pred, nodes = srch.search(board.fen(), depth)
            mv = chess.Move.from_uci(uci)
            print(f"champion plays: {board.san(mv)}   (value {val:+.3f}, {nodes} nodes)")
        board.push(mv)
    print("\n" + str(board))
    print("Result:", board.result())


if __name__ == "__main__":
    main()
