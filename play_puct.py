"""Human vs the trained net+PUCT agent (models/tower_puct.pt) — the RL deliverable.

The agent is the dual-head state-value/policy net inside its PUCT search: action selection is the
argmax over root visit counts, i.e. afterstate action-values reconstructed by search (spec:
rl-categorization :Afterstate-action-value:). This is the AlphaStar-hybrid RL agent, not the
classical alpha-beta engine (that is `play_engine.py`, the non-RL baseline).

Usage: python play_puct.py [playouts]
"""
import sys

import chess
import torch

from resnet_model import ChessResNet
from puct_selfplay import puct_move


def main():
    playouts = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessResNet().to(dev)
    try:
        ck = torch.load("models/tower_puct.pt", map_location=dev)
    except FileNotFoundError:
        print("models/tower_puct.pt not found — train it first: python puct_selfplay.py")
        return
    net.load_state_dict(ck["state_dict"]); net.eval()
    human_white = (input("Play as white? (y/n, default y): ").strip().lower() or "y") == "y"
    print(f"net+PUCT({playouts}). Moves as SAN (Nf3) or UCI (g1f3); 'quit' to exit.")
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
            board.push(mv)
        else:
            print("net+PUCT thinking...")
            mv = puct_move(board, net, dev, playouts)
            if mv is None:
                break
            print(f"agent plays: {board.san(mv)}")
            board.push(mv)
    print("\n" + str(board))
    print("Result:", board.result())


if __name__ == "__main__":
    main()
