import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Fair test: the trained net WITH its PUCT search (the actual agent) vs the ladder — not the
handicapped d1 net-minimax probe. Argmax over root visits, no Dirichlet (a :Measurement-game:)."""
import sys
import numpy as np
import torch

from chessdq.puct_selfplay import puct_move
from chessdq.resnet_model import ChessResNet
from chessdq.measure_ladder import random_mover, heuristic_mover, adj_pst, play, elo_diff


def main():
    playouts = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessResNet().to(dev)
    ck = torch.load("models/tower_puct.pt", map_location=dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    print(f"net+PUCT({playouts}) vs ladder, {games} games/rung\n", flush=True)
    mv = lambda b: puct_move(b, net, dev, playouts)
    for name, opp in (("random", random_mover), ("heuristic-1ply", heuristic_mover)):
        W, D, L = play(mv, opp, games, adj_pst)
        s = (W + 0.5 * D) / (W + D + L)
        print(f"net+PUCT({playouts}) vs {name:14s}: {W}W-{D}D-{L}L  score {s:.2f}  "
              f"elo_diff {elo_diff(s):+.0f}", flush=True)


if __name__ == "__main__":
    main()
