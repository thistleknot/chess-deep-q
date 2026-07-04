"""Measure the trained tower (models/tower.pth) via net_search at a chosen depth vs Stockfish anchor.

Isolates the SEARCH-DEPTH lever now that the value net is trained: same net, vary depth, watch Elo.
Usage: python measure_tower.py [depth] [games] [sf_elo] [beam] [sf_ms] [cap]
"""
import sys, glob, time
import chess, chess.engine
import torch

from resnet_model import ChessResNet, encode18
from elo_calibration import score_to_elo_diff
import net_search


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    sf_elo = int(sys.argv[3]) if len(sys.argv) > 3 else 1320
    q = int(sys.argv[4]) if len(sys.argv) > 4 else 8       # quiescence max plies (0 = off)
    beam = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    sf_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 50
    cap = int(sys.argv[7]) if len(sys.argv) > 7 else 90

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessResNet().to(device)
    net.load_state_dict(torch.load("models/tower.pth", map_location=device)["state_dict"])
    net.eval()

    sf = chess.engine.SimpleEngine.popen_uci(glob.glob("engines/**/stockfish*.exe", recursive=True)[0])
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    lim = chess.engine.Limit(time=sf_ms / 1000.0)

    print(f"tower net-minimax d{depth} q{q} beam{beam} vs SF@{sf_elo}, {games} games...")
    total = 0.0
    start = time.time()
    for g in range(games):
        nw = (g % 2 == 0)
        b = chess.Board(); plies = 0
        t_net = 0.0
        while not b.is_game_over() and plies < cap:
            if (b.turn == chess.WHITE) == nw:
                t0 = time.time()
                mv = net_search.best_move(b, net, device, depth=depth, beam=beam, encode_fn=encode18, q=q)
                t_net += time.time() - t0
            else:
                mv = sf.play(b, lim).move
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv); plies += 1
        if b.is_checkmate():
            ww = b.turn == chess.BLACK; pts = 1.0 if ww == nw else 0.0; res = "1-0" if ww else "0-1"
        elif b.is_game_over():
            pts, res = 0.5, b.result()
        else:
            cp = sf.analyse(b, chess.engine.Limit(time=0.05))["score"].white().score(mate_score=100000)
            if cp is None or abs(cp) <= 150:
                pts, res = 0.5, "adj-draw"
            else:
                w = cp > 150; pts, res = (1.0 if w == nw else 0.0), ("adj-1-0" if w else "adj-0-1")
        total += pts
        print(f"  game {g+1}/{games}: net={'W' if nw else 'B'} {res:9s} pts={pts:.1f} plies={plies} "
              f"net_move_avg={t_net/max(plies//2,1):.2f}s")
    sf.quit()
    score = total / games
    elo = sf_elo + (score_to_elo_diff(score) if 0 < score < 1 else (400 if score >= 1 else -400))
    print(f"\nscore {total:.1f}/{games} = {score:.2f} -> ESTIMATED Elo ~{elo:.0f}  [wall {time.time()-start:.0f}s]")


if __name__ == "__main__":
    main()
