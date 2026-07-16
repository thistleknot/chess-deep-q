"""Place the trained tower on a LADDER so we can see it below the SF floor.

Our other measures only play SF@1320, which is above this net, so they floor to a 920 sentinel and
tell us nothing. This plays the tower (net_search, default 1-ply value-greedy = the strong value
head) vs three rungs: random, 1-ply material/PST heuristic, and SF@1320. The random/heuristic rungs
have no calibrated Elo, so they report an Elo *difference* (ordinal); SF@1320 is anchored.
Answers 'do we have a model' -> beat random and the heuristic and it's a real player, sub-floor.

Usage: python measure_ladder.py [games] [depth]   e.g. python measure_ladder.py 20 1
"""
import sys, glob, math, random, time
import chess, chess.engine, torch

from chessdq.engine import pst_eval
from chessdq.resnet_model import ChessResNet, encode18
from chessdq import net_search


def load_net(dev):
    net = ChessResNet().to(dev)
    sd = torch.load("models/tower.pth", map_location=dev)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    net.eval()
    return net


def elo_diff(s):
    s = min(max(s, 1e-4), 1 - 1e-4)
    return 400 * math.log10(s / (1 - s))


def random_mover(b):
    return random.choice(list(b.legal_moves))


def heuristic_mover(b):
    """1-ply greedy over the mover-perspective PST eval of each child, mate preferred."""
    white = b.turn == chess.WHITE
    best, bv = None, None
    for m in b.legal_moves:
        b.push(m)
        v = 1e9 if b.is_checkmate() else pst_eval(b) * (1 if white else -1)
        b.pop()
        if bv is None or v > bv:
            bv, best = v, m
    return best


def adj_pst(b, net_white):
    """Adjudicate an unfinished game by the (neutral) PST eval; White-absolute centipawns."""
    cp = pst_eval(b)
    if abs(cp) <= 100:
        return 0.5
    return 1.0 if (cp > 0) == net_white else 0.0


def play(net_move, opp_move, games, adjudicate, cap=100):
    W = D = L = 0
    for g in range(games):
        nw = (g % 2 == 0)
        b = chess.Board()
        plies = 0
        while not b.is_game_over() and plies < cap:
            mv = net_move(b) if (b.turn == chess.WHITE) == nw else opp_move(b)
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv)
            plies += 1
        if b.is_checkmate():
            res = 1.0 if (b.turn == chess.BLACK) == nw else 0.0
        elif b.is_game_over():
            res = 0.5
        else:
            res = adjudicate(b, nw)
        W += res == 1.0
        D += res == 0.5
        L += res == 0.0
    return int(W), int(D), int(L)


def report(name, W, D, L, anchor=None):
    n = W + D + L
    s = (W + 0.5 * D) / n
    se = math.sqrt(max(s * (1 - s), 1e-9) / n)
    lo, hi = max(1e-4, s - 1.96 * se), min(1 - 1e-4, s + 1.96 * se)
    if anchor is not None:
        print(f"  {name:22s} {W}W-{D}D-{L}L  score {s:.2f}  -> ~{anchor + elo_diff(s):.0f} Elo "
              f"(CI {anchor + elo_diff(lo):.0f}..{anchor + elo_diff(hi):.0f})")
    else:
        print(f"  {name:22s} {W}W-{D}D-{L}L  score {s:.2f}  -> {elo_diff(s):+.0f} Elo vs opp "
              f"(ordinal; CI {elo_diff(lo):+.0f}..{elo_diff(hi):+.0f})")


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(dev)
    net_move = lambda b: net_search.best_move(b, net, dev, depth=depth, beam=8, encode_fn=encode18)
    print(f"Ladder: tower.pth net-minimax d{depth}, {games} games/rung (alternating colors)\n")
    t = time.time()
    report("vs random", *play(net_move, random_mover, games, adj_pst))
    report("vs heuristic-1ply", *play(net_move, heuristic_mover, games, adj_pst))
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if sfp:
        sf = chess.engine.SimpleEngine.popen_uci(sfp[0])
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
        lim = chess.engine.Limit(time=0.05)

        def sf_move(b):
            return sf.play(b, lim).move

        def adj_sf(b, nw):
            cp = sf.analyse(b, chess.engine.Limit(time=0.05))["score"].white().score(mate_score=100000)
            return 0.5 if (cp is None or abs(cp) <= 150) else (1.0 if (cp > 150) == nw else 0.0)

        report("vs SF@1320 (anchored)", *play(net_move, sf_move, games, adj_sf), anchor=1320)
        sf.quit()
    print(f"\n[wall {time.time()-t:.0f}s]  random/heuristic rungs are ordinal (no calibrated Elo)")


if __name__ == "__main__":
    main()
