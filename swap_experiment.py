"""Localize the constraint (eval vs search) with adequate power + binomial CIs.

The n=6 table distinguished nothing (score SD ~0.13 > the gaps). This runs the eval<->search 2x2
at higher game counts and reports 95% CIs so a difference is real, not noise.

Cells (all vs SF@1320):
  ladder : hand HEURISTIC eval in a real alpha-beta+quiescence at fixed depths 1..3 — if depth-2
           already ~1000, shallow search is the cap (H2); if ~1400, eval matters (net eval is weak).
  cellA  : hand heuristic eval inside net_search depth-2 (same shallow search, swapped eval).
  net    : the learned net inside net_search depth-2 (powered baseline of the diagonal we have).

Usage: python swap_experiment.py <ladder|cellA|net> [games]
"""
import sys, glob, time, math
import chess, chess.engine
import torch

from engine import AlphaBetaEngine, pst_eval
import net_search

SF_ANCHOR = 1320


def sf_engine():
    e = chess.engine.SimpleEngine.popen_uci(glob.glob('engines/**/stockfish*.exe', recursive=True)[0])
    e.configure({'UCI_LimitStrength': True, 'UCI_Elo': SF_ANCHOR})
    return e


def play_match(mover, games, sf_ms=50, cap=100):
    sf = sf_engine(); lim = chess.engine.Limit(time=sf_ms / 1000.0)
    W = D = L = 0
    for g in range(games):
        nw = (g % 2 == 0); b = chess.Board(); plies = 0
        while not b.is_game_over() and plies < cap:
            mv = mover(b) if (b.turn == chess.WHITE) == nw else sf.play(b, lim).move
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv); plies += 1
        if b.is_checkmate():
            res = 1.0 if (b.turn == chess.BLACK) == nw else 0.0
        elif b.is_game_over():
            res = 0.5
        else:
            cp = sf.analyse(b, chess.engine.Limit(time=0.05))['score'].white().score(mate_score=100000)
            res = 0.5 if (cp is None or abs(cp) <= 150) else (1.0 if (cp > 150) == nw else 0.0)
        W += res == 1.0; D += res == 0.5; L += res == 0.0
        print(f"  game {g+1}/{games}: {'W' if nw else 'B'} -> {'win' if res==1 else 'draw' if res==0.5 else 'loss'}", flush=True)
    sf.quit()
    return int(W), int(D), int(L)


def _elo(x):
    x = min(max(x, 1e-4), 1 - 1e-4)
    return SF_ANCHOR - 400 * math.log10(1 / x - 1)


def report(name, W, D, L):
    n = W + D + L
    s = (W + 0.5 * D) / n
    se = math.sqrt(max(s * (1 - s), 1e-9) / n)
    lo, hi = max(1e-4, s - 1.96 * se), min(1 - 1e-4, s + 1.96 * se)
    print(f"\n{name}: {W}W-{D}D-{L}L over {n}  score {s:.2f} (95% CI {lo:.2f}-{hi:.2f})")
    print(f"  implied Elo ~{_elo(s):.0f}  (95% CI {_elo(lo):.0f} .. {_elo(hi):.0f})")


def main():
    mode = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if mode == 'ladder':
        for depth in (1, 2, 3):
            eng = AlphaBetaEngine(time_limit=1e9, max_depth=depth)
            print(f"[engine-heuristic fixed depth {depth}]")
            W, D, L = play_match(lambda b, e=eng: e.search(b)[0], games)
            report(f'engine-heuristic-d{depth}', W, D, L)

    elif mode == 'cellA':
        from resnet_model import encode18
        q = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # quiescence plies (0 isolates search from the q-bug)
        heur = lambda b: math.tanh(pst_eval(b) / 400.0)
        print(f"[heuristic eval inside net_search depth-2 q{q}]")
        W, D, L = play_match(lambda b: net_search.best_move(b, None, dev, depth=2, beam=8,
                                                            encode_fn=encode18, q=q, board_eval=heur), games)
        report(f'heuristic-in-netsearch-d2q{q}', W, D, L)

    elif mode == 'net':
        from resnet_model import ChessResNet, encode18
        net = ChessResNet().to(dev); net.eval()
        net.load_state_dict(torch.load('models/tower.pth', map_location=dev)['state_dict'])
        print("[learned net inside net_search depth-2 q8]")
        W, D, L = play_match(lambda b: net_search.best_move(b, net, dev, depth=2, beam=8,
                                                           encode_fn=encode18, q=8), games)
        report('net-in-netsearch-d2q8', W, D, L)


if __name__ == "__main__":
    main()
