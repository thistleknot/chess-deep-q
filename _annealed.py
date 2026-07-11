import random, chess
from engine import pst_eval, AlphaBetaEngine
from measure_ladder import heuristic_mover, adj_pst, play, elo_diff

# breadth fraction of QUIET moves by ply-from-root: full at 1-2 (sound shallow), anneal deeper.
FRAC = {1: 1.0, 2: 1.0, 3: 0.8, 4: 0.5}
def frac(ply): return FRAC.get(ply, 0.3)

def negamax(b, remaining, ply, alpha, beta):
    if remaining == 0 or b.is_game_over():
        v = pst_eval(b)
        return v if b.turn == chess.WHITE else -v
    moves = list(b.legal_moves)
    forcing = [m for m in moves if b.is_capture(m) or b.gives_check(m)]   # never drop refutations
    quiet = [m for m in moves if not (b.is_capture(m) or b.gives_check(m))]
    f = frac(ply)
    if f < 1.0 and len(quiet) > 1:
        quiet = random.sample(quiet, max(1, round(f * len(quiet))))
    best = -1e18
    for m in forcing + quiet:
        b.push(m); val = -negamax(b, remaining - 1, ply + 1, -beta, -alpha); b.pop()
        if val > best: best = val
        if val > alpha: alpha = val
        if alpha >= beta: break
    return best

def annealed_mover(D):
    def mv(b):
        best_m, best_v = None, -1e18
        for m in b.legal_moves:                       # full breadth at root
            b.push(m); v = -negamax(b, D - 1, 2, -1e18, 1e18); b.pop()
            if v > best_v: best_v, best_m = v, m
        return best_m
    return mv

D = 5
mv = annealed_mover(D)
pst_d = lambda dep: (lambda b: AlphaBetaEngine(eval_fn=pst_eval, time_limit=1e9, max_depth=dep).search(b)[0])
print(f"annealed-breadth negamax (D={D}, full to ply2 + all forcing, quiet annealed):\n", flush=True)
for name, opp in [("heuristic-1ply", heuristic_mover), ("pst-d2", pst_d(2)), ("pst-d3", pst_d(3))]:
    W,Dr,L = play(mv, opp, 8, adj_pst); s=(W+0.5*Dr)/8
    print(f"annealed-D{D} vs {name:14s}: {W}W-{Dr}D-{L}L  score {s:.2f}  elo_diff {elo_diff(s):+.0f} "
          f"{'<-- BEATS' if s>0.5 else ''}", flush=True)
