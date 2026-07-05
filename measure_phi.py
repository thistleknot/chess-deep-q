"""Time-budget kill-check for the deployed NNUE+phi agent (spec/search-mcts.spec.md :Deployed-agent:).

Plays the NNUE+phi agent vs pst AND vs NNUE-no-phi at EQUAL wall-time per move. This tests the deployment
claim: :Phi-widening:'s ~93% node reduction lets the (slow, recompute-per-leaf) NNUE reach enough DEPTH to
compete at equal time — the equal-TIME gate the eval-quality kill-check (measure_nnue.py, equal-DEPTH)
does not test. Persistent engines per side => :Tree-reuse: across moves, as deployed.

Usage: python measure_phi.py [games] [seconds]
"""
import sys

from engine import AlphaBetaEngine, pst_eval
from measure_ladder import play, adj_pst, elo_diff
from nnue_model import load_nnue, make_nnue_eval


def mover(eval_fn, phi, t):
    """One persistent time-budgeted engine (iterative deepening to the clock) -> a move_fn."""
    eng = AlphaBetaEngine(eval_fn=eval_fn, time_limit=t, max_depth=64, phi_widen=phi)
    return lambda b: eng.search(b)[0]


def report(tag, W, D, L):
    s = (W + 0.5 * D) / max(W + D + L, 1)
    print(f"{tag}: {W}W-{D}D-{L}L  score {s:.2f}  elo_diff {elo_diff(s):+.0f}", flush=True)


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    net = load_nnue("models/nnue.pt", "cpu")
    ev = make_nnue_eval(net, "cpu")
    print(f"time budget {t}s/move, {games} games, equal wall-time both sides\n", flush=True)
    report(f"NNUE+phi vs pst @{t}s", *play(mover(ev, True, t), mover(pst_eval, False, t), games, adj_pst))
    report(f"NNUE+phi vs NNUE-nophi @{t}s", *play(mover(ev, True, t), mover(ev, False, t), games, adj_pst))


if __name__ == "__main__":
    main()
