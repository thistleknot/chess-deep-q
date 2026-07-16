import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Time-budget kill-check for the deployed NNUE+phi agent (spec/search-mcts.spec.md :Deployed-agent:).

Plays the NNUE+phi agent vs pst AND vs NNUE-no-phi at EQUAL wall-time per move. This tests the deployment
claim: :Phi-widening:'s ~93% node reduction lets the (slow, recompute-per-leaf) NNUE reach enough DEPTH to
compete at equal time — the equal-TIME gate the eval-quality kill-check (measure_nnue.py, equal-DEPTH)
does not test. Persistent engines per side => :Tree-reuse: across moves, as deployed.

Usage: python measure_phi.py [games] [seconds]
"""
import sys

from chessdq.engine import AlphaBetaEngine, pst_eval
from chessdq.measure_ladder import play, adj_pst, elo_diff
from chessdq.nnue_model import load_nnue, make_incremental_nnue_eval


def mover(make_ev, phi, t):
    """One persistent time-budgeted engine + its OWN eval instance (the incremental accumulator is
    STATEFUL, so each engine builds a fresh one via the factory) -> a move_fn. The FAST incremental
    eval (Stage 3) is what this measure tests against the equal-time wall Phase C hit."""
    eng = AlphaBetaEngine(eval_fn=make_ev(), time_limit=t, max_depth=64, phi_widen=phi)
    return lambda b: eng.search(b)[0]


def report(tag, W, D, L):
    s = (W + 0.5 * D) / max(W + D + L, 1)
    print(f"{tag}: {W}W-{D}D-{L}L  score {s:.2f}  elo_diff {elo_diff(s):+.0f}", flush=True)


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    import os
    net = load_nnue(os.environ.get("NNUE_MODEL", "models/nnue.pt"), "cpu")
    nnue = lambda: make_incremental_nnue_eval(net, "cpu")   # fast incremental eval, fresh per engine
    pst = lambda: pst_eval
    print(f"time budget {t}s/move, {games} games, equal wall-time both sides (FAST incremental eval)\n", flush=True)
    report(f"NNUE+phi vs pst @{t}s", *play(mover(nnue, True, t), mover(pst, False, t), games, adj_pst))
    report(f"NNUE+phi vs NNUE-nophi @{t}s", *play(mover(nnue, True, t), mover(nnue, False, t), games, adj_pst))


if __name__ == "__main__":
    main()
