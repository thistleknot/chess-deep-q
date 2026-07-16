import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Node-budget kill-check — the DEPLOYED spearhead at its ~10k-node design point (the perft-table budget).

Unlike measure_nnue.py (FULL-WIDTH alpha-beta -> pays the perft explosion, e.g. 197742 nodes at d4) and
measure_phi.py (TIME budget -> favors pst's faster eval), this runs BOTH sides as the deployed agent: phi
(:Phi-widening:) at a fixed max_depth with the FAST incremental accumulator. phi caps width to the Fibonacci
schedule, so reaching depth 8 costs ~10k nodes (per the design table), reaching 10-11 stays ~11k. Both sides
get the SAME search (phi, same depth) so EVAL QUALITY decides at the deployment budget. Reports avg nodes/move
and wall-time to confirm the ~10k / fast claim.

Usage: python measure_phi_depth.py [games] [depth]
"""
import sys
import os
import time

from chessdq.engine import AlphaBetaEngine, pst_eval
from chessdq.measure_ladder import play, adj_pst, elo_diff
from chessdq.nnue_model import load_nnue, make_incremental_nnue_eval


def _mover(make_ev, depth):
    """Persistent depth-bounded phi engine (:Tree-reuse: across moves, as deployed) -> (move_fn, engine).
    time_limit huge so it's DEPTH/phi-bounded, not time-bounded; the move_fn records nodes/move."""
    eng = AlphaBetaEngine(eval_fn=make_ev(), time_limit=1e9, max_depth=depth, phi_widen=True)

    def f(b):
        m, info = eng.search(b)
        f.nodes.append(info["nodes"])
        return m
    f.nodes = []
    return f


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    net = load_nnue(os.environ.get("NNUE_MODEL", "models/nnue.pt"), "cpu")
    nn = _mover(lambda: make_incremental_nnue_eval(net, "cpu"), depth)
    pst = _mover(lambda: pst_eval, depth)
    print(f"NNUE+phi vs pst+phi, BOTH @ max_depth={depth} (deployment budget, fast incremental eval)\n", flush=True)
    t0 = time.time()
    W, D, L = play(nn, pst, games, adj_pst)
    dt = time.time() - t0
    s = (W + 0.5 * D) / max(W + D + L, 1)
    avg_nodes = sum(nn.nodes) / max(len(nn.nodes), 1)
    verdict = "BEATS pst" if s > 0.5 else "does NOT beat pst"
    print(f"NNUE+phi@d{depth} vs pst+phi@d{depth}: {W}W-{D}D-{L}L  score {s:.2f}  elo_diff {elo_diff(s):+.0f}  ({verdict})")
    print(f"  avg {avg_nodes:.0f} NNUE nodes/move, {dt / max(W + D + L, 1):.1f}s/game total  (~10k-node design budget)")


if __name__ == "__main__":
    main()
