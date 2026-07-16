import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""H3: does the net's policy head, used as an alpha-beta MOVE-ORDERING prior, buy strength?

Holds the EVAL fixed (pst_eval, the strong 1672@d3 baseline) and varies ONLY move ordering:
  OFF = TT/MVV-LVA/killers/history (engine.py default)
  ON  = same, plus the net policy prior at shallow plies (policy_plies=K)
This is the inversion the plan argues for: cheap strong eval + learned ordering, not a slow learned
leaf value. Two measurements:
  (a) NODES to a fixed depth over a set of non-book midgame positions -- does the policy order well?
  (b) Elo vs SF@1320 at 0.3s/move -- is the ~2-3ms/node net cost repaid by the deeper search?
H3 passes iff nodes drop (a) AND the Elo CI clears the OFF baseline (b). If nodes don't drop, the
policy is too weak to order (expected until Stage-2/3) and we stop -- a clean falsification.

Usage: python measure_policyorder.py [games] [time_s] [policy_plies] [node_depth]
  e.g. python measure_policyorder.py 30 0.3 0 3
"""
import sys, time
import chess
import torch

from chessdq.engine import pst_eval, AlphaBetaEngine
from chessdq.resnet_model import ChessResNet
from chessdq.net_search import policy_priors
from chessdq.measure_gbdt import play_match, report

# Non-book midgame positions for the node-count test (out of the opening book so search runs).
FENS = [
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
    "r1bq1rk1/pp2bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
    "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 10",
    "r1bqr1k1/ppp2ppp/2np1n2/2b5/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQ - 0 9",
    "r3k2r/pppq1ppp/2npbn2/2b1p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 9",
]


def load_net(device):
    net = ChessResNet().to(device)
    sd = torch.load("models/tower.pth", map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    net.eval()
    return net


def count_nodes(policy_fn, plies, depth):
    total = 0
    for fen in FENS:
        eng = AlphaBetaEngine(eval_fn=pst_eval, time_limit=1e9, max_depth=depth,
                              policy_fn=policy_fn, policy_plies=plies)
        eng.search(chess.Board(fen))
        total += eng.nodes
    return total


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    tsec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    plies = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ndepth = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(device)
    pol = lambda b: policy_priors(b, net, device)

    # (a) Node-count test at fixed depth -- pure ordering quality, isolates the net time cost.
    print(f"[a] Nodes to depth {ndepth} over {len(FENS)} midgame positions (eval=pst; ordering only):")
    off_nodes = count_nodes(None, plies, ndepth)
    on_nodes = count_nodes(pol, plies, ndepth)
    drop = 100.0 * (off_nodes - on_nodes) / off_nodes if off_nodes else 0.0
    print(f"    OFF={off_nodes}  ON(policy_plies={plies})={on_nodes}  "
          f"node change {drop:+.1f}% ({'fewer' if drop > 0 else 'MORE'} nodes with policy)\n")

    # (b) Elo at time control -- does the depth gain survive the net cost?
    print(f"[b] Elo vs SF@1320 at {tsec}s/move, {games} games, eval=pst, policy_plies={plies}:")
    off = AlphaBetaEngine(eval_fn=pst_eval, time_limit=tsec)
    on = AlphaBetaEngine(eval_fn=pst_eval, time_limit=tsec, policy_fn=pol, policy_plies=plies)
    t = time.time()
    report("order-OFF", *play_match(lambda b: off.search(b)[0], games))
    report("order-ON", *play_match(lambda b: on.search(b)[0], games))
    print(f"  [wall {time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
