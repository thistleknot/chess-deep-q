import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Place the NNUE+phi DEPLOYMENT agent on the SF-anchored ladder -> calibrated Elo (the 1600 verdict).

measure_ladder.py places the OLD ResNet tower (tower.pth); this places the actual deployed agent:
AlphaBetaEngine with the incremental NNUE eval + :Phi-widening: at a fixed depth (the ~10k-node budget).
Reuses measure_ladder's rungs and SF setup: random + 1-ply heuristic (ordinal) and SF@1320 (ANCHORED).
Beating SF@1320 with score s => ~1320 + elo_diff(s) Elo; score >= 0.83 => 1600+.

Usage: python measure_nnue_ladder.py [games] [depth]
"""
import sys
import os
import glob

import chess
import chess.engine

from chessdq.engine import AlphaBetaEngine
from chessdq.nnue_model import load_nnue, make_incremental_nnue_eval
from chessdq.measure_ladder import play, report, random_mover, heuristic_mover, adj_pst


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    net = load_nnue(os.environ.get("NNUE_MODEL", "models/nnue.pt"), "cpu")
    selective = os.environ.get("PHI_SELECTIVE", "0") == "1"
    eng = AlphaBetaEngine(eval_fn=make_incremental_nnue_eval(net, "cpu"),
                          time_limit=1e9, max_depth=depth, phi_widen=True, selective=selective)
    net_move = lambda b: eng.search(b)[0]
    print(f"NNUE+phi@d{depth} (deployment agent) on the SF-anchored ladder, {games} games/rung\n", flush=True)
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
    else:
        print("  (no stockfish -> no calibrated anchor)")


if __name__ == "__main__":
    main()
