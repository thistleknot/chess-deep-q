"""Fast, REAL Elo anchor: play the net vs Stockfish pinned to a target Elo.

Uses Stockfish's UCI_LimitStrength/UCI_Elo (calibrated, external) so the number is a genuine
anchor, not a self-referential guess against our own engine. A short SF movetime keeps it fast.
The net's match score maps to an implied Elo around the SF setting via the logistic relation.

Usage:
    python measure_sf.py [sf_elo] [games] [sf_ms] [net_cap_plies]
"""
import sys
import glob
import time
import chess
import chess.engine

from legacy.chess_ai import OptimizedChessAI
from elo_calibration import score_to_elo_diff

SF_MIN_ELO = 1320   # Stockfish's UCI_Elo floor


def find_sf():
    hits = glob.glob('engines/**/stockfish*.exe', recursive=True)
    return hits[0] if hits else None


def main():
    sf_elo = int(sys.argv[1]) if len(sys.argv) > 1 else 1320
    # >=30 games is the gate-decision power floor (spec/elo-measurement.spec.md): at n=6 the per-game
    # SD ~0.13 is wider than a gate step, so small-n Elo readings are coin flips, not measurements.
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    # mode: 'mcts' (Russian-doll MCTS) or 'sN' (batched net-minimax depth N, e.g. s2, s3)
    mode = sys.argv[3] if len(sys.argv) > 3 else "mcts"
    sf_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 80

    sf_path = find_sf()
    if not sf_path:
        print("Stockfish not found under engines/. Run the download step first.")
        return
    clamped = max(SF_MIN_ELO, sf_elo)
    if clamped != sf_elo:
        print(f"(SF UCI_Elo floor is {SF_MIN_ELO}; using {clamped}.)")
    sf_elo = clamped

    ai = OptimizedChessAI(training_games=5, verbose=False)
    ai.load_model('chess_model.pth')

    # Move selector for the net side.
    if mode.startswith("s"):
        import net_search
        depth = int(mode[1:])
        net, dev = ai.dqn_agent.q_network, ai.dqn_agent.device
        net.eval()
        def net_move(board):
            return net_search.best_move(board, net, dev, depth=depth, beam=8)
        mode_label = f"net-minimax d{depth}"
    else:
        def net_move(board):
            ai.board = board.copy()
            return ai.get_best_move()
        mode_label = "MCTS"

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    limit = chess.engine.Limit(time=sf_ms / 1000.0)

    print(f"Net ({mode_label}) vs Stockfish@{sf_elo} (real anchor), {games} games, "
          f"SF {sf_ms}ms/move, cap {cap} plies.\n")
    total = 0.0
    start = time.time()
    for g in range(games):
        net_white = (g % 2 == 0)
        board = chess.Board()
        plies = 0
        while not board.is_game_over() and plies < cap:
            if (board.turn == chess.WHITE) == net_white:
                move = net_move(board)
            else:
                move = engine.play(board, limit).move
            if move is None or move not in board.legal_moves:
                break
            board.push(move)
            plies += 1
        # Score: decisive result, else adjudicate by SF eval of the final position.
        if board.is_checkmate():
            white_won = board.turn == chess.BLACK
            pts = 1.0 if white_won == net_white else 0.0
            res = "1-0" if white_won else "0-1"
        elif board.is_game_over():
            pts, res = 0.5, board.result()
        else:
            info = engine.analyse(board, chess.engine.Limit(time=0.05))
            cp = info["score"].white().score(mate_score=100000)
            if cp is None:
                pts, res = 0.5, "adj-draw"
            elif cp > 150:
                pts, res = (1.0 if net_white else 0.0), "adj-1-0"
            elif cp < -150:
                pts, res = (0.0 if net_white else 1.0), "adj-0-1"
            else:
                pts, res = 0.5, "adj-draw"
        total += pts
        print(f"  game {g+1}/{games}: net={'W' if net_white else 'B'} {res:9s} net_pts={pts:.1f} plies={plies}")

    engine.quit()
    import math
    score = total / games
    # Binomial 95% CI on the match score, mapped to Elo around the anchor. A gate is only cleared
    # when the CInterval lies above it; if it straddles the gate the result is INCONCLUSIVE.
    se = math.sqrt(max(score * (1 - score), 1e-9) / games)
    lo, hi = max(1e-4, score - 1.96 * se), min(1 - 1e-4, score + 1.96 * se)

    def _est(s):
        s = min(max(s, 1e-4), 1 - 1e-4)
        return sf_elo + score_to_elo_diff(s)

    print(f"\nNet score vs SF@{sf_elo}: {total:.1f}/{games} = {score:.2f} (95% CI {lo:.2f}-{hi:.2f})")
    print(f"ESTIMATED net Elo: ~{_est(score):.0f}  (95% CI {_est(lo):.0f}..{_est(hi):.0f}; "
          f"real SF anchor, n={games})")
    print(f"[wall {time.time()-start:.0f}s]")


if __name__ == "__main__":
    main()
