"""Measure the alpha-beta teacher (engine.py) vs Stockfish -- the distillation ceiling.

If the teacher beats SF@1320, distillation into the net has real room to grow; its score sets the
realistic strength target for a well-distilled net + lookahead. Fast (pure-Python engine).

Usage: python measure_engine_sf.py [sf_elo] [games] [engine_time_s] [sf_ms] [cap]
"""
import sys, glob, time
import chess, chess.engine
from engine import AlphaBetaEngine
from elo_calibration import score_to_elo_diff


def main():
    sf_elo = int(sys.argv[1]) if len(sys.argv) > 1 else 1320
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    eng_t = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    sf_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    cap = int(sys.argv[5]) if len(sys.argv) > 5 else 100

    sf_path = glob.glob('engines/**/stockfish*.exe', recursive=True)[0]
    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    lim = chess.engine.Limit(time=sf_ms / 1000.0)
    eng = AlphaBetaEngine(time_limit=eng_t)

    print(f"AlphaBetaEngine@{eng_t}s vs Stockfish@{sf_elo}, {games} games, cap {cap} plies.\n")
    total = 0.0; start = time.time()
    for g in range(games):
        eng_white = (g % 2 == 0)
        b = chess.Board(); plies = 0
        while not b.is_game_over() and plies < cap:
            if (b.turn == chess.WHITE) == eng_white:
                mv, _ = eng.search(b)
            else:
                mv = sf.play(b, lim).move
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv); plies += 1
        if b.is_checkmate():
            ww = b.turn == chess.BLACK; pts = 1.0 if ww == eng_white else 0.0; res = "1-0" if ww else "0-1"
        elif b.is_game_over():
            pts, res = 0.5, b.result()
        else:
            cp = sf.analyse(b, chess.engine.Limit(time=0.05))["score"].white().score(mate_score=100000)
            if cp is None: pts, res = 0.5, "adj-draw"
            elif cp > 150: pts, res = (1.0 if eng_white else 0.0), "adj-1-0"
            elif cp < -150: pts, res = (0.0 if eng_white else 1.0), "adj-0-1"
            else: pts, res = 0.5, "adj-draw"
        total += pts
        print(f"  game {g+1}/{games}: eng={'W' if eng_white else 'B'} {res:9s} eng_pts={pts:.1f} plies={plies}")
    sf.quit()
    score = total / games
    est = sf_elo + (score_to_elo_diff(score) if 0 < score < 1 else (400 if score >= 1 else -400))
    print(f"\nEngine score vs SF@{sf_elo}: {total:.1f}/{games} = {score:.2f}")
    print(f"ESTIMATED engine Elo: ~{est:.0f}  (real Stockfish anchor)")
    print(f"[wall {time.time()-start:.0f}s]")


if __name__ == "__main__":
    main()
