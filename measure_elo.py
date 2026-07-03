"""Estimate the learned net's current strength WITHOUT Stockfish.

No external reference engine is available, so we anchor to this repo's own alpha-beta engine
(engine.py). We play the learned value-net (argmax MCTS at inference) against AlphaBetaEngine at a
known time control whose approximate ELO is taken from play_engine.LEVELS, then convert the match
score to an ELO estimate with the standard logistic relation.

Every number here is a DOUBLE estimate: the net's score is noisy over few games, and the engine's
own ELO is only a rough ballpark for a pure-Python alpha-beta. Treat the output as order-of-
magnitude, not a rated figure.

Usage:
    python measure_elo.py [games] [engine_time_s] [max_plies]
"""

import sys
import time
import chess

from chess_ai import OptimizedChessAI
from engine import AlphaBetaEngine, pst_eval
from elo_calibration import score_to_elo_diff

# Approximate ELO anchors for AlphaBetaEngine by search time (from play_engine.LEVELS; rough).
ENGINE_ELO_BY_TIME = {0.1: 1200, 0.5: 1500, 1.5: 1750, 3.0: 1900, 6.0: 2050}

ADJUDICATE_CP = 200          # |pst eval| beyond this (centipawns) decides an unfinished game


def adjudicate(board):
    """Result for a game stopped at the ply cap: by decisive material/PST edge, else draw."""
    if board.is_game_over():
        return board.result()
    score = pst_eval(board)          # White-absolute centipawns
    if score > ADJUDICATE_CP:
        return "1-0"
    if score < -ADJUDICATE_CP:
        return "0-1"
    return "1/2-1/2"


def play_game(ai, engine, net_is_white, max_plies, log):
    board = chess.Board()
    plies = 0
    net_move_time = 0.0
    while not board.is_game_over() and plies < max_plies:
        net_to_move = (board.turn == chess.WHITE) == net_is_white
        if net_to_move:
            ai.board = board.copy()
            t = time.time()
            move = ai.get_best_move()          # inference: argmax MCTS, full schedule progress
            net_move_time += time.time() - t
        else:
            move, _ = engine.search(board)
        if move is None or move not in board.legal_moves:
            break
        board.push(move)
        plies += 1
    result = adjudicate(board)
    return result, plies, net_move_time


def net_score_from_result(result, net_is_white):
    """Points for the net (1 win / 0.5 draw / 0 loss)."""
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == net_is_white else 0.0


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    engine_time = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    max_plies = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    anchor = ENGINE_ELO_BY_TIME.get(engine_time, 1200)
    print(f"Measuring net vs AlphaBetaEngine @ {engine_time}s/move "
          f"(engine ~{anchor} ELO, rough), {games} games, cap {max_plies} plies.\n")

    ai = OptimizedChessAI(training_games=5, verbose=False)
    ai.load_model('chess_model.pth')
    engine = AlphaBetaEngine(time_limit=engine_time)

    total, net_move_time_all = 0.0, 0.0
    start = time.time()
    for g in range(games):
        net_is_white = (g % 2 == 0)
        result, plies, nmt = play_game(ai, engine, net_is_white, max_plies, print)
        pts = net_score_from_result(result, net_is_white)
        total += pts
        net_move_time_all += nmt
        side = "White" if net_is_white else "Black"
        print(f"  game {g+1}/{games}: net={side:5s} result={result:7s} "
              f"net_pts={pts:.1f} plies={plies} net_move_avg={nmt/max(plies//2,1):.2f}s")

    score = total / games
    elo_diff = score_to_elo_diff(score) if 0 < score < 1 else (
        400 if score >= 1 else -400)  # clamp the shutout ends
    est = anchor + elo_diff
    elapsed = time.time() - start
    print(f"\nNet match score: {total:.1f}/{games} = {score:.2f}")
    print(f"Implied ELO gap vs engine: {elo_diff:+.0f}")
    print(f"ESTIMATED net ELO: ~{est:.0f}  (anchor {anchor} + gap; DOUBLE-estimated, noisy)")
    print(f"[wall {elapsed:.0f}s, net thinking {net_move_time_all:.0f}s]")


if __name__ == "__main__":
    main()
