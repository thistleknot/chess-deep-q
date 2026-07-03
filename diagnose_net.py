"""Isolate whether weakness is the net's eval or the MCTS search.

Greedy 1-ply player: for each legal move, evaluate the resulting position with the net (White-
absolute), and the mover picks the move maximizing its own perspective. No MCTS. If this avoids
hanging pieces and grabs free material, the eval is fine and MCTS is the culprit.
"""
import chess
import torch
from neural_network import ChessQNetwork
from board_utils import board_to_tensor
from engine import pst_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ChessQNetwork().to(device)
ckpt = torch.load("models/chess_model.pth", map_location=device)
net.load_state_dict(ckpt["q_network_state_dict"]); net.eval()


def white_value(board):
    with torch.no_grad():
        t = board_to_tensor(board).unsqueeze(0).to(device)
        return float(net(t).item())


def mover_value(board_after, mover_is_white):
    v = white_value(board_after)
    return v if mover_is_white else -v


def greedy_move(board):
    mover_white = board.turn == chess.WHITE
    best, best_v = None, -1e9
    for m in board.legal_moves:
        b = board.copy(); b.push(m)
        v = mover_value(b, mover_white)
        if v > best_v:
            best_v, best = v, m
    return best, best_v


def top_moves(board, k=4):
    mover_white = board.turn == chess.WHITE
    scored = []
    for m in board.legal_moves:
        b = board.copy(); b.push(m)
        scored.append((mover_value(b, mover_white), m, pst_eval(b)))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]


print("=== net value sanity ===")
print("start pos white_value:", round(white_value(chess.Board()), 4))
# free queen: white to move, black queen en prise on d5, white pawn e4 can take? build a position.
free_q = chess.Board("rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
print("\n=== free black queen on d5 (White to move, exd5 wins queen) ===")
for v, m, pst in top_moves(free_q, 5):
    tag = " <-- exd5 (wins Q)" if m.uci() == "e4d5" else ""
    print(f"  net_mover_val={v:+.4f}  move={m.uci()}  pst_after(cp)={pst:+d}{tag}")

print("\n=== hanging: White to move, should NOT hang. Start after 1.e4 ===")
b = chess.Board(); b.push_san("e4")
for v, m, pst in top_moves(b, 4):
    print(f"  net_mover_val={v:+.4f}  move={m.uci()}  pst_after(cp)={pst:+d}")

print("\n=== greedy-net vs pst_eval agreement over 200 random positions ===")
import random
random.seed(1)
agree = 0; total = 0
for _ in range(200):
    bd = chess.Board()
    for _ in range(random.randint(2, 30)):
        if bd.is_game_over(): break
        bd.push(random.choice(list(bd.legal_moves)))
    if bd.is_game_over(): continue
    gm, _ = greedy_move(bd)
    # pst-greedy best move for the mover
    mover_white = bd.turn == chess.WHITE
    pst_best, pv = None, -1e9
    for m in bd.legal_moves:
        bb = bd.copy(); bb.push(m)
        val = pst_eval(bb) * (1 if mover_white else -1)
        if val > pv: pv, pst_best = val, m
    total += 1
    if gm == pst_best: agree += 1
print(f"  greedy-net matches pst-greedy on {agree}/{total} = {agree/max(total,1):.2f}")
