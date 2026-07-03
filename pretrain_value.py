"""Supervised warm-start for the value net: distill a fast search teacher into V(s).

WHY: at inference the MCTS leaf value is taken 100% from the network (learned_leaf_weight -> 1.0),
so an untrained net feeds the search garbage and it hangs pieces. Self-play RL from scratch is far
too slow to fix this in 30 minutes. Instead we regress V(s) directly onto a cheap alpha-beta+
quiescence teacher (engine.AlphaBetaEngine.evaluate) over a large, diverse set of positions. This
is the classic heuristic/supervised bootstrap stage (prompt.md, spec/rl-categorization): OFFLINE,
off-policy, supervised value regression / knowledge distillation -- NOT DQN, SARSA, or actor-critic,
and no policy gradient. Self-play TD(0) then fine-tunes on top to try to exceed the teacher.

Target scale: tanh(white_centipawns / 400) in [-1,1] -- ~4 pawns maps near +/-0.76, a standard
centipawn->win-ish squash, matching the network's tanh value head and White-absolute frame.

Usage:
    python pretrain_value.py [minutes] [teacher_depth] [--fresh]
        minutes       wall-clock training budget (default 20)
        teacher_depth fixed alpha-beta depth for labels (default 3; 2 is faster, 4 stronger/slower)
        --fresh       start from random weights instead of the existing checkpoint
"""

import sys
import time
import math
import random
from collections import deque

import chess
import torch
import torch.nn.functional as F

from neural_network import ChessQNetwork
from engine import AlphaBetaEngine
from board_utils import board_to_tensor

TARGET_CP_SCALE = 400.0          # centipawns -> tanh squash divisor
VAL_BUFFER_MAX = 60_000          # streaming replay of (tensor, target) pairs
GEN_BATCH = 192                  # positions generated+labelled per outer step
SGD_STEPS_PER_BATCH = 24         # gradient steps per outer step
MINIBATCH = 256


def teacher_value(engine, board, depth):
    """White-absolute distillation target in [-1,1]."""
    cp = engine.evaluate(board, depth=depth)
    return math.tanh(cp / TARGET_CP_SCALE)


def random_position(max_plies=40, capture_bias=0.5):
    """A position from a short randomized playout; mild capture bias creates tactical tension."""
    board = chess.Board()
    plies = random.randint(4, max_plies)
    for _ in range(plies):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        caps = [m for m in legal if board.is_capture(m)]
        move = random.choice(caps) if (caps and random.random() < capture_bias) else random.choice(legal)
        board.push(move)
    return board


def make_labelled_batch(engine, depth, n):
    """Generate n non-terminal positions and label them with the teacher."""
    xs, ys = [], []
    while len(xs) < n:
        board = random_position()
        if board.is_game_over():
            continue
        xs.append(board_to_tensor(board))
        ys.append(teacher_value(engine, board, depth))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 20.0
    depth = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 3
    fresh = "--fresh" in sys.argv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessQNetwork().to(device)
    if not fresh:
        try:
            ckpt = torch.load("models/chess_model.pth", map_location=device)
            net.load_state_dict(ckpt["q_network_state_dict"])
            print("Loaded existing weights (fine-tuning on top).")
        except Exception as e:
            print(f"Could not load checkpoint ({e}); starting fresh.")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    engine = AlphaBetaEngine(time_limit=1e9)   # depth-bounded teacher; book/ID bypassed by evaluate()

    # Fixed held-out validation set for honest progress tracking.
    print("Building held-out validation set...")
    val_x, val_y = make_labelled_batch(engine, depth, 512)
    val_x = val_x.to(device); val_y = val_y.to(device)

    buffer = deque(maxlen=VAL_BUFFER_MAX)
    budget = minutes * 60.0
    start = time.time()
    step = 0
    labelled = 0
    print(f"Distilling depth-{depth} teacher for {minutes:.0f} min on {device}...\n")

    while time.time() - start < budget:
        bx, by = make_labelled_batch(engine, depth, GEN_BATCH)
        labelled += len(bx)
        for x, y in zip(bx, by):
            buffer.append((x, y.item()))

        if len(buffer) >= MINIBATCH:
            for _ in range(SGD_STEPS_PER_BATCH):
                idx = random.sample(range(len(buffer)), MINIBATCH)
                xb = torch.stack([buffer[i][0] for i in idx]).to(device)
                yb = torch.tensor([buffer[i][1] for i in idx], dtype=torch.float32, device=device)
                pred = net(xb).squeeze(1)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        step += 1

        if step % 5 == 0:
            net.eval()
            with torch.no_grad():
                vpred = net(val_x).squeeze(1)
                vmse = F.mse_loss(vpred, val_y).item()
                # sign agreement: does the net rank White-better positions as better?
                sign_acc = ((vpred > 0) == (val_y > 0)).float().mean().item()
            net.train()
            el = time.time() - start
            rate = labelled / el
            print(f"[{el:5.0f}s] step {step:4d}  buf {len(buffer):6d}  "
                  f"val_mse {vmse:.4f}  sign_acc {sign_acc:.2f}  "
                  f"labelled {labelled} ({rate:.0f}/s)")

    # Save, preserving the checkpoint's auxiliary fields where possible.
    import os
    os.makedirs("models", exist_ok=True)
    net.eval()
    try:
        ckpt = torch.load("models/chess_model.pth", map_location="cpu")
    except Exception:
        ckpt = {}
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    ckpt["q_network_state_dict"] = sd
    ckpt["target_network_state_dict"] = sd
    ckpt["pretrain"] = {"teacher_depth": depth, "minutes": minutes,
                        "labelled": labelled, "target_cp_scale": TARGET_CP_SCALE}
    torch.save(ckpt, "models/chess_model.pth")
    el = time.time() - start
    print(f"\nDone. {labelled} positions labelled in {el:.0f}s. Saved models/chess_model.pth.")


if __name__ == "__main__":
    main()
