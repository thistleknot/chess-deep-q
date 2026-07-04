"""Stage-3 expert-iteration self-play (spec/self-play-leela.spec.md).

Search improves the policy; the net CLONES the search's chosen move (policy target = one-hot CE) and
REGRESSES the game outcome (value target = z, White-absolute); repeat, bootstrapping from the
improved net. Net-minimax :Search-profile: (net_search). Stockfish is the :Ladder: gate only, never
in the loop. Two :Self-play-bootstrap:s: heuristic (board_eval = tanh(pst/400)) or distilled (net).

The only channel that mints new value knowledge is the OUTCOME along a committed trajectory, so the
value target is z and exploration commits (opening-ply temperature + Dirichlet) are mandatory — else
every self-play game is identical (deterministic argmax) and nothing is learned.
"""
import math
import random
import time
from collections import deque

import chess
import numpy as np
import torch
import torch.nn.functional as F

from resnet_model import ChessResNet, encode18, move_index
from engine import pst_eval
import net_search
from measure_ladder import random_mover, heuristic_mover, adj_pst, play, elo_diff

POLICY_LOSS_WEIGHT = 0.5


def heuristic_eval(board):
    """White-absolute [-1,1] hand eval — the heuristic :Self-play-bootstrap: leaf."""
    return math.tanh(pst_eval(board) / 400.0)


def _sample_root(board, net, dev, depth, board_eval, tau, dir_alpha):
    """Opening-ply committed exploration: softmax over child values (mover frame) mixed with
    Dirichlet, then sample. Returns the chosen move."""
    moves = list(board.legal_moves)
    children = []
    for m in moves:
        c = board.copy(); c.push(m); children.append(c)
    vals = net_search._white_values(children, net, dev, encode_fn=encode18, board_eval=board_eval)
    v = np.array(vals, dtype=np.float64) * (1.0 if board.turn == chess.WHITE else -1.0)
    p = np.exp((v - v.max()) / max(tau, 1e-3)); p /= p.sum()
    if dir_alpha > 0:
        p = 0.75 * p + 0.25 * np.random.dirichlet([dir_alpha] * len(p))
    return moves[int(np.random.choice(len(p), p=p))]


def play_selfplay_game(net, dev, depth, board_eval, temp_plies, dir_alpha, cap=120):
    """One self-play game. Returns (states, policy_idxs, value_targets) — value target = outcome z
    (White-absolute) at every position; policy target = the search's chosen move index."""
    board = chess.Board()
    states, pols, plies = [], [], 0
    while not board.is_game_over() and plies < cap:
        if plies < temp_plies:
            mv = _sample_root(board, net, dev, depth, board_eval, tau=1.0, dir_alpha=dir_alpha)
        else:
            mv = net_search.best_move(board, net, dev, depth=depth, beam=8,
                                      encode_fn=encode18, board_eval=board_eval)
        if mv is None or mv not in board.legal_moves:
            break
        states.append(encode18(board))
        pols.append(move_index(mv))
        board.push(mv); plies += 1
    z = 0.0
    if board.is_checkmate():                       # side to move is mated
        z = 1.0 if board.turn == chess.BLACK else -1.0
    return states, pols, [z] * len(states)


def _train(net, opt, buffer, dev, steps, batch, use_amp):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    net.train()
    losses = []
    for _ in range(steps):
        sample = random.sample(buffer, min(batch, len(buffer)))
        X = torch.stack([s[0] for s in sample]).to(dev)
        pol = torch.tensor([s[1] for s in sample], dtype=torch.long, device=dev)
        y = torch.tensor([s[2] for s in sample], dtype=torch.float32, device=dev)
        opt.zero_grad()
        with torch.autocast("cuda", enabled=use_amp):
            v, p = net(X)
            loss = F.mse_loss(v.squeeze(1), y) + POLICY_LOSS_WEIGHT * F.cross_entropy(p, pol)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(loss.item())
    net.eval()
    return sum(losses) / max(1, len(losses))


def _ladder(net, dev, depth, games, with_sf=False):
    """Per-iteration ladder: net-minimax vs random + heuristic-1ply (ordinal Elo diff). SF optional."""
    net.eval()
    mv = lambda b: net_search.best_move(b, net, dev, depth=depth, beam=8, encode_fn=encode18)
    out = {}
    for name, opp in (("random", random_mover), ("heuristic", heuristic_mover)):
        W, D, L = play(mv, opp, games, adj_pst)
        s = (W + 0.5 * D) / (W + D + L)
        out[name] = {"score": s, "elo_diff": elo_diff(s), "wdl": (W, D, L)}
    return out


def expert_iteration(net, dev, iters, games_per_iter, depth=2, temp_plies=8, dir_alpha=0.3,
                     eval_for_iter=None, ladder_games=12, train_steps=200, batch=256, label="run"):
    """Run `iters` expert-iteration cycles; return a list of per-iteration ladder rows (the curve).
    eval_for_iter(i) -> board_eval (None = net eval; a fn = heuristic bootstrap for that iteration)."""
    use_amp = (dev.type == "cuda")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = deque(maxlen=60000)
    curve = []
    for i in range(iters):
        board_eval = eval_for_iter(i) if eval_for_iter else None
        t0 = time.time()
        for _ in range(games_per_iter):
            st, po, va = play_selfplay_game(net, dev, depth, board_eval, temp_plies, dir_alpha)
            for s in zip(st, po, va):
                buffer.append(s)
        loss = _train(net, opt, buffer, dev, train_steps, batch, use_amp) if buffer else float("nan")
        lad = _ladder(net, dev, depth, ladder_games)
        row = {"iter": i, "eval": "heuristic" if board_eval else "net", "buffer": len(buffer),
               "loss": round(loss, 4),
               "vs_random": round(lad["random"]["score"], 2),
               "vs_heuristic": round(lad["heuristic"]["score"], 2),
               "wall_s": round(time.time() - t0)}
        curve.append(row)
        print(f"[{label}] iter {i}: eval={row['eval']} loss={row['loss']} "
              f"vs_random={row['vs_random']} vs_heuristic={row['vs_heuristic']} "
              f"buf={row['buffer']} [{row['wall_s']}s]", flush=True)
    return curve
