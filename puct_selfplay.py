"""Batched parallel PUCT self-play — the PRIMARY :Search-profile: (spec/self-play-leela.spec.md).

The :Amplification-experiment:. Every prior RL attempt used a WEAK improvement operator (d1
net-minimax, or a sampled/optimistic beam) and plateaued at beats-random/loses-heuristic. PUCT is a
genuinely stronger operator; its per-move net cost is dodged by BATCHING across PUCT_PARALLEL_GAMES
concurrent games (the conv net is launch-bound at ONE position but fast batched). Value is stored
WHITE-ABSOLUTE (backup needs no sign flips; only PUCT selection applies the side-to-move sign),
matching the net's White-absolute tanh value head. Policy target = the root visit distribution
(:Visit-distribution-target:); value target = the committed outcome z. Per-iteration :Ladder: curve
(reused from selfplay._ladder, d1 net-minimax — comparable to the prior plateaus) is the verdict.

Usage: python puct_selfplay.py [iters] [games_per_iter] [playouts]
"""
import math, random, sys, time
from collections import deque

import chess
import numpy as np
import torch
import torch.nn.functional as F

from resnet_model import ChessResNet, encode18, move_index, POLICY_SIZE
import selfplay  # reuse _ladder (d1 net-minimax vs random/heuristic)

PUCT_PLAYOUTS = 128            # operator-strength dial (:Search-measured-gate:): higher = stronger improvement op
PUCT_PARALLEL_GAMES = 48
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
MOVE_TEMPERATURE_PLIES = 8
GAME_CAP = 120
LADDER_PLAYOUTS = 48           # cheap search-measured ladder (net+PUCT, not d1) per iteration
LADDER_GAMES = 10


class Node:
    """One search node. Value stats are WHITE-ABSOLUTE; selection applies the mover sign."""
    __slots__ = ("board", "turn", "expanded", "terminal", "tv",
                 "moves", "P", "N", "W", "Ntot", "children")

    def __init__(self, board):
        self.board = board
        self.turn = board.turn
        self.expanded = False
        self.terminal = False
        self.tv = 0.0          # terminal White-absolute value
        self.moves = None
        self.P = self.N = self.W = None
        self.Ntot = 0
        self.children = {}     # move-index-in-moves -> child Node


def terminal_white_value(board):
    """White-absolute outcome of a game-over board: mated side loses, else draw."""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


def make_child(node, i):
    """Child by playing node.moves[i] (stackless copy — no repetition draws inside search, cap-bounded)."""
    b = node.board.copy(stack=False)
    b.push(node.moves[i])
    ch = Node(b)
    if b.is_game_over():
        ch.terminal = True
        ch.tv = terminal_white_value(b)
        ch.expanded = True     # terminal is a leaf, never eval'd/expanded further
    return ch


@torch.no_grad()
def eval_boards(boards, net, dev):
    """Batched net forward. Returns (white_values[B], policy_logits[B,4096]) as numpy."""
    X = torch.stack([encode18(b) for b in boards]).to(dev)
    v, p = net(X)
    return v.squeeze(1).float().cpu().numpy(), p.float().cpu().numpy()


def expand(node, logits, root_noise=False):
    """Set legal moves, softmax priors over their policy indices, zero visit stats."""
    moves = list(node.board.legal_moves)
    node.moves = moves
    if not moves:
        node.P = np.zeros(0); node.N = np.zeros(0); node.W = np.zeros(0)
        node.expanded = True
        return
    idx = np.fromiter((move_index(m) for m in moves), dtype=np.int64, count=len(moves))
    lg = logits[idx]
    lg = lg - lg.max()
    pri = np.exp(lg); pri /= pri.sum()
    if root_noise:
        pri = 0.75 * pri + 0.25 * np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
    node.P = pri
    node.N = np.zeros(len(moves))
    node.W = np.zeros(len(moves))
    node.Ntot = 0
    node.expanded = True


def select_child(node):
    """PUCT: argmax over moves of sign*Q_white + C_PUCT*P*sqrt(Ntot+1)/(1+N). sign = mover frame."""
    sign = 1.0 if node.turn == chess.WHITE else -1.0
    q = np.where(node.N > 0, node.W / np.maximum(node.N, 1.0), 0.0) * sign
    u = C_PUCT * node.P * math.sqrt(node.Ntot + 1) / (1.0 + node.N)
    return int(np.argmax(q + u))


def select_leaf(root):
    """Descend through expanded nodes to an unexpanded/terminal leaf. Returns (leaf, path)."""
    node, path = root, []
    while node.expanded and not node.terminal:
        i = select_child(node)
        path.append((node, i))
        ch = node.children.get(i)
        if ch is None:
            ch = make_child(node, i)
            node.children[i] = ch
        node = ch
    return node, path


def backup(path, v_white):
    """Add the White-absolute leaf value along the path (no sign flips)."""
    for node, i in path:
        node.N[i] += 1
        node.W[i] += v_white
        node.Ntot += 1


def puct_move(board, net, dev, playouts):
    """Single-board PUCT: argmax over root visits, NO Dirichlet (a :Measurement-game:). This is the
    net's ACTUAL agent — used by the :Search-measured-gate: ladder and puct_eval.py."""
    root = Node(board.copy(stack=False))
    _, ps = eval_boards([root.board], net, dev)
    expand(root, ps[0], root_noise=False)
    if not root.moves:
        return None
    for _ in range(playouts):
        node, path = select_leaf(root)
        if node.terminal:
            backup(path, node.tv)
        else:
            vs, pp = eval_boards([node.board], net, dev)
            expand(node, pp[0])
            backup(path, float(vs[0]))
    return root.moves[int(np.argmax(root.N))]


def search_ladder(net, dev, playouts, games):
    """:Search-measured-gate: ladder — net+PUCT (its real search) vs random + heuristic-1ply.
    Replaces the d1 net-minimax probe, which is blind to this agent (0/16 losses vs 16/16 draws)."""
    from measure_ladder import random_mover, heuristic_mover, adj_pst, play, elo_diff
    net.eval()
    mv = lambda b: puct_move(b, net, dev, playouts)
    out = {}
    for name, opp in (("random", random_mover), ("heuristic", heuristic_mover)):
        W, D, L = play(mv, opp, games, adj_pst)
        s = (W + 0.5 * D) / (W + D + L)
        out[name] = {"score": s, "elo_diff": elo_diff(s), "wdl": (W, D, L)}
    return out


def run_selfplay_batch(net, dev, n_games, playouts, eps=0.0):
    """Play n_games in lockstep, batching every leaf eval across games. Returns a list of
    (state_tensor, (move_indices, visit_probs), z) records.

    eps > 0 blends the hand heuristic into the LEAF VALUE — v_leaf = eps·tanh(pst/400) +
    (1−eps)·V_net (White-absolute) — the AlphaStar-pattern baseline: a strong (heuristic-guided)
    search from iteration 0, off which the net learns; eps anneals to 0 as the net takes over."""
    games = []
    for _ in range(n_games):
        b = chess.Board()
        games.append({"root": Node(b), "ply": 0, "recs": []})
    records = []
    active = games
    while active:
        # (1) expand any un-expanded roots (batched) with root Dirichlet noise
        need = [g for g in active if not g["root"].expanded]
        if need:
            vs, ps = eval_boards([g["root"].board for g in need], net, dev)
            for g, pp in zip(need, ps):
                expand(g["root"], pp, root_noise=True)
        # (2) playout sims — one leaf per game per step, batched across games
        for _ in range(playouts):
            leaves = []
            for g in active:
                node, path = select_leaf(g["root"])
                if node.terminal:
                    backup(path, node.tv)
                else:
                    leaves.append((node, path))
            if leaves:
                vs, ps = eval_boards([nd.board for nd, _ in leaves], net, dev)
                for (nd, path), vv, pp in zip(leaves, vs, ps):
                    expand(nd, pp)
                    v_leaf = float(vv) if eps <= 0.0 else \
                        eps * selfplay.heuristic_eval(nd.board) + (1.0 - eps) * float(vv)
                    backup(path, v_leaf)
        # (3) each game commits a move from the root visit counts, then advances (tree reuse)
        still = []
        for g in active:
            root = g["root"]
            if not root.moves:                       # terminal/no-move root: finalize as draw/mate
                z = root.tv if root.terminal else 0.0
                for st, idx, pr in g["recs"]:
                    records.append((st, (idx, pr), z))
                continue
            N = root.N
            probs = N / N.sum() if N.sum() > 0 else np.ones(len(N)) / len(N)
            didx = np.fromiter((move_index(m) for m in root.moves), dtype=np.int64, count=len(root.moves))
            g["recs"].append((encode18(root.board), didx, probs.copy()))
            if g["ply"] < MOVE_TEMPERATURE_PLIES and N.sum() > 0:
                pick = int(np.random.choice(len(N), p=probs))
            else:
                pick = int(np.argmax(N))
            child = root.children.get(pick) or make_child(root, pick)
            g["root"] = child
            g["ply"] += 1
            if child.terminal or g["ply"] >= GAME_CAP:
                z = child.tv if child.terminal else 0.0
                for st, idx, pr in g["recs"]:
                    records.append((st, (idx, pr), z))
            else:
                still.append(g)
        active = still
    return records


def train_puct(net, opt, buffer, dev, steps, batch, use_amp):
    """MSE(V, z) + 0.5 * soft-CE(policy, visit distribution)."""
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    net.train()
    losses = []
    for _ in range(steps):
        sample = random.sample(buffer, min(batch, len(buffer)))
        X = torch.stack([s[0] for s in sample]).to(dev)
        z = torch.tensor([s[2] for s in sample], dtype=torch.float32, device=dev)
        tgt = torch.zeros(len(sample), POLICY_SIZE, device=dev)
        for r, s in enumerate(sample):
            idx, pr = s[1]
            tgt[r, torch.as_tensor(idx, device=dev)] = torch.as_tensor(pr, dtype=torch.float32, device=dev)
        opt.zero_grad()
        with torch.autocast("cuda", enabled=use_amp):
            v, p = net(X)
            pol_loss = -(tgt * F.log_softmax(p, dim=1)).sum(1).mean()
            loss = F.mse_loss(v.squeeze(1), z) + 0.5 * pol_loss
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(loss.item())
    net.eval()
    return sum(losses) / max(1, len(losses))


MODEL_PATH = "models/tower_puct.pt"


def train_puct_selfplay(iters=8, games=96, playouts=PUCT_PLAYOUTS, dev=None, resume=True):
    """:Train-mode: Stage-3 (self-play-leela.spec.md) — batched-PUCT expert iteration. Resumes from
    MODEL_PATH if present (:Run-contract:), checkpoints each iter, returns the per-iteration curve."""
    import os
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessResNet().to(dev)
    if resume and os.path.exists(MODEL_PATH):
        net.load_state_dict(torch.load(MODEL_PATH, map_location=dev)["state_dict"])
        print(f"resumed from {MODEL_PATH}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = deque(maxlen=120000)
    use_amp = (dev.type == "cuda")
    print(f"PUCT self-play: {iters} iters x {games} games, {playouts} playouts, "
          f"batch {PUCT_PARALLEL_GAMES} games (train on {dev})\n")
    curve = []
    anneal_iters = max(1, iters // 2)                          # heuristic baseline 1→0 over the first half
    for i in range(iters):
        t0 = time.time()
        eps = max(0.0, 1.0 - i / anneal_iters)                 # AlphaStar pattern: baseline → net
        made = 0
        while made < games:
            n = min(PUCT_PARALLEL_GAMES, games - made)
            buffer.extend(run_selfplay_batch(net, dev, n, playouts, eps=eps))
            made += n
        loss = train_puct(net, opt, buffer, dev, steps=300, batch=256, use_amp=use_amp)
        lad = search_ladder(net, dev, LADDER_PLAYOUTS, LADDER_GAMES)   # net+PUCT, not d1
        row = {"iter": i, "eps": round(eps, 2), "loss": round(loss, 3), "buf": len(buffer),
               "vs_random": round(lad["random"]["score"], 2),
               "vs_heuristic": round(lad["heuristic"]["score"], 2),
               "wall_s": round(time.time() - t0)}
        curve.append(row)
        print(f"iter {i}: eps={row['eps']}  loss={row['loss']}  vs_random={row['vs_random']}  "
              f"vs_heuristic={row['vs_heuristic']} (net+PUCT)  buf={row['buf']}  [{row['wall_s']}s]", flush=True)
        os.makedirs("models", exist_ok=True)
        torch.save({"state_dict": net.state_dict()}, MODEL_PATH)
    print(f"saved {MODEL_PATH}")
    return curve


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    playouts = int(sys.argv[3]) if len(sys.argv) > 3 else PUCT_PLAYOUTS
    print("curve:", train_puct_selfplay(iters, games, playouts))


if __name__ == "__main__":
    main()
