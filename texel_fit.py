"""Arm 1.5 — Texel tuning (spec/intervention-queue.spec.md :Texel:).

Logistic regression of the 809-feature linear eval on OWN-game outcomes over quiet positions:
the non-TD family (pure Monte Carlo — no bootstrap, no traces, no max operators). Generation
is 1-ply epsilon-greedy vs a fixed graded mix; labels are game outcomes only (from-scratch
compliant, no engine evaluations anywhere).

Usage: python texel_fit.py [games] [fit_epochs]
Preconditions: models/qlearn_kc7_best.pt (the donor-informed seed). Failure modes: asserts on
missing seed; Stockfish absent -> mix truncates to heuristic (generation still works).
"""
import os
import sys
import glob
import random
import time

import numpy as np
import torch
import chess
import chess.engine

os.environ.setdefault("QLEARN_ENC", "kc")
from chess_rl import ChessEnv
from cem_loop import encode_features
from measure_ladder import heuristic_mover
from measure_elo import measure
from search_policy import search_move
from qlearn import ValueNet

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
FIT_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PLY_CAP = 160
CKPT = "models/qlearn_texel.pt"


def load_seed():
    ck = torch.load("models/qlearn_kc7_best.pt", map_location="cpu")
    net = ValueNet("linear", nin=809)
    net.load_state_dict(ck["state_dict"])
    return net


def opponent_mix():
    """[(prob, name, move_fn)] — heuristic .4, sf skill 0/2/5 .3/.2/.1; truncates without SF."""
    mix = [(0.4, "heuristic", heuristic_mover)]
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if sfp:
        sf = chess.engine.SimpleEngine.popen_uci(sfp[0], timeout=20.0)
        lim = chess.engine.Limit(time=0.05)
        state = {"cfg": None}

        def mover(cfg):
            def mv(board):
                if state["cfg"] != cfg:
                    sf.configure(cfg)
                    state["cfg"] = cfg
                return sf.play(board, lim).move
            return mv
        mix += [(0.3, "sf-skill0", mover({"UCI_LimitStrength": False, "Skill Level": 0})),
                (0.2, "sf-skill2", mover({"UCI_LimitStrength": False, "Skill Level": 2})),
                (0.1, "sf-skill5", mover({"UCI_LimitStrength": False, "Skill Level": 5}))]
        return mix, sf
    print("no stockfish -> heuristic-only generation", flush=True)
    return [(1.0, "heuristic", heuristic_mover)], None


def main():
    torch.manual_seed(0)
    rng = random.Random(0)
    net = load_seed()
    net.eval()

    @torch.no_grad()
    def value_fn(X):
        return net(torch.from_numpy(np.asarray(X, dtype=np.float32).reshape(-1, 809))).numpy().reshape(-1)

    def agent_move(board, eps):
        moves = list(board.legal_moves)
        if rng.random() < eps:
            return moves[rng.randrange(len(moves))]
        X = np.stack([_after(board, m) for m in moves])
        v = value_fn(X)
        return moves[int(v.argmax() if board.turn == chess.WHITE else v.argmin())]

    def _after(board, mv):
        board.push(mv)
        x = encode_features(board)
        board.pop()
        return x

    mix, sf = opponent_mix()
    env = ChessEnv(ply_cap=PLY_CAP)
    Xs, ys = [], []
    t0 = time.time()
    for g in range(GAMES):
        r = rng.random()
        acc = 0.0
        opp = mix[-1][2]
        for p, _name, fn in mix:
            acc += p
            if r <= acc:
                opp = fn
                break
        aw = (g % 2 == 0)
        board = env.reset()
        rows = []
        z, done, prev_cap = 0.0, False, False
        while not done and env.plies < PLY_CAP:
            my = (board.turn == chess.WHITE) == aw
            mv = agent_move(board, 0.1) if my else opp(board)
            cap = board.is_capture(mv)
            board, z, done = env.step(mv)
            # quiet-position filter (Texel): past the opening, not a capture landing, not in check
            if env.plies >= 12 and not cap and not prev_cap and not board.is_check() and not done:
                rows.append(encode_features(board))
            prev_cap = cap
        y = 0.5 if z == 0.0 else (1.0 if z > 0 else 0.0)       # outcome for WHITE
        Xs += rows
        ys += [y] * len(rows)
        if (g + 1) % 100 == 0:
            print(f"gen {g+1}/{GAMES} games, {len(ys)} quiet positions, {time.time()-t0:.0f}s", flush=True)
    if sf is not None:
        sf.quit()
    X = torch.from_numpy(np.stack(Xs))
    y = torch.tensor(ys, dtype=torch.float32)
    print(f"dataset: {X.shape[0]} positions | outcome mean {y.mean():.3f}", flush=True)

    # --- Texel fit: logloss of sigmoid(K * pre-tanh score) vs outcome ------------------------
    net.train()
    K = torch.nn.Parameter(torch.tensor(4.0))                  # outcome scale (learned)
    opt = torch.optim.Adam(list(net.parameters()) + [K], lr=1e-3)
    bce = torch.nn.BCELoss()
    for e in range(FIT_EPOCHS):
        opt.zero_grad()
        s = net.head(X).reshape(-1)                            # pre-tanh linear score
        p = torch.sigmoid(K * s)
        loss = bce(p, y)
        loss.backward()
        opt.step()
        if e % 50 == 0 or e == FIT_EPOCHS - 1:
            print(f"fit {e}: logloss {loss.item():.4f} K {K.item():.2f}", flush=True)
    net.eval()
    os.makedirs("models", exist_ok=True)
    for p_ in (CKPT, CKPT.replace(".pt", "_best.pt")):
        torch.save({"state_dict": net.state_dict(), "arch": "linear", "enc": "kc",
                    "cum_games": GAMES, "ts": int(time.time())}, p_)
    print(f"saved {CKPT}", flush=True)

    # --- measure: 1-ply greedy and d2 w8 search rungs ----------------------------------------
    @torch.no_grad()
    def vfn(Xa):
        return net(torch.from_numpy(np.asarray(Xa, dtype=np.float32).reshape(-1, 809))).numpy().reshape(-1)

    def greedy1(board):
        moves = list(board.legal_moves)
        v = vfn(np.stack([_after(board, m) for m in moves]))
        return moves[int(v.argmax() if board.turn == chess.WHITE else v.argmin())]

    measure(greedy1, f"texel {GAMES}g outcome-fit 1ply", 60, merge=7)
    measure(lambda b: search_move(b, vfn, 0.0, None, width=8, depth=2, encode_fn=encode_features),
            f"texel {GAMES}g outcome-fit search d2 w8", 60, merge=7)


if __name__ == "__main__":
    main()
