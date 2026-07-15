""":Played-buffer: training lane (operator 2026-07-14) — learn from human-vs-champion
games, KnightCap's original FICS recipe (spec/bullet-route.spec.md :Operator-ideas-ledger:).

Purity: the human is an OPPONENT (same legal class as the SF ladder) — their moves shape
which states are visited; every LABEL is self-generated: the lineage net's own d2 search
values blended with the game outcome via the proven trivium weights (0.285 λ-return /
0.341 search / 0.374 outcome; λ=0.7, γ=1 — KC-faithful constants, declared).

Pipeline: data/human_games/*.pgn -> per-position (whitened-x, G) pairs -> online SGD
fine-tune of a COPY of the champion -> models/champion_hb.pt. Promotion is NEVER
automatic: the candidate must beat the champion on the duel ruler, then the ladder.

Usage: python human_replay.py [pgn_dir] [ckpt_in] [ckpt_out]
Failure modes: no PGNs -> exits 0 with a count of 0 (nothing to learn); ZCA identity
failure -> SystemExit (corpus_gen.raw_weights); non-kc ckpt -> assert.
"""
import glob
import math
import os
import sys

import numpy as np
import torch
import chess
import chess.pgn

from corpus_gen import raw_weights
from cem_loop import encode_features

TRIV = (0.285, 0.341, 0.374)   # λ-return / search / outcome (proven start weights)
LAM = 0.7                      # KC-faithful trace
ALPHA = 3e-4
SEARCH_D = 2


def game_pairs(game, srch, Z, mu):
    """(whitened_x, G) pairs for every position of one game, labels self-generated."""
    board = game.board()
    xs, vs = [], []
    for mv in game.mainline_moves():
        xs.append(Z @ (encode_features(board).astype(np.float32) - mu))
        vs.append(float(srch.search(board.fen(), SEARCH_D)[1]))
        board.push(mv)
    res = game.headers.get("Result", "*")
    z = 1.0 if res == "1-0" else -1.0 if res == "0-1" else 0.0
    if not xs:
        return []
    # λ-return over the net's OWN search values along the actual trajectory (γ=1)
    G_lam = [0.0] * len(vs)
    running = z
    for t in range(len(vs) - 1, -1, -1):
        nxt = vs[t + 1] if t + 1 < len(vs) else z
        running = (1 - LAM) * nxt + LAM * running
        G_lam[t] = running
    a, b, c = TRIV
    return [(x, a * gl + b * v + c * z) for x, gl, v in zip(xs, G_lam, vs)]


def main():
    pgn_dir = sys.argv[1] if len(sys.argv) > 1 else "data/human_games"
    ck_in = sys.argv[2] if len(sys.argv) > 2 else "models/champion.pt"
    ck_out = sys.argv[3] if len(sys.argv) > 3 else "models/champion_hb.pt"
    files = sorted(glob.glob(os.path.join(pgn_dir, "*.pgn")))
    print(f"played buffer: {len(files)} game(s) in {pgn_dir}")
    if not files:
        return
    import importlib
    from qlearn import ValueNet
    w, bias = raw_weights(ck_in)                      # identity-gated raw view for search
    srch = importlib.import_module("rsearch4").Searcher(w, bias)
    zf = np.load("models/zca.npz")
    Z, mu = zf["Z"].astype(np.float32), zf["mu"].astype(np.float32)
    ck = torch.load(ck_in, map_location="cpu")
    assert ck.get("enc") == "kc" and ck.get("zca"), "played-buffer lane expects the kc+ZCA champion"
    net = ValueNet("linear", 64, Z.shape[0])
    net.load_state_dict(ck["state_dict"])
    opt = torch.optim.SGD(net.parameters(), lr=ALPHA)
    total, first_loss, last_loss = 0, None, None
    for path in files:
        with open(path) as fh:
            game = chess.pgn.read_game(fh)
        if game is None:
            continue
        pairs = game_pairs(game, srch, Z, mu)
        if not pairs:
            continue
        X = torch.from_numpy(np.stack([p[0] for p in pairs]))
        y = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        pred = net(X)
        loss = torch.mean((pred - y) ** 2)
        if first_loss is None:
            first_loss = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(torch.mean((net(X) - y) ** 2))
        total += len(pairs)
        print(f"  {os.path.basename(path)}: {len(pairs)} positions, "
              f"loss {float(loss):.5f} -> {last_loss:.5f}")
    torch.save({"state_dict": net.state_dict(), "arch": "linear", "enc": "kc", "zca": True,
                "cum_games": int(ck.get("cum_games", 0)) + len(files),
                "hb_source": pgn_dir}, ck_out)
    print(f"candidate written: {ck_out} ({total} positions from {len(files)} games) — "
          f"promotion requires beating the champion on the duel ruler, never automatic")


if __name__ == "__main__":
    main()
