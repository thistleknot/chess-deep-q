""":Played-buffer: training lane (operator 2026-07-14) — learn from human-vs-champion
games, KnightCap's original FICS recipe (spec/bullet-route.spec.md :Operator-ideas-ledger:).

Purity: the human is an OPPONENT (same legal class as the SF ladder) — their moves shape
which states are visited; every LABEL is self-generated: the lineage net's own d2 search
values blended with the game outcome via the proven trivium weights (0.285 λ-return /
0.341 search / 0.374 outcome; λ=0.7, γ=1 — KC-faithful constants, declared).

Pipeline: data/human_games/*.pgn -> per-position (x, G) pairs -> online SGD fine-tune
of a COPY of the champion -> models/champion_hb.pt. Promotion is NEVER automatic: the
candidate must beat the champion on the duel ruler, then the ladder.

Encoder-generic (2026-07-18 repair — the lane was hard-wired to the retired kc+ZCA
lineage and asserted out on the amap champion): the ckpt's own `enc` picks the encoder;
`zca` ckpts train in whitened space exactly as before (kc path unchanged), unwhitened
lineages (amap-897) train in raw feature space. Output ckpt carries the input's
enc/zca metadata, so the duel ruler and native port load the candidate unchanged.

Per-game FIRST-PASS LOSS is the surprise-mass telemetry: how wrong the champion was
about the states this game steered it into (high = the human exposed holes).

Usage: python human_replay.py [pgn_dir] [ckpt_in] [ckpt_out]
Failure modes: no PGNs -> exits 0 with a count of 0 (nothing to learn); ZCA identity
failure -> SystemExit (corpus_gen.raw_weights); enc not kc/amap or arch not linear
-> assert (raw_weights constraint).
"""
import glob
import os
import sys

import numpy as np
import torch
import chess
import chess.pgn

from chessdq.corpus_gen import raw_weights
from chessdq.encoders import get as enc_get

TRIV = (0.285, 0.341, 0.374)   # λ-return / search / outcome (proven start weights)
LAM = 0.7                      # KC-faithful trace
ALPHA = 3e-4
SEARCH_D = 2


def game_pairs(game, srch, feat_fn):
    """(x, G) pairs for every position of one game, labels self-generated.
    feat_fn(board) -> the net's input-space features (whitened for zca ckpts)."""
    board = game.board()
    xs, vs = [], []
    for mv in game.mainline_moves():
        xs.append(feat_fn(board))
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
    from chessdq.qlearn import ValueNet
    ck = torch.load(ck_in, map_location="cpu")
    enc_name = ck.get("enc", "kc")
    assert enc_name in ("kc", "amap") and ck.get("arch", "linear") == "linear", \
        f"played-buffer lane needs a kc/amap linear champion, got {enc_name}/{ck.get('arch')}"
    w, bias = raw_weights(ck_in)                      # identity-gated raw view for search
    srch = importlib.import_module("rsearch4").Searcher(w, bias)
    enc_fn, nin = enc_get(enc_name)
    if ck.get("zca"):
        zf = np.load("models/zca.npz")
        Z, mu = zf["Z"].astype(np.float32), zf["mu"].astype(np.float32)
        feat_fn = lambda bd: Z @ (enc_fn(bd).astype(np.float32) - mu)
        net_in = Z.shape[0]
    else:
        feat_fn = lambda bd: enc_fn(bd).astype(np.float32)
        net_in = nin
    net = ValueNet("linear", 64, net_in)
    net.load_state_dict(ck["state_dict"])
    opt = torch.optim.SGD(net.parameters(), lr=ALPHA)
    total, surprise = 0, 0.0
    for path in files:
        with open(path) as fh:
            game = chess.pgn.read_game(fh)
        if game is None:
            continue
        pairs = game_pairs(game, srch, feat_fn)
        if not pairs:
            continue
        X = torch.from_numpy(np.stack([p[0] for p in pairs]))
        y = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        pred = net(X)
        loss = torch.mean((pred - y) ** 2)
        surprise += loss.item() * len(pairs)
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(torch.mean((net(X) - y) ** 2))
        total += len(pairs)
        print(f"  {os.path.basename(path)}: {len(pairs)} positions, "
              f"surprise {float(loss):.5f} -> {last_loss:.5f}")
    torch.save({"state_dict": net.state_dict(), "arch": "linear", "enc": enc_name,
                "zca": ck.get("zca"), "cum_games": int(ck.get("cum_games", 0)) + len(files),
                "hb_source": pgn_dir}, ck_out)
    print(f"candidate written: {ck_out} ({total} positions from {len(files)} games, "
          f"mean surprise {surprise / max(1, total):.5f}) — promotion requires beating "
          f"the champion on the duel ruler, never automatic")


if __name__ == "__main__":
    main()
