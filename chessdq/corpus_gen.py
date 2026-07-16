"""Self-play corpus generator for the bullet route (goal-1600, spec/sota-notes.md).

Generates games with rsearch4.play_games (native, threaded) using the VOLUME net (pure
self-play lineage, 1484 claims) as both sides, and writes bullet's text format:

    <FEN> | <score> | <result>

- FEN = the TDLeaf SEARCH-LEAF position (quasi-quiet, the NNUE-practice choice)
- score = white-relative centipawns from the net's own backed search value:
  cp = round(800*atanh(v)) so that sigmoid(cp/400) == (v+1)/2 EXACTLY (declared
  scale 400; bullet trains sigmoid(cp/scale)); clipped to ±3000
- result = (z+1)/2, z White-absolute from the native generator (+1/0/-1)

Purity: every field is self-generated (own net, own search, own games) — no external
label source. ZCA back-conversion (w_raw = Z.T @ w', b_raw = b' - w_raw . mu) is gated
by an EXACT identity check (whitened-net value == raw-weight value on a battery); the
right npz is auto-selected between models/zca.npz and models/zca_self.npz by that gate.

Usage: python corpus_gen.py <n_games> <out.txt> [batch=500] [threads=8]
Resume: appends; a sidecar <out>.count tracks games done (load-if-exists checkpoint).
Failure modes: no npz passes the identity gate -> SystemExit; weights not 809 -> native
raises ValueError.
"""
import importlib
import math
import os
import sys

import numpy as np
import torch
import chess

os.environ.setdefault("QLEARN_ENC", "kc")
from chessdq.cem_loop import encode_features  # noqa: E402 — 809-dim kc encoder

CKPT = os.environ.get("CORPUS_CKPT", "models/qlearn_vol_best.pt")
DEPTH = int(os.environ.get("CORPUS_DEPTH", "2"))       # agent search depth (TDLeaf d2, lane parity)
OPP_DEPTH = int(os.environ.get("CORPUS_OPP_DEPTH", "2"))
EPS = float(os.environ.get("CORPUS_EPS", "0.1"))       # volume-lane exploration (state coverage)
PLY_CAP = 160
SEED0 = 977


def raw_weights(ckpt=None):
    """(w_raw, b_raw) in native-eval space: kc-809 (identity-gated through ZCA when the
    ckpt is whitened) or amap-897 (:Native-amap-port: 2026-07-15 — the confirmed
    feature winner rides rsearch4 v3.7 directly; amap lineages train unwhitened)."""
    ck = torch.load(ckpt or CKPT, map_location="cpu")
    assert ck.get("enc") in ("kc", "amap") and ck.get("arch", "linear") == "linear", ck.get("enc")
    w = ck["state_dict"]["head.weight"].numpy().reshape(-1).astype(np.float64)
    b = float(ck["state_dict"]["head.bias"].numpy().reshape(-1)[0])
    if ck.get("enc") == "amap":
        assert not ck.get("zca"), "amap lineages are unwhitened (no 897-dim ZCA exists)"
        assert w.size == 897, w.size
        return list(w), b
    if not ck.get("zca"):
        return list(w), b
    battery = [chess.Board(),
               chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6"),
               chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")]
    for path in ("models/zca.npz", "models/zca_self.npz"):
        z = np.load(path)
        Z, mu = z["Z"].astype(np.float64), z["mu"].astype(np.float64)
        w_raw = Z.T @ w
        b_raw = b - float(w_raw @ mu)
        ok = all(abs((w @ (Z @ (encode_features(bd).astype(np.float64) - mu)) + b)
                     - (w_raw @ encode_features(bd).astype(np.float64) + b_raw)) < 1e-6
                 for bd in battery)
        if ok:
            print(f"ZCA identity gate PASS with {path}")
            return list(w_raw), b_raw
    raise SystemExit("no ZCA npz reproduces the whitened net — conversion unsafe, aborting")


def cp_of(v):
    v = max(-0.9997, min(0.9997, v))
    return max(-3000, min(3000, round(800.0 * math.atanh(v))))


def main():
    n_games = int(sys.argv[1])
    out = sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    pg = importlib.import_module("rsearch4").play_games
    gw, gb = raw_weights()
    cnt_path = out + ".count"
    done = int(open(cnt_path).read()) if os.path.exists(cnt_path) else 0
    print(f"corpus: {done}/{n_games} games already done -> resuming", flush=True)
    positions = 0
    with open(out, "a", encoding="ascii") as fh:
        while done < n_games:
            n = min(batch, n_games - done)
            games = pg(gw, gb, gw, gb, OPP_DEPTH, n, threads, DEPTH, EPS, PLY_CAP,
                       SEED0 + done * 1_000_003, 0.0)
            dec = 0
            for z, _aw, recs in games:
                res = (z + 1.0) / 2.0
                dec += (z != 0.0)
                for fen, v, _pred in recs:
                    fh.write(f"{fen} | {cp_of(v)} | {res:.1f}\n")
                    positions += 1
            fh.flush()
            done += n
            open(cnt_path, "w").write(str(done))
            print(f"  {done}/{n_games} games | +{positions} positions this run | "
                  f"decisive {dec}/{n}", flush=True)
    print(f"corpus complete: {done} games -> {out}", flush=True)


if __name__ == "__main__":
    main()
