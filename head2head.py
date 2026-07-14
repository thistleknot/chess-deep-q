"""Paired head-to-head differential ruler (spec/h2h-instrument.spec.md, charter lane).

Duel two pst-encoded checkpoints: seeded random openings (K plies), each opening played
twice with colors swapped; 1-ply greedy over each net's afterstate V (mate shortcut);
material adjudication at the ply cap. Reports score, Elo diff (measure_elo.elo_diff),
95% Wilson band, decisive rate. No SF, no training, no checkpoint writes.

Usage: python head2head.py A.pt B.pt [games] [tag]
  -> prints one verdict line, appends a row to data/h2h_<tag>.md
Failure modes: non-pst ckpt or missing file -> SystemExit; odd game count is rounded
down to a multiple of 2 (pairing invariant).
"""
import math
import os
import random
import sys

import numpy as np
import torch
import chess

os.environ.setdefault("QLEARN_ENC", "pst")
from qlearn import ValueNet  # noqa: E402
from cem_loop import encode, NIN  # noqa: E402
from measure_elo import elo_diff  # noqa: E402
from search_policy import search_move  # noqa: E402

OPEN_PLIES = 6
PLY_CAP = 160
TAU = 0.02                           # KC-faithful dither: breaks argmax repetition cycles
ADJ_MATERIAL = 1                     # pawns of material = adjudicated win at the cap
# (was 2: at d2 the 2-pawn window drew 88% of champion-vs-585 games — draw flood starved
# the band, G2b came back +37 (-12..+85). 1 pawn declared: between same-class sub-floor
# nets, material at the cap IS the skill differential being measured.)
SEED = 20                            # one seed per battery: identical openings across rungs
VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def load_net(path):
    ck = torch.load(path, map_location="cpu")
    if ck.get("enc", "pst") != "pst":
        raise SystemExit(f"{path}: enc={ck.get('enc')} — instrument requires pst")
    net = ValueNet(ck.get("arch", "linear"), 64, NIN)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net


def evaluator_from_spec(spec):
    """(vf, enc) for a net spec — vf(X)->(n,) WHITE-absolute values, enc(board)->(d,):
    - plain path         -> pst-encoded ValueNet ckpt (linear/mlp)
    - 'nnue:<raw.bin>'   -> bullet (768->H)x2 SCReLU net (nnue_eval.NNUEEval)
    - 'kcz:<ckpt.pt>'    -> kc-809 linear ckpt incl. ZCA back-conversion
                            (identity-gated in corpus_gen.raw_weights)"""
    if spec.startswith("nnue:"):
        from nnue_eval import NNUEEval
        net = NNUEEval(spec[5:])

        def enc(board, _n=net):
            x = np.zeros(1537, dtype=np.float32)
            fs, fn = _n.features(board)
            x[fs] = 1.0
            x[[768 + i for i in fn]] = 1.0
            x[1536] = 1.0 if board.turn == chess.WHITE else 0.0
            return x

        def vf(X, _n=net):
            acc_s = X[:, :768] @ _n.l0w + _n.l0b
            acc_n = X[:, 768:1536] @ _n.l0w + _n.l0b
            h = np.concatenate([np.clip(acc_s, 0, 1) ** 2, np.clip(acc_n, 0, 1) ** 2], axis=1)
            out = h @ _n.l1w + _n.l1b
            return np.where(X[:, 1536] > 0.5, out, -out).astype(np.float64)

        return vf, enc
    if spec.startswith("kcz:"):
        from corpus_gen import raw_weights
        from cem_loop import encode_features
        w, b = raw_weights(spec[4:])
        w = np.asarray(w, dtype=np.float64)

        def vf(X, _w=w, _b=b):
            return np.tanh(X @ _w + _b)

        return vf, encode_features
    net = load_net(spec)

    def vf(X, _n=net):
        with torch.no_grad():
            return _n(torch.from_numpy(X)).numpy().reshape(-1)

    return vf, encode


def mover_from_spec(spec, rng):
    """Mover = evaluator + search policy. Default policy: depth-2 width-8 beam, root
    softmax at TAU=0.02 (the KC-faithful dither). Two pinned failures forced those
    choices (spec :Assert:): 1-ply greedy can't express value quality (champion 0.487
    vs 585-class); pure d2 ARGMAX collapses 29/30 games into fivefold repetition.
    'puct:<sims>:<inner>' swaps the beam for value-only PUCT (spec :MCTS-door:) over
    the inner spec's evaluator — same dither semantics at the root."""
    if spec.startswith("puct:"):
        from puct_value import puct_move
        _, sims_s, inner = spec.split(":", 2)
        sims = int(sims_s)
        vf, enc = evaluator_from_spec(inner)
        return lambda board: puct_move(board, vf, enc, sims, rng)
    vf, enc = evaluator_from_spec(spec)
    return lambda board: search_move(board, vf, TAU, rng, 8, depth=2, encode_fn=enc)


def material_sign(board):
    d = sum(VALS[p] * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
            for p in VALS)
    return 1 if d >= ADJ_MATERIAL else -1 if d <= -ADJ_MATERIAL else 0


def duel_game(white, black, opening):
    """white/black are mover callables (board -> move)."""
    b = chess.Board()
    for mv in opening:
        b.push(mv)
    while not b.is_game_over() and b.ply() < PLY_CAP:
        b.push((white if b.turn == chess.WHITE else black)(b))
    if b.is_game_over():
        r = b.result()
        return 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
    m = material_sign(b)
    return 1.0 if m > 0 else 0.0 if m < 0 else 0.5


def make_openings(n, rng):
    outs = []
    while len(outs) < n:
        b, line = chess.Board(), []
        ok = True
        for _ in range(OPEN_PLIES):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                ok = False
                break
            mv = rng.choice(ms)
            line.append(mv); b.push(mv)
        if ok:
            outs.append(line)
    return outs


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    e = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - e) / d, (c + e) / d


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    games = (int(sys.argv[3]) if len(sys.argv) > 3 else 300) // 2 * 2
    tag = sys.argv[4] if len(sys.argv) > 4 else "adhoc"
    rng = random.Random(SEED)
    A, B = mover_from_spec(a_path, rng), mover_from_spec(b_path, rng)
    score = decisive = 0.0
    for i, opening in enumerate(make_openings(games // 2, rng)):
        for a_white in (True, False):
            s = duel_game(A if a_white else B, B if a_white else A, opening)
            sa = s if a_white else 1.0 - s
            score += sa
            decisive += (s != 0.5)
        if (i + 1) % 25 == 0:
            print(f"  ...{2 * (i + 1)}/{games} games, A score {score / (2 * (i + 1)):.3f}", flush=True)
    p = score / games
    lo, hi = wilson(p, games)
    e, el, eh = elo_diff(p, games), elo_diff(lo, games), elo_diff(hi, games)
    line = (f"h2h {tag}: A={os.path.basename(a_path)} vs B={os.path.basename(b_path)} | "
            f"{games}g score {p:.3f} -> Elo {e:+.0f} (95% {el:+.0f}..{eh:+.0f}) | "
            f"decisive {decisive / games:.0%}")
    print(line)
    os.makedirs("data", exist_ok=True)
    with open(f"data/h2h_{tag}.md", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
