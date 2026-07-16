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
from chessdq.qlearn import ValueNet  # noqa: E402
from chessdq.cem_loop import encode, NIN  # noqa: E402
from chessdq.measure_elo import elo_diff  # noqa: E402
from chessdq.search_policy import search_move  # noqa: E402

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
        from chessdq.nnue_eval import NNUEEval
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
    if spec.startswith("enc:"):
        # any registered encoder + its pst-family linear/mlp ckpt: 'enc:<name>:<ckpt>'
        _, enc_name, ckpt = spec.split(":", 2)
        from chessdq.encoders import get as enc_get
        enc_fn, nin = enc_get(enc_name)
        ck = torch.load(ckpt, map_location="cpu")
        net = ValueNet(ck.get("arch", "linear"), 64, nin)
        net.load_state_dict(ck["state_dict"]); net.eval()

        def vf(X, _n=net):
            with torch.no_grad():
                return _n(torch.from_numpy(X)).numpy().reshape(-1)

        return vf, enc_fn
    if spec.startswith("kcz:"):
        from chessdq.corpus_gen import raw_weights
        from chessdq.cem_loop import encode_features
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
        from chessdq.puct_value import puct_move
        _, sims_s, inner = spec.split(":", 2)
        sims = int(sims_s)
        vf, enc = evaluator_from_spec(inner)
        return lambda board: puct_move(board, vf, enc, sims, rng)
    if spec.startswith("rs:"):
        # native alpha-beta at depth D over a kc-809 linear ckpt — the incumbent's
        # own vehicle (deterministic; a losing side steering into repetition is
        # legitimate play, so no dither on this side)
        import importlib
        _, d_s, inner = spec.split(":", 2)
        depth = int(d_s)
        assert inner.startswith("kcz:"), "rs: mover needs a kcz:<ckpt> inner spec"
        from chessdq.corpus_gen import raw_weights
        w, b = raw_weights(inner[4:])
        srch = importlib.import_module("rsearch4").Searcher(w, b)
        return lambda board: chess.Move.from_uci(srch.search(board.fen(), depth)[0])
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


def _snapshot_spec(spec, run_dir):
    """:Input-snapshot: — copy every file-path token of a mover spec into run_dir and
    point the spec at the frozen copies, so a concurrent trainer writing the source
    ckpt can NEVER touch a running duel (the 2026-07-15 zombie-retrain contamination
    class). Returns (snapped_spec, tags) with tags like ["name.pt#sha8"] for the
    verdict line. Failure mode: absolute Windows paths with drive colons are not
    split-safe — duel specs use repo-relative paths by convention."""
    import hashlib
    import shutil
    parts = [spec] if os.path.isfile(spec) else spec.split(":")
    tags = []
    for i, tok in enumerate(parts):
        if tok and os.path.isfile(tok):
            os.makedirs(run_dir, exist_ok=True)
            dst = os.path.join(run_dir, os.path.basename(tok))
            shutil.copy2(tok, dst)
            with open(dst, "rb") as fh:
                h = hashlib.sha1(fh.read()).hexdigest()[:8]
            tags.append(f"{os.path.basename(tok)}#{h}")
            parts[i] = dst
    return ":".join(parts), tags


def _shard(args):
    """One parallel worker's slice: `pairs` paired openings x 2 colors, movers rebuilt
    in-process from the specs, openings drawn from a shard-DISJOINT rng (seed offset) so
    no two shards replay the same deterministic games. Returns (score, decisive)."""
    a_spec, b_spec, pairs, seed, shard = args
    try:
        import torch
        torch.set_num_threads(1)               # one core per worker; search dominates
    except Exception:
        pass
    rng = random.Random(seed)
    A, B = mover_from_spec(a_spec, rng), mover_from_spec(b_spec, rng)
    score = decisive = 0.0
    for opening in make_openings(pairs, rng):
        for a_white in (True, False):
            s = duel_game(A if a_white else B, B if a_white else A, opening)
            score += s if a_white else 1.0 - s
            decisive += (s != 0.5)
    print(f"  shard {shard}: {2 * pairs}g done, A score {score / (2 * pairs):.3f}", flush=True)
    return score, decisive


def main():
    from chessdq.thermal import engage
    engage()                    # :Thermal-guard: — cap cores/priority, children inherit
    a_path, b_path = sys.argv[1], sys.argv[2]
    games = (int(sys.argv[3]) if len(sys.argv) > 3 else 300) // 2 * 2
    # :Games-cap: (operator law 2026-07-15): "explorations with 20, full run with 50,
    # that's it — if you are just trying to measure elo, 50 is sufficient." The cap
    # clamps ANY requested games count; going past it requires the operator explicitly
    # setting H2H_CAP. Verdict lines always print the 95% band, so a 50g verdict
    # honestly carries its ~±97 Elo resolution.
    games = min(games, max(2, int(os.environ.get("H2H_CAP", "50")) // 2 * 2))
    tag = sys.argv[4] if len(sys.argv) > 4 else "adhoc"
    run_dir = os.path.join("data", "runs", tag)
    a_snap, a_tags = _snapshot_spec(a_path, run_dir)   # :Input-snapshot: — movers load
    b_snap, b_tags = _snapshot_spec(b_path, run_dir)   # frozen copies, never live files
    intag = " | in " + " ".join(a_tags + b_tags) if (a_tags or b_tags) else ""
    # :Sharded-duel: + :Group-sequential: (operator 2026-07-15 — "cap the games at no
    # more than 50"): games are independent at fixed depth (no clocks -> contention
    # cannot bias them), so each 50-game BLOCK fans out over H2H_SHARDS workers, and the
    # duel STOPS at the earliest block with an answer:
    #   - stop WIN/LOSS: interim band (z=2.29, Pocock bound for repeated looks) excludes 0
    #   - stop TIE (futility): >=300 games, band contains 0, |point| < 15 Elo
    #   - hard cap: the games argument (spend past 50 happens ONLY while inconclusive)
    # Paired openings preserved within shards; disjoint rng streams per (round, shard).
    # H2H_SHARDS=1 recovers the serial path exactly as before.
    shards = max(1, int(os.environ.get("H2H_SHARDS", "6")))
    if shards > 1:
        block = max(2, int(os.environ.get("H2H_BLOCK", "50"))) // 2 * 2
        import multiprocessing as mp
        score = decisive = 0.0
        played = 0
        stop = ""
        print(f"h2h {tag}: blocks of {block}g x {shards} shards, cap {games}g, "
              f"sequential stop", flush=True)
        with mp.Pool(shards) as pool:
            for rnd in range(max(1, games // block)):
                pairs = min(block, games - played) // 2
                if pairs <= 0:
                    break
                split = [pairs // shards + (1 if k < pairs % shards else 0)
                         for k in range(shards)]
                jobs = [(a_snap, b_snap, n, SEED + 1000 * (rnd * 64 + k + 1), k)
                        for k, n in enumerate(split) if n]
                parts = pool.map(_shard, jobs)
                score += sum(p_[0] for p_ in parts)
                decisive += sum(p_[1] for p_ in parts)
                played += 2 * pairs
                pi_ = score / played
                ilo, ihi = wilson(pi_, played, z=2.29)
                e_pt = elo_diff(pi_, played)
                print(f"  block {rnd + 1}: {played}g, A score {pi_:.3f} | interim Elo "
                      f"{e_pt:+.0f} ({elo_diff(ilo, played):+.0f}..{elo_diff(ihi, played):+.0f})",
                      flush=True)
                if ilo > 0.5 or ihi < 0.5:
                    stop = f" [early stop: band excludes zero @ {played}g]"
                    break
                if played >= 300 and abs(e_pt) < 15:
                    stop = f" [futility stop: tie @ {played}g]"
                    break
        p = score / played
        lo, hi = wilson(p, played)
        e, el, eh = elo_diff(p, played), elo_diff(lo, played), elo_diff(hi, played)
        line = (f"h2h {tag}: A={os.path.basename(a_path)} vs B={os.path.basename(b_path)} | "
                f"{played}g score {p:.3f} -> Elo {e:+.0f} (95% {el:+.0f}..{eh:+.0f}) | "
                f"decisive {decisive / played:.0%}{stop}{intag}")
        print(line)
        os.makedirs("data", exist_ok=True)
        with open(f"data/h2h_{tag}.md", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return
    rng = random.Random(SEED)
    A, B = mover_from_spec(a_snap, rng), mover_from_spec(b_snap, rng)
    score = decisive = 0.0
    for i, opening in enumerate(make_openings(games // 2, rng)):
        for a_white in (True, False):
            s = duel_game(A if a_white else B, B if a_white else A, opening)
            sa = s if a_white else 1.0 - s
            score += sa
            decisive += (s != 0.5)
        if (i + 1) % 25 == 0:
            # :Interim-band: — running Wilson band at every checkpoint. Repeated looks
            # inflate type-I error, so interim bands use z=2.29 (Pocock-style bound);
            # ONLY the final 95% band is the pre-registered verdict.
            gp = 2 * (i + 1)
            pi_ = score / gp
            ilo, ihi = wilson(pi_, gp, z=2.29)
            print(f"  ...{gp}/{games} games, A score {pi_:.3f} | interim Elo "
                  f"{elo_diff(pi_, gp):+.0f} ({elo_diff(ilo, gp):+.0f}..{elo_diff(ihi, gp):+.0f})"
                  + ("  [interim band excludes zero]" if ilo > 0.5 or ihi < 0.5 else ""),
                  flush=True)
    p = score / games
    lo, hi = wilson(p, games)
    e, el, eh = elo_diff(p, games), elo_diff(lo, games), elo_diff(hi, games)
    line = (f"h2h {tag}: A={os.path.basename(a_path)} vs B={os.path.basename(b_path)} | "
            f"{games}g score {p:.3f} -> Elo {e:+.0f} (95% {el:+.0f}..{eh:+.0f}) | "
            f"decisive {decisive / games:.0%}{intag}")
    print(line)
    os.makedirs("data", exist_ok=True)
    with open(f"data/h2h_{tag}.md", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
