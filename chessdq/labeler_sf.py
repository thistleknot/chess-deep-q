"""Canonical per-process Stockfish labelling worker (Stage-1 teacher distillation).

Spec: teacher-distillation.spec.md. One OS process owns one engine and appends
soft-MultiPV distillation labels to its own shard. It is spawned N times by the
`gen_labels.py` multiprocessing launcher (:Process-separated-labeling:) — never as a
thread inside the trainer (the retired 58 labels/s GIL violation).

What this worker does per position (all reusing ONE MultiPV analyse, no extra search):
  - :Position-source: positions come from TEACHER SELF-PLAY trajectories, not uniform
    random playout. Each game starts from a few random opening plies then Stockfish
    plays it out, the next move sampled via :Trajectory-sampling:.
  - :Trajectory-sampling: the continuing move is a temperature-softmax sample over the
    MultiPV top-K centipawns (mover-relative). Temperature anneals from
    TRAJECTORY_TEMP_OPENING -> TRAJECTORY_TEMP_LATE across TRAJECTORY_OPENING_PLIES,
    keeping trajectories near the strong-play manifold.
  - :Quiet-filter: a position is admitted to the value set only if QUIET — side to move
    not in check AND the teacher's best move is not a capture/check whose eval swing
    (best vs second-best MultiPV cp) exceeds QUIET_MARGIN_CP. A NON_QUIET_SLICE_FRAC
    slice of non-quiet positions is kept as pre-quiescence insurance, flagged
    "non_quiet": true.
  - :Distillation-label: value = tanh(white_cp/400) (White-absolute), wdl = [w,d,l] or
    null, policy = softmax(cp_i / (100*POLICY_TARGET_TEMP)) over the K MultiPV moves as
    [[uci, prob], ...] (one-hot best-move fallback when MultiPV is unavailable).

Emitted row schema (one JSON object per line):
  {"fen": str, "value": float, "wdl": [w,d,l]|null, "policy": [[uci, prob], ...]}
with an extra "non_quiet": true on the insurance slice.

Preconditions: a Stockfish binary exists under engines/**/stockfish*.exe.
Failure modes: value/policy fail fast (no swallow); only the optional WDL is guarded
to null per python-chess not always exposing a WDL model.

Usage (back-compatible arg shape):
    python labeler_sf.py <shard_path> <seconds> <depth> [seed]
"""

import os
import sys
import json
import math
import time
import glob
import random

import chess
import chess.engine

# --- Distillation constants (developer-tuned distillation rules; spec constants req) ---
TEACHER_DEPTH = 8            # fixed shallow teacher depth (~2200+ strength)
MULTIPV_K = 5                # MultiPV candidates per position -> soft policy target
POLICY_TARGET_TEMP = 0.7     # policy softmax temperature; scaled to cp as 100*TEMP (see softmax_cp)
QUIET_MARGIN_CP = 70         # :Quiet-filter: best-vs-second cp swing above which a tactical best move is non-quiet
NON_QUIET_SLICE_FRAC = 0.1   # pre-quiescence insurance: fraction of non-quiet positions kept (flagged)

# --- Trajectory constants (:Trajectory-sampling: annealed temperature) ---
TRAJECTORY_TEMP_OPENING = 1.0   # hot (diverse) sampling early
TRAJECTORY_TEMP_LATE = 0.15     # near-best sampling later
TRAJECTORY_OPENING_PLIES = 20   # plies over which temperature decays opening->late
TRAJECTORY_RANDOM_PLIES = (4, 8)  # inclusive range of random opening plies seeding each game
TRAJECTORY_MAX_PLIES = 120      # ~60 moves: covers the master band (40-45) + engine-length grinding
                                # endgames where material is CONVERTED (the phase 80/40-moves truncated)
MATERIAL_IMBALANCE_FRAC = float(os.environ.get("MATERIAL_IMBALANCE_FRAC", "0.35"))  # :Material-coverage:
#   fraction of games seeded from a material imbalance (proven +362 lever; env-tunable, kept at 0.35 so the
#   ONLY changed variable is EXPLORE_FRAC -- clean attribution of the both-edges lever)
EXPLORE_FRAC = float(os.environ.get("EXPLORE_FRAC", "0.15"))  # :Both-edges: fraction of trajectory moves
#   played UNIFORM-RANDOM over ALL legal moves (not just the MultiPV top-K) so the walk visits BAD /
#   post-blunder positions -- the negative edge the softmax-over-good-moves sampler never reaches. SF then
#   labels those positions with their TRUE value (oracle -> no exploration/greedy-backup bias, unlike
#   model-free RL). softmax(good) + uniform(bad edge) + material seeding = both edges covered.

TARGET_CP_SCALE = 400.0         # tanh(white_cp / 400) value convention
MATE_SCORE = 100000             # cp assigned to forced mate by score(mate_score=...)


def softmax_cp(cps, temp):
    """Numerically stable softmax over centipawn scores at temperature `temp`.

    Centipawns are large (hundreds), so the effective softmax scale is 100*temp:
    a temp of 0.7 makes a 70cp gap an e^-1 mass ratio. Returns a probability list
    aligned with `cps`.
    """
    scale = 100.0 * float(temp)
    m = max(cps)
    exps = [math.exp((c - m) / scale) for c in cps]
    s = sum(exps)
    return [e / s for e in exps]


def anneal_temp(ply):
    """Annealed :Trajectory-sampling: temperature: OPENING at ply 0, decaying linearly
    to LATE at TRAJECTORY_OPENING_PLIES, flat thereafter."""
    frac = min(1.0, ply / float(TRAJECTORY_OPENING_PLIES))
    return TRAJECTORY_TEMP_OPENING + frac * (TRAJECTORY_TEMP_LATE - TRAJECTORY_TEMP_OPENING)


def multipv_candidates(board, infos):
    """Extract (move, mover_relative_cp) candidates from a MultiPV analyse result.

    `infos` is the list python-chess returns for eng.analyse(..., multipv=K), best-first.
    cps are POV of the side to move so higher == better for the mover. Entries without a
    pv or a resolvable score are skipped.
    """
    cands = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        sc = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
        if sc is None:
            continue
        cands.append((pv[0], sc))
    return cands


def wdl_white(pov_score):
    """White-absolute [w, d, l] probabilities from a PovScore, or None.

    Guarded (per task/spec) because python-chess does not always expose a WDL model;
    a missing/zero WDL yields null rather than failing the label.
    """
    try:
        w = pov_score.white().wdl()
        tot = w.wins + w.draws + w.losses
        if tot <= 0:
            return None
        return [w.wins / tot, w.draws / tot, w.losses / tot]
    except Exception:
        return None


def build_label(board, infos):
    """Build a :Distillation-label: dict (without the non_quiet flag) from a MultiPV result.

    Returns (row_dict, cands) where cands are the mover-relative candidates used by the
    quiet filter, or (None, None) if the position has no usable evaluation.
    """
    cands = multipv_candidates(board, infos)
    if not cands:
        return None, None

    white_cp = infos[0]["score"].white().score(mate_score=MATE_SCORE)
    if white_cp is None:
        return None, None
    value = math.tanh(white_cp / TARGET_CP_SCALE)

    wdl = wdl_white(infos[0]["score"])

    if len(cands) >= 2:
        probs = softmax_cp([c for _, c in cands], POLICY_TARGET_TEMP)
        policy = [[m.uci(), float(p)] for (m, _), p in zip(cands, probs)]
    else:
        # MultiPV unavailable (single candidate) -> one-hot best-move fallback.
        policy = [[cands[0][0].uci(), 1.0]]

    row = {"fen": board.fen(), "value": value, "wdl": wdl, "policy": policy}
    return row, cands


def is_quiet(board, cands):
    """:Quiet-filter: True iff the position's value is well-defined without search.

    Quiet requires: side to move NOT in check, and the teacher's best move is not a
    capture/check whose eval swing (best minus second-best mover cp) exceeds
    QUIET_MARGIN_CP. A lone forced move that is tactical is treated as non-quiet.
    Reuses the already-computed MultiPV candidates — no extra search.
    """
    if board.is_check():
        return False
    best = cands[0][0]
    tactical = board.is_capture(best) or board.gives_check(best)
    if not tactical:
        return True
    if len(cands) >= 2:
        swing = cands[0][1] - cands[1][1]
    else:
        swing = MATE_SCORE  # single forced tactical move -> non-quiet
    return swing <= QUIET_MARGIN_CP


def sample_move(board, cands, rng, temp):
    """:Trajectory-sampling: temperature-softmax sample over the MultiPV candidate moves."""
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0][0]
    probs = softmax_cp([c for _, c in cands], temp)
    r = rng.random()
    acc = 0.0
    for (move, _), p in zip(cands, probs):
        acc += p
        if r <= acc:
            return move
    return cands[-1][0]


def perturb_material(board, rng):
    """:Material-coverage: remove 1-3 random non-king pieces to seed a material-imbalanced but LEGAL
    start, so trajectories carry the decisive-material quiet positions balanced SF-vs-SF play omits.
    Each removal is reverted if it makes the position invalid (e.g. leaves the non-mover in check)."""
    n_remove = int(rng.choice([1, 1, 2, 3]))
    squares = [sq for sq, p in board.piece_map().items() if p.piece_type != chess.KING]
    rng.shuffle(squares)
    removed = 0
    for sq in squares:
        if removed >= n_remove:
            break
        piece = board.remove_piece_at(sq)
        if board.is_valid():
            removed += 1
        else:
            board.set_piece_at(sq, piece)   # revert an illegal removal
    return board


def new_game(rng):
    """Seed a fresh :Position-source: game with TRAJECTORY_RANDOM_PLIES random opening plies, and with
    probability MATERIAL_IMBALANCE_FRAC seed a :Material-coverage: material imbalance."""
    board = chess.Board()
    n = rng.randint(*TRAJECTORY_RANDOM_PLIES)
    for _ in range(n):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    if rng.random() < MATERIAL_IMBALANCE_FRAC and not board.is_game_over():
        perturb_material(board, rng)
    return board


def run_worker(shard, seconds, depth, seed=0):
    """Label teacher-self-play positions into `shard` for `seconds`, returning label count.

    Importable entrypoint used by gen_labels.py (multiprocessing) and by tests.
    Emits the dict schema; applies :Quiet-filter: with NON_QUIET_SLICE_FRAC insurance.
    """
    rng = random.Random(seed or None)
    sf_path = glob.glob("engines/**/stockfish*.exe", recursive=True)[0]
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 1, "Hash": 16})
    limit = chess.engine.Limit(depth=depth)

    n = 0
    start = time.time()
    board = new_game(rng)
    ply = 0
    with open(shard, "a") as fh:
        while time.time() - start < seconds:
            # 50-move rule ("boxing decision"): no capture/pawn move in 50 moves -> draw, so games run
            # long ONLY while making progress (conversion) and end the moment they stall to shuffling.
            if board.is_game_over() or board.can_claim_fifty_moves() or ply >= TRAJECTORY_MAX_PLIES:
                board = new_game(rng)
                ply = 0
                continue

            infos = eng.analyse(board, limit, multipv=MULTIPV_K)
            row, cands = build_label(board, infos)
            if row is None:
                # unusable eval (e.g. terminal) -> just advance the game
                board = new_game(rng)
                ply = 0
                continue

            quiet = is_quiet(board, cands)
            keep = quiet or (rng.random() < NON_QUIET_SLICE_FRAC)
            if keep:
                if not quiet:
                    row["non_quiet"] = True
                fh.write(json.dumps(row) + "\n")
                n += 1
                if n % 200 == 0:
                    fh.flush()

            # :Both-edges: with prob EXPLORE_FRAC play a uniform-random legal move (reach a BAD position
            # SF labels next iteration) instead of the softmax-over-MultiPV good-move sample. Fills the
            # negative-coverage holes; SF is the oracle so the off-distribution label is still true.
            if EXPLORE_FRAC > 0 and rng.random() < EXPLORE_FRAC:
                legal = list(board.legal_moves)
                move = rng.choice(legal) if legal else None
            else:
                move = sample_move(board, cands, rng, anneal_temp(ply))
            if move is None:
                board = new_game(rng)
                ply = 0
                continue
            board.push(move)
            ply += 1

    eng.quit()
    return n


def main():
    shard = sys.argv[1]
    seconds = float(sys.argv[2])
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else TEACHER_DEPTH
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    start = time.time()
    n = run_worker(shard, seconds, depth, seed)
    print(f"{shard}: {n} labels in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
