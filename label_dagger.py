"""DAgger adversarial labelling worker (Stage-1 teacher distillation, :DAGGER-aggregation:).

Spec: teacher-distillation.spec.md (:DAGGER-aggregation:, operationalizing :Search-visited-
positions:). One OS process owns one Stockfish engine AND one agent AlphaBetaEngine, and appends
soft-MultiPV distillation labels for the positions the AGENT's OWN search walks into. Spawned N
times by the `gen_labels.py` launcher (set LABELER_MODULE=label_dagger.py) exactly like
`labeler_sf.py`; the emitted row schema is identical, so shards merge into the same
:Cumulative-dataset: (data/distill_sf.jsonl) and train with the same loader.

WHY (RL_FINDINGS coda 8 / depth-amplifies-holes): the eval scores some positions falsely-high and
its OWN deep search steers into them (champion 1467@d4 collapses to 926@d11). Those positions are
OFF the SF-self-play + random-explore training distribution = distribution shift. DAgger closes the
gap: run the CURRENT agent, label the states IT visits with the expert (Stockfish), retrain — so the
eval learns the true value of the exact positions its search walks into.

The ONLY difference from `labeler_sf.py` (whose `build_label`, `is_quiet`, `new_game`, `softmax_cp`
and constants are imported UNCHANGED) is the trajectory move source:
  - labeler_sf : the continuing move is a temperature-softmax sample over the teacher's MultiPV
                 (:Trajectory-sampling:), or a uniform-random explore move.
  - label_dagger: the continuing move is the AGENT's own search move (`eng.search(board)[0]`) using
                 the deployed NNUE eval + phi_widen/selective search profile. NOT the teacher's move,
                 NOT random — this is the DAgger difference: visit the agent's OWN (hole) positions.
Per position Stockfish still supplies the correct label (β=0 DAgger: pure learner rollout, expert-
only labels), so the corpus covers the agent-visited distribution with true values.

Emitted row schema (identical to labeler_sf.py, one JSON object per line):
  {"fen": str, "value": float, "wdl": [w,d,l]|null, "policy": [[uci, prob], ...]}
with an extra "non_quiet": true on the pre-quiescence insurance slice.

Preconditions: a Stockfish binary under engines/**/stockfish*.exe AND an NNUE model at NNUE_MODEL
(env, default models/nnue.pt). Failure modes: value/policy fail fast (no swallow); only the optional
WDL is guarded to null. If the loaded eval is not the incremental NNUE, run_worker asserts (the
labeller must trace the DEPLOYED agent's distribution, not a placeholder).

Env knobs:
  NNUE_MODEL   -- path to the current eval whose holes are being covered (default models/nnue.pt)
  DAGGER_DEPTH -- agent search max_depth, the feasible-speed hole proxy (default 8; d11 too slow)
  plus every labeler_sf.py env (MATERIAL_IMBALANCE_FRAC, etc.) since new_game is imported.

Usage (back-compatible arg shape — same as labeler_sf.py; the trailing depth is the SF TEACHER
depth, the agent depth comes from DAGGER_DEPTH):
    python label_dagger.py <shard_path> <seconds> <sf_depth> [seed]
"""

import os
import sys
import json
import time
import glob
import random

import chess
import chess.engine

# Reuse labeler_sf's label/quiet/seed machinery UNCHANGED — the ONLY new logic here is the
# agent-picks-the-move trajectory. Importing (not duplicating) keeps the row schema and the
# :Quiet-filter: / :Position-source: behaviour bit-identical, so shards merge into one corpus.
from labeler_sf import (
    build_label,
    is_quiet,
    new_game,
    TEACHER_DEPTH,
    MULTIPV_K,
    NON_QUIET_SLICE_FRAC,
    TRAJECTORY_MAX_PLIES,
)

from nnue_model import load_nnue, make_incremental_nnue_eval, IncrementalNNUE
from engine import AlphaBetaEngine

# --- :DAGGER-aggregation: constants (spec/teacher-distillation.spec.md) ---
NNUE_MODEL = os.environ.get("NNUE_MODEL", "models/nnue.pt")   # current eval whose holes we cover
AGENT_DEPTH = int(os.environ.get("DAGGER_DEPTH", "8"))        # agent search depth (feasible hole proxy)


def build_agent(depth=AGENT_DEPTH, model_path=NNUE_MODEL):
    """Construct the DAgger AGENT engine: the DEPLOYED AlphaBetaEngine over the current NNUE eval.

    Require: an NNUE checkpoint at `model_path`.
    Guarantee: returns an AlphaBetaEngine whose eval_fn is the incremental NNUE and whose search
      profile matches the deployed agent (phi_widen + selective spearhead, depth-bounded), so the
      positions it walks into ARE the inference-time (hole) distribution DAgger must label.
    Assert: the eval_fn is an IncrementalNNUE (not the pst fallback) — the labeller must trace the
      real agent, so a mis-load fails fast rather than silently labelling a placeholder's walk.
    """
    net = load_nnue(model_path, "cpu")
    eval_fn = make_incremental_nnue_eval(net, "cpu")
    assert isinstance(eval_fn, IncrementalNNUE), "DAgger agent eval must be the incremental NNUE"
    eng = AlphaBetaEngine(
        eval_fn=eval_fn,
        time_limit=1e9,        # depth-bounded, not wall-clock-bounded (one selective pass to max_depth)
        max_depth=depth,
        phi_widen=True,
        selective=True,
    )
    return eng


def run_worker(shard, seconds, depth, seed=0):
    """Label AGENT-visited positions into `shard` for `seconds`, returning label count.

    Importable entrypoint used by gen_labels.py (multiprocessing, via LABELER_MODULE) and tests.
    `depth` is the Stockfish TEACHER depth (arg-shape parity with labeler_sf.run_worker); the AGENT
    search depth is AGENT_DEPTH (env DAGGER_DEPTH). Emits the SAME dict schema as labeler_sf; applies
    the same :Quiet-filter: with NON_QUIET_SLICE_FRAC insurance. :DAGGER-aggregation: difference: the
    trajectory's continuing move is `eng.search(board)[0]` (the agent), not a teacher/random sample.
    """
    rng = random.Random(seed or None)

    # Expert (Stockfish) — the label oracle, identical setup to labeler_sf.run_worker.
    sf_path = glob.glob("engines/**/stockfish*.exe", recursive=True)[0]
    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    sf.configure({"Threads": 1, "Hash": 16})
    limit = chess.engine.Limit(depth=depth)

    # Learner (deployed agent) — chooses the trajectory move (the DAgger difference).
    eng = build_agent()
    print(f"[label_dagger] agent eval={type(eng.eval_fn).__name__} agent_depth={AGENT_DEPTH} "
          f"sf_depth={depth} model={NNUE_MODEL}", flush=True)

    n = 0
    start = time.time()
    board = new_game(rng)
    ply = 0
    with open(shard, "a") as fh:
        while time.time() - start < seconds:
            # 50-move rule ("boxing decision"): games run long ONLY while making progress and end the
            # moment they stall — identical reset conditions to labeler_sf.run_worker.
            if board.is_game_over() or board.can_claim_fifty_moves() or ply >= TRAJECTORY_MAX_PLIES:
                board = new_game(rng)
                ply = 0
                continue

            # (a) TEACHER label: Stockfish MultiPV analyse -> the training row (SF's TRUE value/policy).
            infos = sf.analyse(board, limit, multipv=MULTIPV_K)
            row, cands = build_label(board, infos)
            if row is None:
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

            # (b) DAgger trajectory move: the AGENT picks (its own deep search over the current NNUE),
            # NOT the teacher's move and NOT a random/softmax sample. This walks the game into the
            # agent's OWN (hole) distribution; SF labels those states above with their true value.
            move = eng.search(board)[0]
            if move is None:                      # terminal / no legal move -> reseed
                board = new_game(rng)
                ply = 0
                continue
            board.push(move)
            ply += 1

    sf.quit()
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
