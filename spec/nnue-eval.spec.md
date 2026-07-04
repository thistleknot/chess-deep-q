---
description: 'The NNUE-style fast, hole-free leaf evaluator: king-relative sparse features + an incremental accumulator, CPU-trained on Stockfish labels, wired into the alpha-beta search as eval_fn — the measured path from ~1428 toward 2000+'
import:
  - prior-evaluator
  - learned-model
  - teacher-distillation
  - elo-measurement
---

***definitions***

- :NNUE-eval: is the fast leaf evaluator: board → White-absolute centipawns via a small net over sparse king-relative features. It exists to break the one contradiction the whole measurement thread exposed — accuracy ⊥ nodes/sec: it is accurate like a net yet µs-cheap like pst (a handful of small matmuls over a ~256-wide :Accumulator:), so it can be searched DEEP, the property the 9.6 ms/call conv net lacked. It replaces/augments :Prior-lineage: pst as the :Eval-wiring: leaf. Its value frame is the :Value-target-convention: (White-absolute centipawns), identical to `engine.pst_eval`, so it drops into `AlphaBetaEngine` unchanged.
- :Feature-set: is the sparse binary input — KING-RELATIVE piece placement (the NNUE property that makes the eval hard to fool: piece value is conditioned on king safety, so trapped-king / exposed-king positions are not "holes"). v1 = a king-BUCKETED HalfKP-lite: each non-king piece fires one feature per side-perspective, indexed by (own-king bucket, piece square, piece type+colour); the 64 king squares collapse to NNUE_KING_BUCKETS to keep the table size and the training-data demand tractable on ~66k labels. (Full HalfKP — 64 king-sq × 64 piece-sq × 10 piece — is the scale-up once data allows.) Activation is sparse (≤ 30 pieces) → the :Accumulator: is a sum of ≤ 30 weight columns.
- :Accumulator: is the incrementally-maintained first-layer sum, A = Σ W[:, f] over active features f, held per king-perspective (own / opponent). CONTRACT: after a move, A is UPDATED — subtract the moved/captured pieces' old features, add the new ones — O(changed) not O(pieces); a KING move triggers a full refresh of that side's accumulator (its bucket changed). Correctness INVARIANT: the incrementally-maintained A equals the full recompute at every position.
- :NNUE-net: is the small head over the (own‖opp) accumulator: clipped-ReLU → 1–2 narrow hidden layers → a scalar White-absolute centipawn output. Tiny (the bulk of the parameters is the sparse first-layer table; the head is thousands of weights), so a full forward is µs on CPU. Optional int16-accumulator / int8-weight quantization is the further inference-speed lever — DEFERRED; float v1 first.
- :NNUE-training: is CPU training on the EXISTING Stockfish-distilled labels (`data/distill_cp.jsonl` / `distill_sf.jsonl`: fen → SF centipawns), MSE to White-absolute cp. CPU because NNUE is small + sparse (CPU-friendly) and it SIDESTEPS the GPU throughput wall that capped the conv tower — the whole reason A beats the self-play/distillation-on-GPU path on this hardware. No new labelling: the data is already on disk.
- :Eval-wiring: is the integration into `engine.py`: :NNUE-eval: is an `eval_fn(board) → White-absolute cp` plugged as `AlphaBetaEngine(eval_fn=nnue_eval)`. v1 RECOMPUTES features per leaf (still µs — no engine change needed, the immediate deliverable). The :Accumulator: threaded through the search's make/unmake is the OPTIMIZATION (an `engine.py` change), gated behind the :Accumulator: invariant test — added only after v1 clears the :NNUE-kill-check:.
- :NNUE-kill-check: is the SHIP GATE, written up front: :NNUE-eval: MUST beat `pst_eval` at EQUAL search depth AND at equal wall-time, on the :Ladder: and vs SF@1320, under the same n ≥ 30 / CI :Measurement-power: discipline that settled the track decision. If it cannot beat its own predecessor under identical search and time, it DOES NOT SHIP (it guards the FINDINGS pattern where all six prior learned evals landed below pst — richer-but-exploitable is worse than clean-and-fast).

***implementation reqs***

- Constants: NNUE_KING_BUCKETS, NNUE_ACC_DIM (per-perspective accumulator width, e.g. 256), NNUE_HIDDEN, and the :Feature-set: table shape — developer-tuned capacity, sized against the ~66k-label data budget, not a fixed architecture.
- Files: NEW `nnue_model.py` (feature extraction + :Accumulator: + :NNUE-net:), `train_nnue.py` (CPU training on the distill_*.jsonl labels → `models/nnue.*`). Measurement reuses `measure_ladder.py` / `measure_timed.py` for the :NNUE-kill-check:; no new measure harness.
- :NNUE-eval: MUST emit White-absolute centipawns (same frame as `pst_eval` and the :Value-target-convention:), so `AlphaBetaEngine.eval_fn` is unchanged.
- Training runs on CPU and MUST NOT require CUDA.

***test reqs***

- The :Accumulator: invariant: over a random playout, the incrementally-maintained A after each make (and after unmake) equals a full recompute — asserted position by position, including a king move (full-refresh path).
- The :NNUE-kill-check: comparison: :NNUE-eval: vs `pst_eval` at equal fixed depth and at equal 0.3 s/move on the :Ladder:, n ≥ 30 with CIs.

***functional specs***

- :NNUE-eval: must plug in as `eval_fn` without engine changes (v1).
  - Given a board, When :NNUE-eval: is called, Then it returns a White-absolute centipawn score and `AlphaBetaEngine(eval_fn=nnue_eval)` runs unmodified.
- The :Accumulator: incremental update must equal the full recompute.
  - Given a move that is not a king move, Then A is updated by the changed pieces' features only; Given a king move, Then that side's accumulator is fully refreshed; in both cases the result equals the full recompute (the invariant).
- :NNUE-training: must be CPU-only and reuse existing labels.
  - Given the Stockfish-distilled label files, When training runs, Then it minimises MSE to White-absolute cp on CPU, producing `models/nnue.*`, with no CUDA dependency and no new labelling.
- The :NNUE-kill-check: must gate shipping, not just report.
  - Given the trained :NNUE-eval: does NOT beat `pst_eval` at equal depth AND equal time (CIs overlapping or below), Then it MUST NOT replace pst as the engine default; the pst engine remains the shipped strength.
  - Given it beats pst decisively under both disciplines, Then it becomes the alpha-beta `eval_fn` and the :Accumulator: search-integration optimisation is unlocked.
- The v1→optimisation order is fixed: correctness (recompute-per-leaf + kill-check) before speed (incremental accumulator threaded through search).
