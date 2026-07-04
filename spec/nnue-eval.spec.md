---
description: 'The NNUE-style fast, hole-free leaf evaluator: king-relative sparse features + an incremental accumulator, CPU-trained on Stockfish labels, wired into the alpha-beta search as eval_fn — the measured path from ~1428 toward 2000+'
import:
  - prior-evaluator
  - learned-model
  - teacher-distillation
  - elo-measurement
---

***definitions***

- :NNUE-eval: is the fast leaf evaluator: board → White-absolute centipawns via a small net over sparse king-relative features. It exists to break the one contradiction the whole measurement thread exposed — accuracy ⊥ nodes/sec: it is accurate like a net yet µs-cheap like pst (a handful of small matmuls over a ~256-wide :Accumulator:), so it can be searched DEEP, the property the 9.6 ms/call conv net lacked. It replaces/augments :Prior-lineage: pst as the :Eval-wiring: leaf. Its value frame is the :Value-target-convention: (White-absolute centipawns), identical to `engine.pst_eval`, so it drops into `AlphaBetaEngine` unchanged. PURPOSE (the repo's intent is chess-through-RL, so this is stated up front): :NNUE-eval: is NOT a supervised destination — it is the µs-cheap eval that makes DEEP self-play (d3+) affordable, i.e. the throughput fix that lets expert iteration AMPLIFY. It is Stage 1 of the RL :Prior-lineage: (supervised bootstrap → self-play outcomes take over the loss via :NNUE-training:), the enabling step for B, not a replacement of it.
- :Feature-set: is the sparse binary input — KING-RELATIVE piece placement (the NNUE property that makes the eval hard to fool: piece value is conditioned on king safety, so trapped-king / exposed-king positions are not "holes"). v1 = a king-BUCKETED HalfKP-lite: each non-king piece fires one feature per side-perspective, indexed by (own-king bucket, piece square, piece type+colour); the 64 king squares collapse to NNUE_KING_BUCKETS to keep the table size and the training-data demand tractable on ~66k labels. (Full HalfKP — 64 king-sq × 64 piece-sq × 10 piece — is the scale-up once data allows.) Activation is sparse (≤ 30 pieces) → the :Accumulator: is a sum of ≤ 30 weight columns.
- :Accumulator: is the incrementally-maintained first-layer sum, A = Σ W[:, f] over active features f, held per king-perspective (own / opponent). CONTRACT: after a move, A is UPDATED — subtract the moved/captured pieces' old features, add the new ones — O(changed) not O(pieces); a KING move triggers a full refresh of that side's accumulator (its bucket changed). Correctness INVARIANT: the incrementally-maintained A equals the full recompute at every position.
- :NNUE-net: is the small head over the (own‖opp) accumulator: clipped-ReLU → 1–2 narrow hidden layers → a scalar White-absolute centipawn output. Tiny (the bulk of the parameters is the sparse first-layer table; the head is thousands of weights), so a full forward is µs on CPU. Optional int16-accumulator / int8-weight quantization is the further inference-speed lever — DEFERRED; float v1 first.
- :NNUE-training: is CPU training with TWO target sources, blended by the :Distillation-anchor: α — this is the supervised→RL transition, spec'd in from day one:
  - (a) BOOTSTRAP — MSE(V, tanh(SF_cp/400)) on the EXISTING Stockfish-distilled labels (`data/distill_cp.jsonl` / `distill_sf.jsonl`, fen → SF cp). This is the supervised Stage-1 that MINTS the fast eval from data already on disk (α = 1; NOT RL — a teacher's per-position opinion, no committed outcome).
  - (b) RL — MSE(V, :Lambda-return:) from SELF-PLAY outcomes propagated back from the search leaf: **TD-Leaf(λ)** (the KnightCap/Giraffe lineage). SAME net, SAME search — only the ORACLE swaps (per-position teacher label → committed-outcome credit), which is exactly what makes (b) reinforcement learning where (a) is not.
  α anneals from 1 (pure bootstrap) toward the :Distillation-anchor: floor as strength is proven, so self-play outcomes take over the loss — the only channel that can EXCEED the teacher (:Surpass-teacher-gate:). Training may run on GPU OR CPU — NNUE is small and DENSE (a sparse embedding lookup + two tiny matmuls, NO convolutions), so it is NOT subject to the conv tower's launch-bound / thermal GPU wall; batched training is GPU-fast. The deployed eval's device is likewise an IMPLEMENTATION choice, NOT a spec mandate — the only hard requirement is that it clears the equal-time :NNUE-kill-check: (fast enough to win at equal wall-time, whatever the device). Engineering note only: sequential per-node alpha-beta pays kernel-launch latency, so a launch-free eval (CPU/quantized SIMD) is the usual fast path and batched GPU suits net-minimax — but the gate measures the OUTCOME, not the device. The bootstrap is what makes the RL phase AFFORDABLE (the fast eval buys d3+ self-play, the amplification regime).
- :Eval-wiring: is the integration into `engine.py`: :NNUE-eval: is an `eval_fn(board) → White-absolute cp` plugged as `AlphaBetaEngine(eval_fn=nnue_eval)`. v1 RECOMPUTES features per leaf (still µs — no engine change needed, the immediate deliverable). The :Accumulator: threaded through the search's make/unmake is the OPTIMIZATION (an `engine.py` change), gated behind the :Accumulator: invariant test — added only after v1 clears the :NNUE-kill-check:.
- :NNUE-kill-check: is the TWO-STAGE ship gate, written up front, because the repo's intent is chess-through-RL — strength alone is necessary but not sufficient:
  - **Stage 1 (the eval is good AND fast):** :NNUE-eval: MUST beat `pst_eval` at EQUAL search depth AND equal wall-time, on the :Ladder: and vs SF@1320, under the same n ≥ 30 / CI :Measurement-power: discipline that settled the track decision. If it cannot beat its own predecessor under identical search and time, it DOES NOT replace pst (guards the FINDINGS pattern — richer-but-exploitable is worse than clean-and-fast). Necessary, not sufficient.
  - **Stage 2 (it serves the RL goal):** with the µs eval making d3 self-play affordable, the expert-iteration rerun (:Self-play-bootstrap: at depth, TD-Leaf target) MUST produce a RISING :Ladder: curve — the amplification B never got a fair test of at d1. FALSIFICATION: if the curve is STILL flat at d3 with a µs eval, the amplification diagnosis was WRONG and self-play is reconsidered from scratch (the d1 result condemned d1, not self-play; this is the fair retest). Only when BOTH stages pass is the C synthesis realised: distilled → fast eval → deep self-play → outcomes drive the loss.

***implementation reqs***

- Constants: NNUE_KING_BUCKETS, NNUE_ACC_DIM (per-perspective accumulator width, e.g. 256), NNUE_HIDDEN, and the :Feature-set: table shape — developer-tuned capacity, sized against the ~66k-label data budget, not a fixed architecture.
- Files: NEW `nnue_model.py` (feature extraction + :Accumulator: + :NNUE-net:), `train_nnue.py` (CPU training on the distill_*.jsonl labels → `models/nnue.*`). Measurement reuses `measure_ladder.py` / `measure_timed.py` for the :NNUE-kill-check:; no new measure harness.
- :NNUE-eval: MUST emit White-absolute centipawns (same frame as `pst_eval` and the :Value-target-convention:), so `AlphaBetaEngine.eval_fn` is unchanged.
- Both training and inference are DEVICE-AGNOSTIC. Training: GPU or CPU (NNUE is small/dense, not conv-launch-bound). Inference: whatever clears the equal-time :NNUE-kill-check: — the gate measures time, not device (CPU/quantized is the usual fast path for per-node alpha-beta, but not mandated).

***test reqs***

- The :Accumulator: invariant: over a random playout, the incrementally-maintained A after each make (and after unmake) equals a full recompute — asserted position by position, including a king move (full-refresh path).
- The :NNUE-kill-check: comparison: :NNUE-eval: vs `pst_eval` at equal fixed depth and at equal 0.3 s/move on the :Ladder:, n ≥ 30 with CIs.

***functional specs***

- :NNUE-eval: must plug in as `eval_fn` without engine changes (v1).
  - Given a board, When :NNUE-eval: is called, Then it returns a White-absolute centipawn score and `AlphaBetaEngine(eval_fn=nnue_eval)` runs unmodified.
- The :Accumulator: incremental update must equal the full recompute.
  - Given a move that is not a king move, Then A is updated by the changed pieces' features only; Given a king move, Then that side's accumulator is fully refreshed; in both cases the result equals the full recompute (the invariant).
- :NNUE-training: must reuse existing labels and support BOTH target sources; it may run on GPU or CPU (the µs-CPU requirement is on inference, not training).
  - Given the Stockfish-distilled label files, When training runs, Then it minimises MSE to White-absolute cp (on whichever device is faster), producing `models/nnue.*`, with no new labelling.
  - Given the deployed eval, Then it clears the equal-time :NNUE-kill-check: (fast enough to win at equal wall-time); the device is unconstrained.
  - Given :Distillation-anchor: α = 1 (bootstrap), Then the target is the SF label; Given α at its floor (RL phase), Then the loss is dominated by MSE(V, self-play :Lambda-return:) (TD-Leaf), the SF term tethering but not driving — the SAME net/architecture, only the oracle swapped.
- The :NNUE-kill-check: is two-stage and gates on the RL goal, not just strength.
  - STAGE 1 — Given the trained :NNUE-eval: does NOT beat `pst_eval` at equal depth AND equal time (CIs overlapping or below), Then it MUST NOT replace pst as the engine default; the pst engine remains the shipped strength.
  - STAGE 2 — Given Stage 1 passes (eval good+fast) BUT the d3 self-play rerun (TD-Leaf target) stays FLAT on the :Ladder:, Then the amplification diagnosis is REJECTED: strength is achieved but the RL goal is not, and self-play is reconsidered from scratch (the fair retest the d1 result never was).
  - Given BOTH stages pass, Then the NNUE eval is the alpha-beta `eval_fn`, the :Accumulator: search-integration optimisation is unlocked, AND the RL loop has a working amplification regime (the C synthesis realised).
- The v1→optimisation order is fixed: correctness (recompute-per-leaf + kill-check) before speed (incremental accumulator threaded through search).
