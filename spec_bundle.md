# chess-deep-q — full spec bundle

Spec-driven development: this is the authoritative source of truth; code traces to it. Single entry point is `chess-rl.spec.md` (its imports transitively close the whole set). Files below are in dependency reading order.

## For reviewers — live decisions & open questions

**Where the implementation actually is (measured, honest):**
- Learned net currently ~920 Elo vs a real Stockfish anchor (SF 18; UCI_Elo floor 1320). It is
  *undertrained*, not broken — see blocker 1.
- The repo's pure-Python alpha-beta engine (hand eval): n=30 with CIs, pst measures ~1428 at a real
  0.3s/move [CI 1306-1584] and ~1672 at fixed-depth-3 (unbounded time). The old "4-0 -> ~1720" was
  n=4 noise. It is the baseline + a candidate teacher, not the learned model.
- Only Stage 1 (Stockfish distillation) has run. Stage 2/3 (λ-return refinement, self-play) are
  authored in spec but gated behind an unmet 1200-Elo gate and have not executed.

**Goal (ambition):** frontier strength, **2600+**, honestly measured; plus an **Elo↔temperature
proxy** so ONE strong model can be dialed to any point on the human rating curve (see
dynamic-difficulty + elo-measurement).

**Open questions where SME input is most valuable:**
1. **Training throughput.** The residual tower runs ~1 step/s at batch 512 and ~12 s/step at batch
   2048 on a Max-Q laptop GPU (superlinear — thermal/throttle). A 200 s train phase buys tens of
   gradient steps, not thousands. Is a residual tower the wrong net for this hardware? CPU training
   (batched CPU hit ~18k samples/s for the tiny net), a smaller net, or gradient accumulation?
2. **Search ceiling. ANSWERED (n=30, this iteration): search depth reached within the time budget
   dominates.** At a fixed 0.3s/move the evals rank strictly by per-node SPEED, not accuracy:
   pst 1428 >> linear 1200 > gbdt 735 >> hybrid(pst+residual) -280 (0/30) — anything that slows the
   per-node eval collapses search depth and strength. The learned leaf value is a dead end here; the
   fast smooth hand eval searched deeper wins. Using the net as a move-ordering PRIOR orders ~8%
   fewer nodes at ply<=1 but costs ~9.6ms/call (112% of a 0.3s budget) -> not repaid at time control
   until a cheaper policy exists. So: *search*, not the net, is the binding constraint on this
   hardware; the net earns its place only as a cheap ordering/window signal, not a leaf eval.
3. **Teacher & data.** To exceed 1720 toward 2600+, Stockfish must be the teacher (the 1720 engine
   caps too low). Distill SF eval + best move (dense, low-variance signal) vs learn from SF
   self-play games (in-distribution, outcome-grounded) vs both? Teacher depth vs data volume under
   the throughput constraint?
4. **RL-rule soundness.** Value = search-bootstrapped λ-return (tree-backup, off-policy-safe);
   policy = MCTS visit-count distillation (expert iteration). Is the classification and the
   off-policy-safety argument sound? See `rl-categorization.spec.md` and `value-target.spec.md`.

## Contents

1. `README.md`
2. `chess-rl.spec.md`
3. `elo-measurement.spec.md`
4. `annealing-schedule.spec.md`
5. `prior-evaluator.spec.md`
6. `learned-model.spec.md`
7. `teacher-distillation.spec.md`
8. `search-mcts.spec.md`
9. `value-target.spec.md`
10. `self-play-leela.spec.md`
11. `training-loop.spec.md`
12. `dynamic-difficulty.spec.md`
13. `elo-calibration.spec.md`
14. `rl-categorization.spec.md`
15. `terminal-interface.spec.md`



================================================================================
FILE: spec\README.md
================================================================================

# chess-deep-q spec

Source-of-truth specs for the RL alignment described in `../prompt.md` ("apply RL to a
system such as chess when you already have a known evaluator/prior"). Authored with the
`spec` skill (rendered layer). **Spec-driven development: the spec is authoritative; code traces
to it, not the reverse.**

**Single entry point: [`chess-rl.spec.md`](chess-rl.spec.md)** — the root rendered spec. Its
`import:` list (training-loop, rl-categorization, dynamic-difficulty, elo-calibration,
terminal-interface) transitively closes over all 13 specs below, so following imports from that
one file reaches the entire set. This index gives the human reading order; `chess-rl.spec.md` is
the machine-followable root.

Import order flows bottom-up:

1. **elo-measurement** — the real Stockfish anchor, measured Elo, and the gates that own all
   progress. No imports; everything gates on it.
2. **annealing-schedule** — the shared coefficient service (prior → learned handoff), driven by
   Elo-gated progress. Imports 1.
3. **prior-evaluator** — the fixed heuristic prior and the :Prior-lineage: it starts
   (heuristic → distilled teacher → learned net). Imports 1–2.
4. **learned-model** — the dual-head residual tower (value + policy), value-target convention,
   batch-evaluate API, reward frame and sign. Imports 2–3.
5. **teacher-distillation** — Stage 1: Stockfish distillation under the 5-minute cumulative
   run contract (process-separated labelling, dedup'd dataset, Elo trend). Imports 1–4.
6. **search-mcts** — PUCT search with batched leaf evaluation, the exposed search value, and two
   regression-pinned conventions (negamax backup sign; argmax-Q root selection at small budgets).
   Imports 2–5.
7. **value-target** — the value-head learning rule: a search-bootstrapped λ-return (tree-backup,
   off-policy-safe) shared by the refinement and self-play stages; TD(0) and Monte-Carlo are its
   endpoints. Imports 2, 4, 6.
8. **self-play-leela** — Stage 3: expert iteration (visit-count policy distillation, λ-return
   value target, Dirichlet carve-out, surpass-teacher gate, optional σ-matched early opponent).
   Imports 1–2, 4–7.
9. **training-loop** — the staged loop: gate-driven stage controller, reward assignment,
   failure-mode monitoring. Imports 1–8.
10. **dynamic-difficulty** — adapt the opponent's move-selection temperature to the human
    player's skill band (regret-tracked, *relative*). Imports learned-model + search-mcts.
11. **elo-calibration** — the temperature→*absolute*-Elo dial: calibrate one net to any target
    strength on the human curve. Imports elo-measurement + dynamic-difficulty.
12. **rl-categorization** — qualified three-stage classification (supervised distillation →
    off-policy λ-return refinement → expert iteration; never SARSA/PPO/literal-DQN).
13. **terminal-interface** — the terminal human-vs-computer front-end: move entry, in-game
    commands (full word + first-letter shortcut), and the per-turn board readout that surfaces
    the estimated Elo. Imports dynamic-difficulty + elo-calibration.

The load-bearing idea across the set: a **prior lineage** — the hand heuristic bootstraps the
distilled Stockfish teacher, the teacher bootstraps the self-play learner, and the learner
eventually surpasses the teacher — where every handoff is **annealed toward the learned model**
(never toward randomness; root Dirichlet noise in self-play games is the one bounded, constant
carve-out) and **gated by measured Elo against a real Stockfish anchor**, never by wall-clock or
game count. Failure modes (teacher lock-in, reward hacking, policy collapse, stagnation at a
gate) are monitored, not assumed away. Two search bugs found by measurement are pinned as
regression specs in search-mcts: the negamax backup sign convention and argmax-Q root selection
at small simulation budgets.

The value head learns a **search-bootstrapped λ-return** (value-target), not literal TD(0) and not
pure Monte-Carlo — the bootstrap share β anneals down from lean-on-the-distilled-value (low
variance early) toward the ground-truth game outcome (AlphaZero MC) as strength is proven. The
policy learns by MCTS visit distillation (expert iteration), never policy gradient. An optional
σ-matched "just-above" opponent (self-play-leela) can sharpen early training but is annealed out in
favor of full-strength symmetric self-play, since a weakened opponent yields lower-quality targets.


================================================================================
FILE: spec\chess-rl.spec.md
================================================================================

---
description: 'Single entry point for the chess-RL spec set — the staged, Elo-gated learning system; imports the DAG roots to close the whole namespace'
import:
  - training-loop
  - rl-categorization
  - dynamic-difficulty
  - elo-calibration
  - terminal-interface
---

***definitions***

- :Chess-RL-system: is the whole agent: a dual-head learned network trained in three Elo-gated stages — supervised Stockfish distillation, annealed off-policy λ-return refinement, and Leela-style expert-iteration self-play — planning with MCTS/PUCT over the known game rules, and dialable to any target strength by the :Difficulty-controller:. Its single load-bearing invariant: **every prior→learned handoff is annealed (never toward randomness) and gated by measured Elo against a real Stockfish anchor, never by wall-clock or game count.**
- :Rung: is one strength milestone on the ladder the system climbs — beat baseline, then 1200, 1600, and surpass the teacher — each a :Elo-gate: that must be cleared by :Measured-elo: before the next stage's coefficients advance.

***implementation reqs***

- This file imports the three DAG roots (:Stage-controller: via training-loop, the classification via rl-categorization, the strength dial via dynamic-difficulty), which transitively pull in every other spec (elo-measurement, annealing-schedule, prior-evaluator, learned-model, teacher-distillation, search-mcts, value-target, self-play-leela). Read `README.md` for the numbered reading order.
- No new concepts are defined for the mechanics here; each lives in its owning spec. This root states only the end-to-end contract that spans them.

***functional specs***

- The :Chess-RL-system: must advance strictly by measured strength, never by schedule.
  - Given a :Rung:'s :Elo-gate: has not been cleared by :Measured-elo:, Then the next stage MUST NOT start and every annealed coefficient holds (see annealing-schedule, training-loop).
- The value head must learn the search-bootstrapped λ-return, and the policy head must learn by MCTS visit distillation — never TD(0)-only, never policy gradient (see value-target, self-play-leela, rl-categorization).
- The two search conventions must never regress: side-to-move-relative negamax backup, and argmax-Q root selection at small simulation budgets (see search-mcts).
- One trained :Chess-RL-system: must be dialable to any absolute strength across the human rating range by the :Absolute-strength-dial: (the :Temperature-elo-curve:), with the :Difficulty-controller: tracking the seated human relative to that operating point (see elo-calibration, dynamic-difficulty, elo-measurement).


================================================================================
FILE: spec\elo-measurement.spec.md
================================================================================

---
description: 'Measured strength against a real Stockfish anchor — the honest progress signal and gate authority'
---

***definitions***

- :Elo-anchor: is a real Stockfish binary (in `engines/`) pinned to a fixed strength via UCI_LimitStrength/UCI_Elo; it is the only externally calibrated reference, so every strength claim is stated relative to it. Stockfish enforces a hard floor of UCI_Elo = 1320 — strength below the floor is inferred by the logistic relation, never by configuring a weaker anchor.
- :Measured-elo: is the Elo estimate produced by an alternating-color match against the :Elo-anchor:, with ply-capped games adjudicated by Stockfish evaluation, and the match score converted through the logistic relation (`measure_sf.py`).
- :Elo-gate: is a strength threshold (1200, then 1600, then teacher-strength) that must be cleared by :Measured-elo: before annealing progress or a training-stage transition may advance past it.
- :Measurement-game: is a game played purely for measurement: every training-time noise source (root Dirichlet noise, move temperature, shaped reward) is off and the agent plays argmax.
- :Measurement-power: is the requirement that GAMES_PER_MEASUREMENT be large enough that the 95% confidence interval on :Measured-elo: is NARROWER than the :Elo-gate: step it must resolve. The per-game score SD is √(p(1−p)/n); at n=6 it is ~0.13 — wider than the gap between adjacent strength levels — so gate decisions at n=6 are coin flips and any single-lever conclusion drawn from them is unfounded. Adequate power is ~30–50 games at a fast time control (games are cheap relative to a labeling grind).
- :Ladder: is the sub-floor strength placement. Because SF's UCI_Elo floor is 1320, a net weaker than that shutouts to the anchor sentinel and cannot be located by the anchor alone (the whole "always 920" artifact). The :Ladder: adds ORDINAL rungs BELOW the floor — a uniform-random mover and a 1-ply material/PST heuristic — reported as an Elo DIFFERENCE (score → 400·log10(s/(1−s))), never a calibrated Elo (no external anchor exists below 1320). Beating random and the 1-ply heuristic DECISIVELY is the operational "do we have a model" test — a real player above the trivial baselines — while a shutout (e.g. 20-0) is a BOUND (≥ the ordinal ceiling), not a point. Implemented in `measure_ladder.py` (net_search vs random / heuristic-1ply / SF@1320). First result on the Stage-1 tower: 20-0 vs both sub-floor rungs, 0-20 vs SF@1320 — a genuine sub-floor player.
- :Compute-frontier: is the score-vs-compute selection rule for a sampled search config. The beam's deterministic leaf-eval count (total_calls) is the compute axis — linear in wall-time when ops are homogeneous; the frontier plots :Measured-elo: D against it. The zero-search baseline (rfr — argmax :Policy-head:, or random before a policy exists) anchors it, and a config's ADVANTAGE is its PAIRED excess over the rfr on a fixed color-balanced opening suite (common random numbers → the shared opening/colour/anchor noise cancels, so Var(advantage) ≪ Var(D)). Sharpe = advantage / sec-per-move (measured wall-time is the real denominator; advantage / total_calls is the hardware-independent proxy, faithful only for homogeneous ops). The :Run-contract: picks the Pareto-frontier config with the best D whose total_calls fits the budget; the tangency (max Sharpe from the rfr) is the best score-per-compute point.

***implementation reqs***

- `measure_sf.py` is the sole measurement authority; self-reported training signals (loss curves, validation MSE, sign accuracy) never gate anything.
- Constant: ANCHOR_ELO, ELO_GATES, GAMES_PER_MEASUREMENT, and the adjudication centipawn threshold — developer-tuned measurement rules.
- The Stockfish binary lives under `engines/` and is discovered by glob, not PATH.

***test reqs***

- The harness must reproduce a known-strong engine's superiority over the anchor: `engine.py`'s hand-eval alpha-beta must score decisively above 0.5 vs SF@1320. (Honest numbers, n=30 with CIs: the older "4-0 → ~1720 at 0.3s/move" was n=4 noise; at a real 0.3s budget pst measures ~1428 [CI 1306–1584], while fixed-depth-3 unbounded-time measures ~1672. Small-n superlatives are forbidden by :Measurement-power:.)

***functional specs***

- :Measured-elo: must come from alternating colors so first-move advantage cancels.
  - Given N measurement games, Then the agent plays White in ⌈N/2⌉ and Black in ⌊N/2⌋.
- Ply-capped games must be adjudicated, not discarded.
  - Given the ply cap is reached, When the game is scored, Then Stockfish evaluation of the final position decides win/draw/loss by the centipawn threshold.
- The score-to-Elo conversion must be the logistic relation, and shutouts must be reported honestly.
  - Given match score s in (0,1), Then Elo = anchor + 400·log10(s/(1−s)).
  - Given s in {0,1}, Then the result is reported as a bound ("below/above anchor±cap"), never a point estimate.
- :Elo-gate:s must be the only authority for progress, and gate decisions must meet :Measurement-power:.
  - Given :Measured-elo: below the next :Elo-gate:, Then gated annealing progress MUST NOT advance past its segment boundary and the next training stage MUST NOT start.
  - Given GAMES_PER_MEASUREMENT so small that the 95% CI straddles the gate, Then the result is INCONCLUSIVE (neither pass nor fail); GAMES_PER_MEASUREMENT MUST be sized so the CI is narrower than the gate step before any lever is judged by it.
- :Measurement-game:s must be noise-free.
  - Given any measurement game, Then Dirichlet noise, move temperature, and shaped reward are all disabled and the agent plays argmax.
- The anchor floor must be respected.
  - Given a requested anchor below 1320, Then the harness clamps to 1320 and says so; strength below the floor is inferred through the logistic relation from the score against 1320.
- Sub-floor strength must be placed on the :Ladder:, not left at the shutout sentinel.
  - Given a net that shutouts vs SF@1320, Then it is measured vs the random and 1-ply-heuristic rungs and reported as an ordinal Elo difference; a 20-0 rung result is a bound, not a point.
  - Given the net beats both sub-floor rungs decisively, Then the "we have a model" criterion is satisfied — a real player above the trivial baselines — even while sub-floor vs the anchor.


================================================================================
FILE: spec\annealing-schedule.spec.md
================================================================================

---
description: 'Progress-driven coefficients that hand control from priors (heuristic, then distilled teacher) to the learned model'
import:
  - elo-measurement
---

***definitions***

- :Training-progress: is a scalar in [0,1] expressing how far training has advanced within the current gate segment, 0 at the segment start and approaching the segment boundary as training volume accumulates.
- :Gated-progress: is the schedule input that replaces raw game-count progress: it advances with training volume inside a gate segment but **clamps at the segment boundary until :Measured-elo: clears the segment's :Elo-gate:**. The schedule itself stays pure and stateless; evaluating the gate and clamping is the caller's job (the training loop), which passes the resulting scalar in.
- :Annealing-schedule: is the single stateless source of truth that maps :Gated-progress: to every coefficient governing prior-versus-learned control; it is a `service` — it holds no mutable state and coordinates no entities, it only computes.
- :Demo-share: is the fraction of each training batch drawn from teacher-labelled data (the cumulative distillation dataset) versus self-play replay; 1.0 during pure distillation, decaying with progress, floored above zero until the surpass-teacher gate clears.
- :Teacher-policy-weight: is the blend share of the frozen teacher-distilled policy snapshot versus the current learned policy when forming the PUCT prior P; it decays to zero.
- :Teacher-leaf-weight: is the share of a search-leaf value taken from the previous prior-lineage member (hand heuristic, then teacher snapshot; the lineage itself is defined in prior-evaluator) rather than the current learned model; it generalizes and subsumes the earlier learned-leaf-weight (learned share = 1 − teacher share) and decays to zero.
- :Shaped-reward-weight: is the multiplier on prior-derived intermediate reward, distinct from the fixed terminal outcome.
- :Prior-bias-temperature: is the softening applied to the policy prior before PUCT selection (higher meaning flatter, less prior-dominated search bias).
- :Bootstrap-share: (β) is the share of the value-learning target taken from the bootstrapped search value versus the realized game outcome; it sets the λ of the value-target's λ-return (defined in value-target, which reads this knob) as λ = 1 − β. It decreases monotonically toward BOOTSTRAP_SHARE_FLOOR as the learner strengthens (β high early leans on the trustworthy distilled value for low variance; β→floor lets λ→~1, approaching AlphaZero Monte-Carlo). Decreasing β moves toward the ground-truth outcome, not toward randomness.

***implementation reqs***

- Constant: LEAF_WEIGHT_START / LEAF_WEIGHT_END — bounds of the learned leaf share (see :Teacher-leaf-weight:), prior-heavy at start, learned-heavy at end.
- Constant: DEMO_SHARE_START / DEMO_SHARE_END / DEMO_SHARE_FLOOR — decay bounds of :Demo-share:; the floor applies until the surpass-teacher :Elo-gate: clears, after which it may be zero.
- Constant: TEACHER_POLICY_WEIGHT_START / TEACHER_POLICY_WEIGHT_END — bounds of :Teacher-policy-weight:.
- Constant: SHAPED_WEIGHT_START / SHAPED_WEIGHT_END / SHAPED_WEIGHT_FLOOR — decay bounds of :Shaped-reward-weight:, never below the floor.
- Constant: PRIOR_TEMP_START / PRIOR_TEMP_END — bounds of :Prior-bias-temperature:, flatter with progress.
- Constant: BOOTSTRAP_SHARE_START / BOOTSTRAP_SHARE_END / BOOTSTRAP_SHARE_FLOOR — decay bounds of :Bootstrap-share:; β starts high and decays toward the floor (λ toward 1).
- Constant: MCTS_EXPLORATION_START / MCTS_EXPLORATION_END (c_puct bounds) and the per-move simulation-budget / sample-width bounds — the search knobs the schedule also owns, moved out of inline call-site math.
- All endpoints live in `constants.py`; they are developer-tuned Constants (the program means something different when changed), not deployment Config.

***functional specs***

- :Annealing-schedule: must map :Gated-progress: to :Teacher-leaf-weight:, :Demo-share:, :Teacher-policy-weight:, :Shaped-reward-weight:, :Prior-bias-temperature:, the PUCT exploration weight, and per-move search budgets.
  - Given progress 0, When any coefficient is requested, Then it returns its prior-heavy / teacher-heavy / shaping-heavy endpoint.
  - Given progress 1, Then it returns its learned-heavy endpoint.
- Every handoff coefficient must be monotone toward the learned model.
  - Given gated progress a < b, Then demo_share(a) >= demo_share(b), teacher_policy_weight(a) >= teacher_policy_weight(b), teacher_leaf_weight(a) >= teacher_leaf_weight(b), shaped_reward_weight(a) >= shaped_reward_weight(b), and bootstrap_share(a) >= bootstrap_share(b).
- :Bootstrap-share: decreasing is toward the ground-truth game outcome (λ→1), consistent with "reduced bootstrap control, never increased randomness".
  - Given progress 1, Then bootstrap_share is >= BOOTSTRAP_SHARE_FLOOR (a light search bootstrap is always retained).
- :Demo-share: must not starve the teacher anchor prematurely.
  - Given the surpass-teacher :Elo-gate: has not cleared, Then :Demo-share: >= DEMO_SHARE_FLOOR > 0 (teacher data anchors against forgetting).
  - Given the surpass-teacher gate has cleared, Then the floor may be zero.
- :Shaped-reward-weight: must not fall below SHAPED_WEIGHT_FLOOR.
  - Given progress 1, Then shaped_reward_weight >= SHAPED_WEIGHT_FLOOR.
- Progress must hold while strength is unproven.
  - Given :Gated-progress: clamped at a gate boundary across several runs, Then every coefficient holds constant (no drift without measured improvement).
- Every knob must express reduced prior control, never increased randomness — annealing hands behavior to the learned model, it does not inject noise. (Load-bearing distinction from prompt.md: "the point is not become random.")
  - Root Dirichlet noise is explicitly NOT a schedule knob: it is constant AlphaZero-style structured exploration owned by the self-play stage, applied only in self-play training games and off in every :Measurement-game:.
- :Annealing-schedule: must be pure — the same :Gated-progress: must yield identical coefficients on every call, so search and training read one consistent schedule.


================================================================================
FILE: spec\prior-evaluator.spec.md
================================================================================

---
description: 'The fixed heuristic prior, and the prior lineage it starts: heuristic -> distilled teacher -> learned net'
import:
  - elo-measurement
  - annealing-schedule
---

***definitions***

- :Prior-evaluator: is the fixed, hand-authored scoring of a chess position from material, mobility, king safety, pawn structure, space and coordination; it is the bootstrap knowledge the learner starts from and must eventually surpass.
- :White-absolute-frame: is the sign convention where a positive score favors White regardless of side to move; it is the single frame shared by prior, learned value, and reward.
- :Move-category-weights: is the prior's search bias — a mapping from tactical move classes (captures, checks, threats, development) to sampling weights that decide which moves a search even considers.
- :Prior-lineage: is the ordered succession of priors — hand heuristic, then distilled teacher-net snapshot, then current learned net — where each member bootstraps the next, each handoff is annealed and gated by :Measured-elo:, and every earlier member remains a fallback for the one after it. The teacher snapshot *supersedes* the heuristic as the operative prior once it measures stronger against the :Elo-anchor:; the heuristic is never deleted.

***implementation reqs***

- The :Prior-evaluator: lives in `evaluation.py` and is never trained; no gradient ever updates it.
- Constant: PIECE_VALUES and the per-feature evaluation weights — domain rules of the prior, developer-tuned.

***test reqs***

- A position winning for White and its mirror winning for Black, to assert frame symmetry.

***functional specs***

- The :Prior-evaluator: must score every non-terminal position in the :White-absolute-frame:.
  - Given a position better for White, Then its score should be positive; Given the color-mirrored position, Then its score should be negative of the first.
- The :Prior-evaluator: must stay fixed across all of training.
  - Given any training progress, When the evaluator is queried, Then it returns the same score for the same position (no learned drift). (Maintain: prior is a constant reference, not a moving target.)
- The prior's influence must be annealable by callers, not by mutating the prior itself.
  - Search leaves and shaped reward should scale prior contribution via the :Annealing-schedule:; the prior remains whole and callers down-weight it.
- :Move-category-weights: must be softenable before sampling so the prior's search bias can decay with progress.
  - Given rising :Prior-bias-temperature:, When categories are sampled, Then the effective weights should flatten toward uniform, widening the move set the learner may discover. (This is the guard against prompt.md's "prior lock-in".)
- The :Prior-lineage: handoff must be measured, not assumed.
  - Given the distilled teacher snapshot measures stronger than the heuristic against the :Elo-anchor:, Then the teacher snapshot becomes the operative prior for leaf blending and shaped reward.
- Lineage fallback must step one member back, never to random.
  - Given net inference fails at any lineage level, When a value is needed, Then evaluation falls back one lineage step (learned → teacher → heuristic); the heuristic is the terminal fallback.


================================================================================
FILE: spec\learned-model.spec.md
================================================================================

---
description: 'The dual-head learned network (residual tower: value + policy) and the agent that trains it'
import:
  - annealing-schedule
  - prior-evaluator
---

***definitions***

- :Learned-value: is the network's scalar estimate of a position's worth in the :White-absolute-frame:, in [-1,1], updated by training and intended to eventually outrun every earlier :Prior-lineage: member.
- :Policy-head: is the network's distribution over a flat move index (from-square × to-square + promotion piece); it is trained by cross-entropy — on Stockfish MultiPV soft targets during distillation, and on MCTS root visit distributions during self-play — never by policy gradient.
- :Residual-tower: is the shared trunk feeding both heads: a small residual convolutional stack (on the order of 4–6 blocks × 64 filters) sized by the throughput budget below, not by a fixed architecture clause. (This supersedes the earlier "value-only is intentional" clause, which is deleted: the policy head is required for PUCT self-play.)
- :Input-planes: is the board encoding: 12 piece-placement planes plus a mandatory side-to-move plane (and castling/en-passant planes where implemented). The side-to-move plane became mandatory the moment a :Policy-head: exists — 12 placement planes cannot identify the mover, and a policy is meaningless without knowing whose move it is. The value output remains White-absolute regardless.
- :Encoding-packing: is the storage rule for the binary :Input-planes:: the encoded dataset and replay board tensors are 0/1 planes (several constant-fill), so they MUST be stored PACKED (uint8 or bit-packed) and unpacked to float only at batch assembly — fp32-on-disk is up to 32× wasteful (the ~217 MB encoded cache packs to single-digit MB). This is the memory reduction that COUNTS. WEIGHT quantization (NF4 / QLoRA-style) is redundant here: the tower is ~1 M params (~4 MB), so 4-bit saves nothing material, and the compute-precision win is already taken by AMP fp16 autocast. The binding constraints are batched-eval throughput (NET_MIN_BATCHED_EVALS_PER_SEC) and DATA storage, not weight memory — precision reduction is applied to DATA, never weights.
- :Value-target-convention: is the single scale shared by every stage: tanh(white_centipawns / 400), White-absolute, in [-1,1] — used identically by teacher labels, leaf blending, TD targets, and the terminal ±1 outcomes it must be commensurate with.
- :DQN-agent: is the `service` that owns the network(s), the target network, the replay buffer, and the training step; it coordinates learning but delegates move choice to search. (Name kept for compatibility; see rl-categorization.)
- :Replay-transition: is a stored (state, move, reward, next-state, done) tuple, the off-policy record of experience the learner samples from.
- :Terminal-reward: is the fixed game-outcome signal (+1 win / 0 draw / -1 loss) that the objective must ultimately serve.

***implementation reqs***

- Constant: GAMMA, LEARNING_RATE, BATCH_SIZE, REPLAY_CAPACITY, TARGET_UPDATE_INTERVAL — developer-tuned learning rules, centralized in `constants.py`.
- Constant: NET_MIN_BATCHED_EVALS_PER_SEC — the throughput budget any capacity change must respect. Empirical anchors (tiny 2-conv net, batch 256): ~18k evals/s CPU, ~122k evals/s GPU; single-position ~2–3 ms. Any tower change is conformant only if it still meets the constant.
- The network MUST expose a batch-evaluate API (one forward pass for a batch of positions); search code that loops single-position inference is non-conformant.
- Encoded board tensors and replay states MUST be stored packed (uint8/bit-packed per :Encoding-packing:) and unpacked to float only per batch; fp32-on-disk encodings are non-conformant. Weight quantization is out of scope (nets too small; AMP owns the compute-precision win).

***test reqs***

- A position one legal move from mate for each color, to assert terminal reward sign.
- A position and its color-mirror with side to move flipped, to assert the symmetry spec below.
- A throughput benchmark batch, to assert NET_MIN_BATCHED_EVALS_PER_SEC.

***functional specs***

- :Learned-value: must share the :White-absolute-frame: and :Value-target-convention: with every :Prior-lineage: member so all values are blendable at a search leaf.
  - Given a position winning for White, Then a trained :Learned-value: should be positive.
- Color symmetry must hold across both heads.
  - Given a position and its color-mirror with turn flipped, Then the value negates and the policy distribution maps through the mirror transform.
- Batched evaluation must be the only evaluation path used at scale.
  - Given a batch of B positions, Then evaluation is one forward pass (or ⌈B/max_batch⌉ passes), never B single-position calls.
  - Given the benchmark batch, Then measured throughput >= NET_MIN_BATCHED_EVALS_PER_SEC.
- :Terminal-reward: must be signed by the winner, not hardcoded.
  - Given White delivers checkmate, When the transition is stored, Then reward should be +1; Given Black delivers checkmate, Then reward should be -1. (This corrects a sign that previously taught winning positions as losses.)
- Intermediate reward must be **potential-based** (Ng 1999), so it cannot change the optimal policy: the shaping term is F = γ·Φ(s′) − Φ(s), where Φ(s) is the operative :Prior-lineage: member's normalized score, scaled by :Shaped-reward-weight:. Raw `weight × prior-score` shaping is non-conformant — it can alter the optimum the :Terminal-reward: defines.
  - Given the potential-based form, Then the optimal policy is provably invariant to the shaping and :Shaped-reward-weight: (the floor) is harmless, not merely small. (Guards prompt.md's "reward hacking" by construction, not by magnitude.)
  - Given late training, Then shaped magnitude is small relative to :Terminal-reward: regardless, so game result dominates the learning signal.
- The :DQN-agent: must learn off-policy from sampled :Replay-transition:s with a periodically-synced target network.
  - If the buffer holds fewer than BATCH_SIZE transitions, Then the training step is a no-op.
- Exploration must be structured (search-driven), so the agent must not expose a uniform-random epsilon path as its behavior policy.
  - The vestigial epsilon-greedy branch should be removed; move choice comes from search, not a coin flip.
  - The standard A2C entropy-bonus (c_e·H(π) in the policy loss) is DELIBERATELY ABSENT: the policy is trained by cross-entropy to MCTS visit counts (expert iteration), and exploration entropy is injected UPSTREAM as Dirichlet root noise + visit temperature (AlphaZero-style, see self-play-leela). So the missing H(π) term is a documented choice, not an omission; adding a policy-gradient entropy bonus would double-count the exploration the search targets already carry.


================================================================================
FILE: spec\teacher-distillation.spec.md
================================================================================

---
description: 'Stage 1 — supervised distillation of a Stockfish teacher into the dual-head net, under a 5-minute cumulative run contract'
import:
  - elo-measurement
  - annealing-schedule
  - prior-evaluator
  - learned-model
---

***definitions***

- :Teacher: is Stockfish analysing at a fixed shallow depth — depth-8 measures ≈2200+ strength at ≈180 labels/s per engine process. The repo's Python alpha-beta (`engine.py`, ~1720 measured) is explicitly NOT a labelling teacher (3–8 labels/s), though it remains the strength milestone a distilled-and-searched net should approach.
- :Distillation-label: is one supervised example: (FEN, value target per the :Value-target-convention:, policy target = softmax over the MultiPV candidates' centipawns at temperature POLICY_TARGET_TEMP with zero mass elsewhere).
- :Cumulative-dataset: is the append-only labelled set on disk (`data/distill_sf.jsonl`; ~29k depth-8 rows already exist), deduplicated by (FEN, depth), from which every run trains — each ≤5-minute run climbs from the accumulated total rather than starting over.
- :Process-separated-labeling: is the requirement that labelling engines run in separate OS processes (multiprocessing workers, or distinct label-then-train phases) — never as threads sharing the trainer's Python GIL.
- :Run-contract: is the unit of operation: one ≤5-minute run = append labels + train + measure :Measured-elo: + append the (dataset-size, Elo) point to a persistent trend log.
- :Position-source: is where the positions to be labelled come from: **strong-game trajectories** — self-play games by the :Teacher: begun from a few random opening plies (for diversity), from which positions are sampled — NOT uniform random-playout positions. Random-playout positions are largely off-distribution (positions no strong player reaches), so labelling them wastes capacity; in-distribution positions make the distilled value and policy targets relevant to real play, and give the learner a better off-policy dataset to bootstrap from.
- :Trajectory-sampling: is how a :Position-source: game advances at each move: a temperature-softmax sample over the :Teacher:'s MultiPV top-K candidates (scored by centipawns), rather than always its single best move. The MultiPV analyse is already computed for the policy target, so sampling adds NO search cost and does not deepen the teacher (respects LABELS_PER_SEC_FLOOR). Temperature is annealed high→low across the game (more diverse in the opening, near-best later) so trajectories stay near the strong-play manifold — hot sampling throughout would drift positions off-distribution and forfeit the in-distribution benefit.
- :Search-visited-positions: is a required slice of the :Position-source: drawn from the (often non-quiet) distribution the SEARCH will actually query at inference — sampled interior/leaf nodes of the teacher's search trees, or trajectory positions perturbed by 1–2 plies of sampled captures — not game-trajectory positions alone. Rationale (the q8 blowup): a learned eval's error is DISTRIBUTION-DEPENDENT — the net collapsed on off-distribution capture-resolved positions (−1300 Elo) while the hand heuristic degraded gracefully (−400), because material terms still work off-distribution but learned features do not. Since the eval lives inside deep search, it MUST be trained on the mid-tactical, non-quiet positions the search visits. This is the NNUE data-generation principle (search-visited positions labeled at low depth) and applies to BOTH the NN and GBDT evals; strong-game trajectories alone underrepresent it.

- :Dataset-curation: (OPTIONAL, deferred until the 1200 gate clears) is an importance+diversity filter on which labelled positions enter the :Cumulative-dataset:, borrowing temperature+MMR from LLM decoding: :Trajectory-sampling: gives move-level diversity, while an MMR-style selector keeps a position only if it is both *informative* — high teacher-vs-current-net value disagreement (hard-example / uncertainty sampling) — and *novel* — dissimilar (cosine over the net's trunk embedding) to positions already kept. Rationale: under the throughput constraint only a few gradient steps run per cycle, so each stored position should be maximally informative and non-redundant. It is a sample-efficiency refinement, OFF until rung 1 (1200) clears, to avoid premature complexity. Note: kept positions' trunk embeddings go stale as the net trains — either re-embed the kept set at curation time or accept the drift.

***implementation reqs***

- `distill_sf.py` owns Stage 1; labels come only from the :Teacher:.
- Constant: TEACHER_DEPTH, MULTIPV_K, POLICY_TARGET_TEMP, LABELS_PER_SEC_FLOOR (≥100/s per engine process) — developer-tuned distillation rules.
- Constant: TRAJECTORY_TEMP_OPENING / TRAJECTORY_TEMP_LATE and TRAJECTORY_OPENING_PLIES — the annealed :Trajectory-sampling: temperature bounds and the ply count over which it decays.
- Constant: CURATION_ENABLED (default false until the 1200 gate) and CURATION_MMR_LAMBDA — the :Dataset-curation: on/off flag and its relevance-vs-diversity balance.
- The trend log persists across runs; it is the Stage-1 observability artifact.

***test reqs***

- A fixed FEN with a known Stockfish depth-8 centipawn score, pinning the tanh(cp/400) convention and its sign.
- A GIL regression check: concurrent labelling + training throughput must stay within a small factor of the separate-process baselines (≈180 labels/s per engine; SGD steps unimpeded).

***functional specs***

- Labelling and training must never share a Python interpreter.
  - Given labelling runs concurrently with torch training, Then labellers occupy separate OS processes; If in-process threads are used instead, Then the run is non-conformant. (Pins the measured failure: threaded labellers + trainer yielded 58 labels/s, ~80 SGD steps in 8 minutes, and *worse* validation than a prior shorter run.)
- Policy targets must be soft where affordable, one-hot as fallback.
  - Given MultiPV output for a position, Then the policy target is softmax(cp_i / POLICY_TARGET_TEMP) over the K candidates with zero mass elsewhere.
  - Given MultiPV is unavailable or drops throughput below LABELS_PER_SEC_FLOOR, Then the one-hot best-move target is the fallback.
- The :Cumulative-dataset: must accumulate, deduplicate, and never regress.
  - Given a new label whose FEN already exists at >= its depth, Then it is not appended.
  - Given a completed run, Then dataset row count and :Measured-elo: are appended to the trend log.
- Stagnation must be surfaced, never annealed past.
  - Given no Elo improvement over M consecutive runs within a gate segment, Then escalation (more data, deeper teacher, more capacity) is flagged to the user — :Gated-progress: never advances to paper over stagnation.
- Positions must come from :Position-source: strong-game trajectories, not uniform random playouts.
  - Given a game generated by the :Teacher: from random opening plies, When positions are sampled from it, Then those in-distribution positions are labelled and appended to the :Cumulative-dataset:.
- :Position-source: games must advance by :Trajectory-sampling:, at no extra search cost.
  - Given the teacher's MultiPV top-K for a position, When the trajectory's next move is chosen, Then it is a temperature-softmax sample over those K candidates — no analyse beyond the one already run for the label.
  - Given opening plies, Then the sampling temperature is TRAJECTORY_TEMP_OPENING (diverse); Given plies past TRAJECTORY_OPENING_PLIES, Then it decays toward TRAJECTORY_TEMP_LATE (near-best), keeping positions near the strong-play manifold.
- The labelled set must include :Search-visited-positions:, not game-trajectory positions alone.
  - Given the eval will be queried inside deep search, When the dataset is built, Then a slice of positions is drawn from search-visited (non-quiet) states — teacher search-tree nodes or trajectory positions perturbed by 1–2 sampled-capture plies — and labelled at the teacher depth.
  - Given an eval trained only on quiet trajectory positions, Then it MUST be expected to degrade on the non-quiet positions search visits (distribution-dependent error); this applies to the NN and GBDT evals equally.
- :Trajectory-sampling: must not weaken the labels.
  - Given any sampled trajectory move, Then each position's value and policy LABEL is still the teacher's full TEACHER_DEPTH eval / MultiPV, independent of which move was sampled to continue the game.
- :Dataset-curation:, when enabled, must select on informativeness AND novelty, never informativeness alone.
  - Given a candidate labelled position and CURATION_ENABLED, Then it is kept only if its MMR score (relevance = teacher-vs-net value disagreement, diversity = 1 − max cosine similarity to kept positions, balanced by CURATION_MMR_LAMBDA) exceeds threshold.
  - Given CURATION_ENABLED is false (the default until the 1200 gate), Then all in-distribution labelled positions are kept (exact-FEN dedup only).
- Stage-1 exit is gated, not scheduled.
  - Given the distilled net (with its conformant search profile) clears the 1200 :Elo-gate: and then the 1600 :Elo-gate:, Then the next stage may begin; Given neither gate has cleared, Then self-play training MUST NOT start.


================================================================================
FILE: spec\search-mcts.spec.md
================================================================================

---
description: 'PUCT tree search with batched leaf evaluation, lineage-blended leaves, and two regression-pinned sign/selection conventions'
import:
  - annealing-schedule
  - prior-evaluator
  - learned-model
  - teacher-distillation
---

***definitions***

- :PUCT-search: is the tree search that selects child actions by Q + c_puct · P · √N/(1+n), where P is the policy prior; the Russian-doll progressive-narrowing search is retained as the small-budget profile of the same contract.
- :Leaf-value: is the estimate assigned to a search-leaf position, formed by blending the operative :Prior-lineage: member with the :Learned-value: by :Teacher-leaf-weight:.
- :Negamax-backup-convention: is the sign rule for backing a leaf value up the path: leaf values are produced White-absolute, MUST be converted to side-to-move-relative at the leaf, and MUST be sign-flipped BEFORE each parent update so each node's Q is always the mover's value at that node. (Regression pin: the backup previously left the leaf White-absolute and flipped after the first update, corrupting Q for Black-to-move nodes and for any simulation whose depth parity put White at the deepest decision — the net-as-Black collapsed in 9–13 plies.)
- :Root-selection-rule: is the budget-aware root move choice: when visits-per-child are below MIN_ROOT_VISITS, the root move MUST be argmax mean-Q over *visited* children; visit-count argmax is permitted only above the threshold. (Regression pin: at ~200 simulations over ~21 root moves every child got ~10–13 visits — counts too flat to separate a free-queen capture with Q=0.93 from quiet moves with Q~0.6, so the search ignored the free queen.)
- :Batched-leaf-evaluation: is the requirement that pending leaves are collected (wave collection or virtual loss) and evaluated in one network batch per wave, exploiting the batch-evaluate API; per-leaf single inference is non-conformant (~2–3 ms each vs 18k–122k/s batched).
- :Progressive-narrowing: is the per-level reduction of sampled move breadth (many candidates near the root, few deep), the small-budget profile's focusing mechanism.
- :Search-value: is the negamax-backed value estimate at a node after its backups — the value the search already computed to select a move. It is exposed in the :White-absolute-frame: so the value-target stage can bootstrap from it for free (no extra inference), and is the improved on-policy value that makes the value target off-policy-safe (tree-backup).
- :Quiescence-search: is the requirement that the net-minimax profile evaluates a leaf ONLY at a quiet position: at a fixed-depth frontier that is non-quiet (mid-capture sequence, hanging piece, in check), the search extends captures (and checks) until quiet before applying the value net. Without it, fixed-depth-2 hands the net non-quiet positions where any static eval is noise — the horizon effect that, not eval quality, caps net-minimax below a quiescence-equipped alpha-beta of *worse* eval. The primary profile also needs the standard stack: iterative deepening, a transposition table, and move ordering.
- :Policy-move-ordering: is the consumer of the :Policy-head: inside the net-minimax profile (minimax does not otherwise read P): the policy distribution orders moves for search and selects the beam. This closes a dead-weight defect (the policy head would otherwise be unused in the primary profile) and makes cloning self-reinforcing — a better policy yields better ordering, hence deeper effective search, hence better cloning targets.
- :Search-window-reuse: is the receding-horizon carry of the scored sampled subtree across moves: after the played move and the opponent's reply, the surviving subtree is kept as a warm start, the window shifts by the resolved plies, and one fresh fetch_k frontier layer is added — so effective horizon accumulates across moves at constant per-move budget, densifying the sparse sample along the played line. It is LEAKY: only the subtree under the opponent's ACTUAL reply survives, so a reply outside the sampled top_k re-seeds (the ponder-miss); policy-guided fetch reuses more than random fetch. The realized advantage :Delta: = (a move's backed-up value after the deeper re-search) − (its value estimated when it was selected) is the SIGNED TD error (unlike the non-negative :Move-margin:), and is exactly the :Search-value: the value-target stage bootstraps from — one search yields the move, the margin, and the training target.
- :Phi-rotation: is scheduled iterative deepening over the Fibonacci widths: each pass shifts one phi-step of width into one phi-step of depth, per layer, gated on that layer's top_k ORDERING being unchanged across the last two passes (unstable ordering ⇒ the layer is unsaturated ⇒ hold its width). Stability is NECESSARY, NOT SUFFICIENT — two jointly-too-shallow passes can agree yet be wrong; that error is corrected later by :Delta:, so premature deepening is bounded-lag, not permanent. Narrowing applies to EXPANSION ONLY, never RETENTION: scored candidates stay in the table (root-layer candidates indefinitely — bounded and cheap, they are the decision; deeper shelved siblings under a recency/decay eviction, being re-derivable), and narrowing only stops spending fan-out outside the shrunk top_k. If the incumbent line's backed-up value drops between passes (:Delta: < 0), the shelved siblings are re-opened at the shallowest divergent layer — aspiration-window fail-low re-search — so the rotation is not a one-way ratchet into a refuted line. (Together with :Search-window-reuse: this re-derives iterative deepening + aspiration windows + transposition retention for the sampled beam.)
- :Decision-rule: is that the played move is the argmax over ROOT MOVES of their backed-up (alternating max/min) value; raw comparison of leaf scores ACROSS DEPTHS is FORBIDDEN — an interior leaf-eval is superseded by its subtree's backup, and a high value sitting behind an opponent refutation is unreachable. In a sampled beam these backed-up values are sample-OPTIMISTIC (an unsampled opponent refutation inflates the alternating max/min): correct as the target, biased high until :Search-window-reuse: densifies and the :Delta: < 0 re-widening of :Phi-rotation: refutes — so :Decision-rule: and that re-widening are one anti-optimism loop.
- :Move-margin: is max − mean(top_k) over the surviving lines' root-perspective backed-up values — the DECISIVENESS of the winner over the field it beat (always ≥ 0, distinct from the signed :Delta:). It calibrates compute and difficulty: margin ≥ θ_easy for two consecutive passes terminates deepening early (easy-move detection) and marks the position as safe for :Strength-temperature: sampling; a low margin marks the position as ambiguous — exactly where deepening compute (and honest play) belong.
- :Root-commit: is the rule that the beam's FINAL pick is argmax over root-perspective backed-up values (τ→0 at the terminal level only), NEVER a temperature sample. The exploration budget is spent UPSTREAM — uniform fetch, temperature/MMR survival pruning, and the ε-mixture's direct-π branch — so committing the max at the root does not starve on-policy data (the played max is already max-over-a-random-subset, itself stochastic across draws). Committing the exploited move is also what makes the next volley's :Delta: a clean AUDIT: δ measures the move you would actually play, not a lottery; and δ's SPREAD across draws is the winner's-curse / sample-optimism magnitude that drives pair_k and the fetch random→policy shift.
- :Volley-growth: is the bounded schedule-deepening across re-searches (volleys): volley t uses fib_schedule(depth₀+t, 1, max(1, phi_start₀−t)), so the root width is HELD and one thin level is APPENDED at the bottom per volley — append-down, NEVER shift-down (shifting deletes a level at the phi_start floor rather than deepening). Planned growth of +k depth requires phi_start₀ = k+1; growth caps at +(phi_start₀−1). Appended tail levels are width 1–2 (single-digit Fibonacci), so a deep thin tail is one noisy principal variation — prefer widening the appended level or holding (:Growth-gate:) over ratcheting width-1 tails.
- :Growth-gate: extends depth by another level only WHILE the root :Move-margin: < θ_easy (position still ambiguous — deepening can change the decision) OR the incumbent line's :Delta: is unstable across the last two volleys (backups still revising); a high, stable margin BANKS the compute (easy move). This is :Value-of-information: steering the growth schedule, not merely early-stop.
- :Reuse-precedence: ranks the two horizon-growth mechanisms: the schedule (:Volley-growth:) is MINOR and bounded (+phi_start₀−1); the receding window (:Search-window-reuse:) is MAJOR and compounds every volley the opponent's reply stays inside the carried tree, but LEAKS to zero when it does not. The reuse HIT RATE (fraction of opponent replies found in the sampled subtree) MUST be logged — it decides when policy-guided fetch pays over uniform fetch, and pair_k is itself a reuse-persistence knob (larger pair_k covers more replies at linear cost), not only a per-move quality knob.

***implementation reqs***

- The search lives in `mcts.py`; it reads coefficients from the :Annealing-schedule:, it does not compute its own schedule inline.
- Constant: MIN_ROOT_VISITS — the visits-per-child threshold of the :Root-selection-rule:.
- :Gated-progress: must be threaded to the leaf evaluation, not stopped at the search entry point.
- `net_search.py` (batched fixed-depth minimax with beam pruning, ~0.9 s/move at depth 2) is the interim conformant fast profile until PUCT lands.
- The alternating max/min fold is implemented as ONE sign trick — a parent takes `max over −child` in the side-to-move frame — so after an even ply count the value is back in root perspective. Its two code sites are the `(−1)^k` alternation in the n-step returns and the `mover_val = -v` negamax commit in the beam fan-out; both trace to :Negamax-backup-convention:.

***test reqs***

- The historical failing position — opponent has just left a queen en prise (e.g. `rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w`) — with a ~200-simulation budget, to assert the :Root-selection-rule:.
- A mirror pair of positions backed up through alternating plies, to assert the :Negamax-backup-convention: for both colors.
- A midgame position plus a stub network returning a fixed value, to assert the leaf blend endpoints.
- A TRAP position where a move wins material immediately but loses it to a forced reply (poisoned pawn / trapped queen, the Qxb7-behind-...Rb8 shape): its eval right after the capture is positive but its backed-up value is negative. Assert (a) :Decision-rule: DECLINES it — the move backs up to the min over the opponent's replies (not the best leaf in its subtree), so the quiet alternative is played; and (b) a beam whose pair_k is below the refutation's policy rank backs the trap up OPTIMISTICALLY (the pre-capture positive value), pinning the sample-optimism bias, which the next move's :Search-window-reuse: :Delta: < 0 corrects.

***functional specs***

- The :Negamax-backup-convention: must hold on every backup.
  - Given a leaf value backed up through alternating plies, Then the sign presented to each node is that node's mover-relative value (flip BEFORE each update, starting from a side-to-move-relative leaf).
- The :Root-selection-rule: must hold at small budgets.
  - Given the free-queen test position and a ~200-simulation budget, When the search completes, Then the capture is the root choice.
  - Given visits-per-child below MIN_ROOT_VISITS, Then unvisited children are ineligible and the choice is argmax mean-Q; Given visits above the threshold, Then visit-count argmax is permitted.
- :Batched-leaf-evaluation: must bound network calls.
  - Given N pending leaves in a wave and network max batch B, Then network calls <= ⌈N/B⌉.
- :Leaf-value: must blend by :Teacher-leaf-weight: along the :Prior-lineage:, so earlier priors guide early search and the learned network takes over late.
  - Given progress 0, When a non-terminal leaf is evaluated, Then :Leaf-value: is approximately the operative prior's normalized score.
  - Given progress 1, Then :Leaf-value: is approximately the :Learned-value:.
- Terminal leaves must short-circuit the blend with fixed outcome values in the mover-relative frame required by the :Negamax-backup-convention:.
  - Given a checkmate leaf, Then the value is a loss for the side to move, independent of the blend weight.
- If network evaluation fails, Then the leaf must fall back one :Prior-lineage: step rather than abort the simulation.
- PUCT priors must come from the annealed policy blend.
  - Given :Teacher-policy-weight: w, Then P is the frozen teacher policy snapshot blended with the current learned policy by w, softened by :Prior-bias-temperature: before selection.
  - Given rising progress, When a node expands, Then reliance on teacher/prior bias loosens, letting the learner explore off-book moves.
- :Progressive-narrowing: must be preserved in the small-budget profile; the annealed sample widths come from the :Annealing-schedule:, keeping one source of truth.
  - Exploration must stay structured (PUCT/UCB plus weighted sampling); widening must not degrade to uniform-random legal moves except as a last-resort fallback.
- The net-minimax profile must apply the value net only to quiet leaves (:Quiescence-search:).
  - Given a fixed-depth frontier position that is non-quiet (a capture is pending, a piece hangs, or the side to move is in check), When it would be evaluated, Then the search first extends captures/checks until quiet and evaluates there.
  - Given a quiet frontier position, Then the value net is applied directly.
- The :Policy-head: must have a consumer in the net-minimax profile via :Policy-move-ordering:.
  - Given the policy distribution P at a node, When moves are ordered/pruned for search, Then ordering and beam selection follow P (better P → better ordering → deeper effective search → better cloning targets).
- The batching/pruning tension must be acknowledged, not assumed away.
  - :Batched-leaf-evaluation: (parallel frontiers) and deep alpha-beta pruning (sequential) pull opposite ways; the beam-pruned batched minimax is a deliberate middle path. Depth targets MUST NOT be specified assuming true alpha-beta branching factors — expect weaker pruning than incremental-update CPU NNUE.
- :Decision-rule: must select over root backed-up values, never raw cross-depth scores.
  - Given leaf values at different depths, When the root move is chosen, Then selection is argmax over ROOT moves' alternating-max/min backed-up values; a raw leaf score is never compared to one at a different depth.
  - Given a root move whose high value stands behind an opponent refutation in the beam, Then that value MUST be backed up (min at the opponent ply) before the move is eligible.
  - Given a move, When its value is formed, Then it is the leaf value at the END of its principal variation, folded up one ply at a time by alternating max (the mover) / min (the opponent) — equivalently `max over −child` in the side-to-move frame — and the position's eval IMMEDIATELY AFTER the move is superseded the instant the move is expanded. The best raw leaf anywhere in the subtree is NOT the value: every deep score is gated by the opponent's min nodes between the root and it (raw max over all leaves assumes the opponent cooperates). Worked fold — White to move, Qxb7 (eval +1.0 right after the capture) vs Nf3: after Qxb7 ...Rb8 traps the queen, White's deepest choice max(−7.0,−7.5)=−7.0, Black's node min(−7.0,+1.0)=−7.0, so Qxb7 backs up to −7.0 (the +1.0 is overwritten); Nf3 backs up to +0.2; argmax(−7.0,+0.2) ⇒ play Nf3.
- :Phi-rotation: must narrow expansion without discarding scored candidates, and self-correct on :Delta: < 0.
  - Given a layer narrowed at an earlier pass, When a later pass needs a shelved sibling, Then it is still in the table (root-layer indefinitely, deeper siblings under decay) and re-openable — narrowing stopped fan-out, not retention.
  - Given the incumbent line's backed-up value drops between passes (:Delta: < 0), Then the shelved siblings at the shallowest divergent layer are re-opened (aspiration fail-low re-search).
  - Given a layer's top_k ordering changed across the last two passes, Then its width is held, not rotated into depth.
- :Move-margin: must gate early termination of deepening.
  - Given :Move-margin: ≥ θ_easy for two consecutive passes, Then deepening terminates early for that move (easy-move detection); Given a low margin, Then deepening continues (the compute belongs there).
- :Root-commit: must confine stochasticity to fetch and survival, not the final pick.
  - Given the terminal beam level, Then the played move is argmax over root backed-up values; Given the fetch and survival-prune stages, Then temperature/MMR sampling applies there (and the ε-mixture's π branch supplies on-policy exploration).
- :Volley-growth: must append-down with a non-shrinking root, and :Growth-gate: must condition extension.
  - Given volley t, Then widths = fib_schedule(depth₀+t, 1, max(1, phi_start₀−t)) and widths[0] is non-decreasing across volleys (append-down, never shift-down).
  - Given the root :Move-margin: ≥ θ_easy and stable :Delta: across two volleys, Then depth is held and compute banked; Given a low margin or unstable :Delta:, Then one level may be appended.
- :Reuse-precedence: must log reuse and cap schedule growth.
  - Given a volley, Then the reuse hit rate (opponent reply found in the carried subtree) is logged; horizon beyond +(phi_start₀−1) MUST come from :Search-window-reuse:, not the schedule.
- :Search-window-reuse: must carry the subtree and expose :Delta:.
  - Given the opponent's reply lies in the retained subtree, Then it is reused as a warm start and only a fresh frontier layer is expanded; Given the reply lies outside the sampled top_k, Then the window re-seeds (ponder-miss).
  - Given a move's deeper re-search completes, Then :Delta: = (new backed-up value) − (prior selection estimate) is exposed as the :Search-value: the value-target bootstraps from.
- The search must expose its :Search-value: alongside the chosen move and visit distribution, in the :White-absolute-frame:, so the value-target stage can bootstrap from it.
  - Given a completed search at a node, Then the negamax-backed node value is available to the caller without re-running inference.


================================================================================
FILE: spec\value-target.spec.md
================================================================================

---
description: 'The value-head learning target: a search-bootstrapped lambda-return shared by the refinement and self-play stages'
import:
  - annealing-schedule
  - search-mcts
  - learned-model
---

***definitions***

- :Lambda-return: is the value target for a stored trajectory step: G_t^λ, the exponentially-weighted average of n-step returns, computed in the recursive TD(λ) form G_t = r_t + γ·[(1−λ)·V_boot + λ·G_{t+1}] for non-terminal t and G_t = r_t at a terminal. It is White-absolute in [-1,1]. It unifies TD(0) and Monte-Carlo: λ=0 collapses it to :Search-bootstrap-value:-based TD(0) (r + γ·V), λ=1 collapses it to the discounted terminal outcome (Monte-Carlo z). Implemented in `value_targets.py`.
- :Search-bootstrap-value: is the V_boot each n-step return bootstraps from: the :Search-value: (the negamax-backed node value from the search already run to pick the move), falling back to the target-net V(s′) when no search value exists (e.g. a shallow measurement profile), then to 0. Using the search value gives the tree-backup property that makes the target off-policy-safe.
- :Off-policy-correction: is the (default-absent) importance-sampling correction. Tree-backup safety covers only the BOOTSTRAP term (V_boot is a freshly computed :Search-value: at s′, independent of the behaviour policy). The λ-weighted TAIL G_{t+1} chains through the actual stored trajectory, so at λ near 1 the target is essentially the behaviour policy's Monte-Carlo outcome, UNCORRECTED. This is sound only while replay is recent — hence :Replay-window:. A truncated-IS Retrace(λ) weight is available behind a flag (default OFF) for staler data.
- :Replay-window: is the bounded window of the last REPLAY_WINDOW self-play games from which trajectories are sampled. It is the recency guard that makes the uncorrected λ-tail sound (AlphaZero's implicit fix): fresh trajectories are near-on-policy, so the uncorrected Monte-Carlo tail stays low-bias.
- :Advantage-shrinkage: is a SIGNIFICANCE filter on the advantage A = G_t^λ − V(s_t): a transition enters the policy update only if its advantage is statistically distinguishable from 0 given its uncertainty — |A| / σ_A > z_α, equivalently the interval A ± z_α·σ_A excludes 0 (the regression-coefficient t-test read off a whitened advantage; "act only on significant changes"). It DROPS insignificant transitions (a prioritized-replay filter), not zeroes them — zeroing a kept sample saves no compute (the backward pass still runs), whereas dropping focuses gradient on the capability boundary (:Value-of-information:) and denoises Bellman-residual noise. Mechanism, named precisely: this is a HARD significance gate — keep-at-full-value or DROP — i.e. subset selection / L0, NOT soft-threshold shrinkage (L1-prox would shrink each kept |A| by ~z_α·σ_A while keeping every sample, saving no compute); the term "shrinkage" names the sparsifying effect on the signal, not an L1 shrink of retained values. Whitening MUST be ZCA (zero-phase), not PCA, so within-trajectory correlated advantages are decorrelated IN THE ADVANTAGE BASIS and each transition's significance test is independent. It is SELF-ANNEALING: as V converges σ_A shrinks, so a FIXED α automatically admits progressively finer real advantages — this supersedes any hand-tuned magnitude dead-zone ε (deleted). σ_A is estimated cheaply as the batch advantage spread (= standard advantage normalization → significance RELATIVE to the batch, not an absolute p-value), or better from the λ-return's n-step variance / the :Search-window-reuse: δ-spread. Multiple-comparisons applies (at α over a B-batch, ~α·B pass by chance): control the false-discovery rate (Benjamini-Hochberg) across the batch, not a per-transition α, when it bites. STAGE-2+ only — advantage exists only once the λ-return refinement runs; Stage-1 distillation (supervised MSE) has none. NOT a precision/bit trick: A is a transient fp32 scalar; the memory win lives in :Encoding-packing: (learned-model).
- :Distillation-anchor: is an optional small, persistent SUPERVISED term blended into the self-play value loss: DISTILL_ANCHOR_ALPHA · MSE(V(s), tanh(SF_cp(s)/400)) over a held Stockfish-labelled anchor set, ADDED to the self-play outcome/λ-return loss. It tethers the value head to Stockfish's calibrated evals where they exist, guarding the self-play stages against drift / reward-hacking — the value analog of DEMO_SHARE_FLOOR (data) and the shaped-reward floor (reward). DISTILL_ANCHOR_ALPHA anneals from high early (lean on the teacher) toward a small FLOOR (never 0 — a permanent anchor), the same prior-lineage shape as :Bootstrap-share:. It is a REGULARIZER, not the objective: self-play outcomes remain the primary value signal and are the only channel that can EXCEED the teacher. OFF by default (α=0) in the pure heuristic-vs-distilled bootstrap comparison, which measures UNAIDED self-play; ON as the anti-drift anchor when chasing the :Surpass-teacher-gate:.

***implementation reqs***

- The return math lives in pure, stateless functions in `value_targets.py` (`td0`, `mc_return`, `nstep_return`, `lambda_return`, `retrace_weights`), decoupled from any trainer; consumed by the Stage-2 refinement trainer and the Stage-3 self-play trainer.
- λ is derived from the schedule's :Bootstrap-share: β as λ = 1 − β; there is no separate λ constant.
- Constant: GAMMA — **pinned to 1** for the value target (chess is episodic with a bounded horizon and no natural discount, matching AlphaZero). With γ=1, λ=1 is exactly the terminal outcome z; any γ<1 would make λ=1 a *discounted* z, so the "collapses to z" claim holds only at γ=1.
- Constant: REPLAY_WINDOW — the :Replay-window: size (last-N self-play games) that keeps the uncorrected λ-tail near-on-policy.
- Constant: DISTILL_ANCHOR_ALPHA (start/floor) — the annealed :Distillation-anchor: weight; the anchor set is the held Stockfish-labelled :Cumulative-dataset:. Floor > 0 (permanent tether), α=0 disables it (the pure-self-play comparison).
- Constant: ADV_ALPHA — the :Advantage-shrinkage: significance level (z_α). The filter ZCA-whitens advantages and is applied at replay sampling, so dropped transitions cost no gradient step; the default σ_A estimator is the batch advantage spread (advantage normalization). No annealed magnitude ε — the significance test self-anneals via σ_A.
- The White-absolute ↔ side-to-move frame conversion is a single shared helper (`to_stm`/`to_white`), the same sign rule as the :Negamax-backup-convention:.

***test reqs***

- A scripted trajectory asserting :Lambda-return: with λ=0 equals the exact `neural_network.py:274` TD(0) formula per step (backward-compat pin), and with λ=1 equals the discounted Monte-Carlo return.
- A trajectory whose terminal is reached before n steps, asserting n-step returns cap at the terminal (collapse to Monte-Carlo).
- A frame round-trip asserting to_white(to_stm(v, stm), stm) == v and a sign flip for Black.
(All four are pinned in `test_value_targets.py`.)

***functional specs***

- The value target must be the :Lambda-return: with λ = 1 − :Bootstrap-share:, bootstrapped from the :Search-bootstrap-value:.
  - Given a stored trajectory and schedule β at the game's :Gated-progress:, When value targets are formed, Then each target is the (1−β)-return bootstrapped from the :Search-value:.
- Missing search values must degrade gracefully, never abort.
  - Given a step with no :Search-value:, Then the bootstrap falls back to the target-net V(s′).
- The default path must apply no importance-sampling correction, and must lean on recency instead.
  - Given the default configuration, When value targets are formed, Then :Off-policy-correction: is absent: the bootstrap term is tree-backup-safe and the λ-tail is recency-guarded by sampling only the last REPLAY_WINDOW games.
  - Given trajectories staler than :Replay-window:, Then Retrace(λ) must be enabled (flag) — the uncorrected tail is not safe on stale data.
- The target must reduce to the legacy rule at the endpoints, so the change is a strict generalization.
  - Given β = 1 (λ = 0), Then the target equals r + γ·V(s′) — the current `neural_network.py:274` behaviour.
  - Given β at its floor (λ → 1), Then the target approaches the Monte-Carlo outcome used by AlphaZero self-play.
- :Advantage-shrinkage: must DROP by significance, not zero by magnitude, and must whiten with ZCA.
  - Given |A| ≤ z_α·σ_A (the confidence interval includes 0), Then the transition is excluded from the update batch — not kept with A zeroed (a kept-and-zeroed sample still runs the backward pass).
  - Given correlated within-trajectory advantages, Then they are ZCA-whitened (not PCA) before the per-transition significance tests, so the tests are independent in the advantage basis.
  - Given V converging (σ_A shrinking), Then a fixed ADV_ALPHA admits progressively finer advantages — there is no separate magnitude-ε anneal.
  - Given a large batch filtered at per-transition α, Then false-discovery control (Benjamini-Hochberg) governs, not a raw per-test α.
  - Given Stage 1 (supervised distillation), Then :Advantage-shrinkage: does not apply (no advantage exists until the λ-return refinement).
- :Distillation-anchor: must be a floored regularizer, not the objective.
  - Given self-play value training with the anchor ON, Then the loss is (self-play outcome/λ-return MSE) + DISTILL_ANCHOR_ALPHA·MSE(V, SF-label); Given rising :Gated-progress:, Then DISTILL_ANCHOR_ALPHA anneals toward its floor (never 0) so the net stays tethered to calibrated evals while self-play outcomes drive improvement past the teacher.
  - Given the pure bootstrap comparison, Then the anchor is OFF (α=0) so unaided self-play is what the :Ladder: measures.


================================================================================
FILE: spec\self-play-leela.spec.md
================================================================================

---
description: 'Stage 3 — expert iteration: PUCT self-play, visit-count policy distillation, outcome value regression, surpassing the teacher'
import:
  - elo-measurement
  - annealing-schedule
  - learned-model
  - teacher-distillation
  - search-mcts
  - value-target
---

***definitions***

- :Expert-iteration-cycle: is one turn of the improvement loop: a strong search plays games and emits (position, a policy target, and the value target); the net's heads are trained toward those targets; search over the improved net then produces better targets. Search is the policy-improvement operator; no policy gradient is involved. The value target is always the :Lambda-return: (value-target); the policy target and the cost depend on the :Search-profile:.
- :Search-profile: is which search generates self-play — two conformant options with the SAME value target:
  - **Net-minimax (PRIMARY, hardware-favored):** the trained value net inside deep batched `net_search` (the NNUE-shaped path). Policy target = the search's chosen move (behavioral cloning of the deeper search). Cheap (~seconds/move, no per-move playout budget) — this is what makes self-play affordable here.
  - **PUCT (research arc):** hundreds of playouts/move producing a :Visit-distribution-target:. Higher-quality policy targets but ~100× the per-move cost; used only when compute allows.
- :Off-policy-handoff: is the annealed shift of the self-play behaviour/data source from the Stockfish :Teacher: to the net's own :Search-profile: play, driven by :Demo-share:. It is sequenced by DATA QUALITY, not compute: SF is the stronger player (better data) until the net approaches it, after which self-play generates data that can EXCEED any fixed teacher — the mechanism for surpassing the teacher toward frontier.
- :Visit-distribution-target: is the normalized root visit count vector at move temperature τ, stored as the policy target for the position; it is a valid target only when the game's simulation budget made visits informative.
- :Training-game-budget: is the per-move simulation budget used in self-play *training* games: it must guarantee visits-per-candidate at or above the informative threshold, reconciling visit targets with the :Root-selection-rule: (which exists precisely because low-budget visits are uninformative). Fast low-budget profiles are for measurement and human games, not for generating policy targets.
- :Root-dirichlet-noise: is constant Dir(α) noise mixed into the root prior in self-play training games only — the explicit, bounded carve-out from "never anneal toward randomness": it is fixed structured exploration, never annealed, and always OFF in :Measurement-game:s.
- :Surpass-teacher-gate: is the :Elo-gate: at the :Teacher:'s own anchor-measured strength; only after it clears may :Demo-share: reach zero.
- :Self-play-bootstrap: is the leaf evaluator the self-play search STARTS from, along the :Prior-lineage:. Two valid entry points: (a) the HAND HEURISTIC via an eps-blend leaf — board_eval = eps·tanh(pst/…) + (1−eps)·:Learned-value:, eps annealing 1→0 so the heuristic guides early self-play and the net takes over (pure prior-lineage, NO teacher in the loop); (b) a DISTILLED net snapshot (eps=0 from a checkpoint, e.g. the Stockfish-distilled tower — warmer start). SF is never in the self-play loop under either; it is only the :Ladder: gate. The two bootstraps are compared as a :Ladder:-measured experiment — each one's per-iteration ladder curve — to see whether unaided heuristic self-play catches the distilled start (and whether either catches SF).
- :Strength-matched-opponent: is an OPTIONAL self-play opponent mode for early training: rather than a second network, one side is the same net with its selection temperature modulated to play "just above" the reference player's strength, at a setpoint of mean + k·σ of that player's per-move quality (regret) distribution. It reuses the regret tracking in `difficulty.py` (restoring the σ term the human-difficulty path dropped) and the temperature↔strength map in `elo_calibration.py`. It is a zone-of-proximal-development / matched-difficulty curriculum, NOT gating, league, or a separately-trained second net.

***implementation reqs***

- Self-play data generation and torch training must occupy separate OS processes (the :Process-separated-labeling: rule generalizes to any data-gen + training concurrency).
- Constant: DIRICHLET_ALPHA, MOVE_TEMPERATURE_PLIES (τ > 0 only for the opening plies), TRAINING_SIM_BUDGET, VISITS_PER_CANDIDATE_MIN.

***test reqs***

- A completed self-play game record, to assert target frames: policy targets are distributions over legal moves, value targets equal the game outcome in the :White-absolute-frame: at every stored position.

***functional specs***

- The :Self-play-bootstrap: must be a prior-lineage leaf, annealed toward the net, with SF out of the loop.
  - Given the heuristic bootstrap, Then early self-play leaves are the hand heuristic (eps→1), eps anneals to 0, and the net trains on the resulting outcomes with no Stockfish in the loop.
  - Given the distilled bootstrap, Then self-play starts from a distilled net snapshot (eps=0).
  - Both bootstraps are placed on the :Ladder: per iteration for comparison; SF is the gate, never the in-loop teacher.
- Self-play must emit expert-iteration targets; the value target is profile-independent, the policy target is profile-specific.
  - Given a self-play move under the net-minimax :Search-profile:, Then the stored policy target is the deeper search's chosen move (cloning) and the stored value target is the :Lambda-return: in the :White-absolute-frame:.
  - Given a self-play move under the PUCT :Search-profile:, Then the stored policy target is the root :Visit-distribution-target: and the value target is the same :Lambda-return: (which reduces to outcome z as :Bootstrap-share: β → its floor).
- The :Off-policy-handoff: must be data-quality-sequenced, not compute-gated.
  - Given the net measures weaker than the :Teacher:, Then :Demo-share: keeps the off-policy source predominantly SF (better data); Given the net approaches the :Teacher:, Then the source anneals toward the net's own :Search-profile: self-play, which may exceed the teacher.
  - Self-play is NOT withheld because it is expensive — the net-minimax profile is cheap; it is sequenced after distillation only because SF is the stronger data source until the net catches up.
- Replay batches must honor the annealed :Demo-share:.
  - Given :Demo-share: d at current :Gated-progress:, Then each training batch draws fraction d from the :Cumulative-dataset: and 1−d from self-play replay.
- Exploration noise must stay inside training games.
  - Given a :Measurement-game:, Then :Root-dirichlet-noise: and move temperature are disabled and the agent plays argmax.
  - Given a self-play training game, Then :Root-dirichlet-noise: is applied at the root with constant α — it is never a schedule knob.
- Policy targets must be budget-qualified.
  - Given a training game whose simulation budget satisfied visits-per-candidate >= VISITS_PER_CANDIDATE_MIN, Then its policy targets are stored.
  - If the budget fell short, Then the game's policy targets are discarded while its value targets may be kept.
- The teacher is outgrown by measurement, not by schedule.
  - Given the :Surpass-teacher-gate: clears (measured Elo > the teacher's anchor-measured Elo), Then Stage 3 may continue teacher-free (:Demo-share: floor 0) and the teacher remains only as a measurement reference.
  - Given the gate has not cleared, Then :Demo-share: >= DEMO_SHARE_FLOOR (guards catastrophic forgetting of teacher knowledge).

- The :Strength-matched-opponent: is optional, early-only, and annealed out — the standard mode is full-strength symmetric self-play. (This encodes the critique: a deliberately temperature-weakened opponent produces easier, lower-quality games and noisier visit-count/value targets; keeping it late would cap target quality. Its value is faster early improvement against an opponent slightly above the learner, not final strength.)
  - Given the :Strength-matched-opponent: is enabled, When it selects a move, Then its temperature is modulated so its expected per-move quality tracks the setpoint mean + k·σ of the reference player's regret distribution (opponent slightly stronger than the learner).
  - Given rising :Gated-progress:, Then the σ-offset k and the matched-opponent's temperature anneal toward zero, converging to full-strength symmetric self-play (both sides argmax + :Root-dirichlet-noise:) before the targets are used to chase the :Surpass-teacher-gate:.
  - Given the :Strength-matched-opponent: is disabled (the default), Then both sides play the same full-strength net with :Root-dirichlet-noise:.


================================================================================
FILE: spec\training-loop.spec.md
================================================================================

---
description: 'The staged training loop: gate-driven stage control, schedule application, reward assignment, and failure-mode monitoring'
import:
  - elo-measurement
  - annealing-schedule
  - prior-evaluator
  - learned-model
  - teacher-distillation
  - search-mcts
  - value-target
  - self-play-leela
---

***definitions***

- :Stage-controller: is the orchestrator of the three training stages — Stage 1 teacher distillation, Stage 2 annealed off-policy refinement, Stage 3 expert iteration — whose transitions are gated exclusively by :Elo-gate:s, never by wall-clock or game count.
- :Elo-trend: is the persisted per-run sequence of (dataset size, training volume, :Measured-elo:) points — the primary monitored signal across runs; loss curves are secondary diagnostics only.
- :Self-play-game: is one game the agent plays against itself, producing a stream of :Replay-transition:s (Stage 2) or expert-iteration targets (Stage 3) and a set of per-game monitoring metrics.
- :Teacher-agreement: is the fraction of moves in a game where the played move equals the operative :Prior-lineage: member's greedy move — a lock-in / derivative-play signal that should fall as the learner outgrows its prior. (Generalizes the earlier prior-agreement metric.)
- :Opening-diversity: is the count of distinct first moves seen across recent games — a policy-collapse signal that should stay above one.
- :Ahead-but-lost: is the count of games the mover led by a wide evaluation margin yet did not win — a reward-hacking signal.
- :Stop-training: is the three-gate rule for ending a training stage. adv = G − V(s); the baseline drives its mean toward 0, so E[adv²] is Bellman-residual energy — a cheap proxy for critic/policy fit ON THE SELF-PLAY DISTRIBUTION, not strength. Low E[adv²] is FOUR-way ambiguous: nothing left to learn (victory), entropy collapse (the policy stopped visiting surprising states), a frozen distribution, or the critic having memorised/overfit the self-play data — and three of the four mimic convergence. So a stage MUST require ALL of: a robust (median/Huber, mate-spike-resistant — raw MAD is insufficient) EWMA of E[adv²] below θ for K consecutive checkpoints, AND the external :Measured-elo: rung plateaued (the expensive probe that alone catches overfit), AND policy entropy at or above a floor defined so it holds :Opening-diversity: > 1 (rejecting collapse). A DISTINCT stop retires the search arm, not training: when the beam's measured excess over the zero-search policy (:Compute-frontier: advantage vs the bare-policy rfr) reaches 0, search is distilled into the policy and ε may retire.
- :Value-of-information: is the law unifying every stop/allocate decision in the system — spend compute where estimates still disagree, stop where they have stopped disagreeing. adv-variance measures disagreement across STATES (training), :Search-window-reuse:'s δ across PASSES (per line), :Move-margin: across SIBLINGS (per move), :Phi-rotation:'s ordering-stability across LAYERS (width). One statistic at four scopes; it is the epistemic (uncertainty-reduction) term of expected free energy.

***implementation reqs***

- The loop lives in `chess_ai.py`; the :Stage-controller: reads :Measured-elo: from the measurement authority and computes :Gated-progress: for the schedule (the schedule itself stays pure).
- Monitoring metrics are stored per game in the existing game-history record and plotted alongside loss and :Elo-trend:s.
- Any concurrent data generation (labelling or self-play) and torch training occupy separate OS processes (:Process-separated-labeling: generalized).

***functional specs***

- Stage transitions must be gated, not scheduled.
  - Given Stage 1 has not cleared the 1200 :Elo-gate:, Then Stage 2 self-play MUST NOT start (self-play data from a weak policy is noise, not signal).
  - Given a gate has not cleared, Then :Gated-progress: clamps at its segment boundary and all schedule coefficients hold.
- Every run must obey the :Run-contract: regardless of stage.
  - Given any run ends, Then checkpoint, dataset, and an :Elo-trend: point persist, so the next ≤5-minute run resumes cumulatively.
- :Self-play-game: must compute :Gated-progress: once at game start and drive all schedule reads from it.

  Input: board — the game position, chess.Board
  Parameters: max_moves ∈ ℤ⁺
  Initialize: progress ← stage-controller's :Gated-progress:   # global to this game, clamped to [0,1]
  Initialize: beta ← schedule.bootstrap_share(progress)        # value-target :Bootstrap-share:, λ = 1 − beta
  Initialize: teacher_hits ← 0                                  # global to this game

  Loop while board is not game-over and move_count < max_moves:
      state_before ← copy of board                               # transient
      move ← search selects a move at progress                   (Require: move is legal)
      record whether move equals the operative prior's greedy move into teacher_hits
      push move onto board
      When board is checkmate:
          reward ← +1 if the checkmated side is the opponent else -1   # White-absolute
          done ← true
      Otherwise When board is stalemate or insufficient material:
          reward ← 0
          done ← true
      Otherwise:
          reward ← shaped_reward_weight(progress) × (γ·Φ(board) − Φ(state_before))   # potential-based (Ng '99); Φ = normalized operative-prior score
          done ← false
      store the transition with its value target as the :Lambda-return: at λ = 1 − beta, bootstrapped
        from the step's :Search-value: (Stage 2 transition / Stage 3 expert-iteration targets)
      When buffer has at least BATCH_SIZE samples:
          run one training step honoring :Demo-share: and append its loss
  Assert: every stored terminal reward is one of {+1, 0, -1}

  Given White mates on the final move, When the game ends, Then the last stored reward MUST be +1.
  Given late-game progress, When a quiet move is scored, Then shaped reward magnitude SHOULD be small versus a terminal ±1.

- :Teacher-agreement: must be derived as teacher_hits / move_count and stored on the game record.
  - Given a game where every move matched the operative prior, Then :Teacher-agreement: is 1.0 (maximal lock-in warning).
- :Opening-diversity: must be updated from first moves across games and stored for plotting.
  - If :Opening-diversity: stays at 1 across many games, Then policy collapse is flagged for the user.
- :Ahead-but-lost: must be incremented when a side held a wide evaluation lead yet failed to win.
- Monitoring metrics must be observable: the training-plot step must render :Teacher-agreement:, :Opening-diversity:, and the :Elo-trend: to disk after training.
- :Stop-training: must require all three gates; any single gate alone MUST NOT stop a stage.
  - Given robust-EWMA(E[adv²]) < θ for K checkpoints but the rung :Measured-elo: still climbing, Then training continues (the cheap proxy saturated, capability did not).
  - Given robust-EWMA(E[adv²]) < θ and the rung eval plateaued but policy entropy below its floor (:Opening-diversity: at 1), Then STOP is refused and collapse is flagged — low variance here is collapse, not convergence.
  - Given all three hold (low robust adv² for K checkpoints, rung plateau, entropy ≥ floor), Then the stage may stop.
  - Given the beam's :Compute-frontier: advantage over the zero-search policy reaches 0, Then the search/ε arm retires (search distilled into the policy), independently of the training-stop gates.
- The :Value-of-information: law must be the shared rationale for every allocate/stop decision: compute is spent where estimates still disagree (high adv-variance states, unstable :Phi-rotation: layers, low :Move-margin: moves, :Delta:-divergent lines) and withdrawn where they agree.


================================================================================
FILE: spec\dynamic-difficulty.spec.md
================================================================================

---
description: 'Adapt the computer opponent to the human player by tracking move quality and tuning selection temperature'
import:
  - learned-model
  - search-mcts
---

***definitions***

- :Move-regret: is a move's quality on a position-independent scale: the learned value of the position after the played move minus the learned value after the policy's best move, taken in the mover's perspective; 0 is optimal and more-negative is worse.
- :Player-skill-level: is the exponentially-weighted MEAN of the human's :Move-regret: samples — a single moving estimate of how well the player is currently playing. (Variance/standard-deviation is deliberately not tracked: with one human there is exactly one skill to estimate, and a mean plus an additive offset is the whole signal.)
- :Strength-temperature: is the temperature applied when the computer picks its root move — 0 means argmax (strongest), higher means flatter sampling over visit counts (weaker); it is the single lever difficulty tuning moves. It is NOT the annealing :Prior-bias-temperature:, which softens prior move categories during expansion.
- :Difficulty-controller: is the per-game `entity` that observes both players' :Move-regret:, holds the :Player-skill-band: and the current :Strength-temperature:, and drives the opponent toward a target strength; it carries identity and mutable state across the moves of one game.

***implementation reqs***

- Constant: STRENGTH_TEMP_MIN / STRENGTH_TEMP_MAX — bounds of :Strength-temperature:; at or below MIN the computer plays argmax.
- Constant: DIFFICULTY_OFFSET — additive regret bias on the setpoint; positive makes the opponent play slightly better than the player (harder), negative gives a handicap. Replaces the earlier sigma-based offset.
- Constant: PLAYER_EMA_ALPHA — smoothing of the :Player-skill-level:.
- Constant: DIFFICULTY_GAIN — proportional gain of the temperature controller.
- Constant: USE_DYNAMIC_DIFFICULTY — feature flag, off by default; when off, root selection is plain argmax and no scoring runs.
- Regret reuses `get_q_value` (learned value) and one policy search for the best move; the computer's own search yields its best move for free, so only the human move costs an extra search.

***test reqs***

- A midgame position where the policy's best move and a clearly weaker legal move have distinct values, to assert regret ordering and temperature monotonicity.

***functional specs***

- Where :Difficulty-controller: is enabled, the computer must pick its root move using :Strength-temperature: rather than argmax.
  - Given :Strength-temperature: at or below STRENGTH_TEMP_MIN, When the root move is chosen, Then it is the argmax (identical to today).
  - Given a higher :Strength-temperature:, When the root move is chosen, Then weaker moves gain probability and mean :Move-regret: worsens. (Assert: expected regret is monotonically non-increasing in strength as temperature falls.)
- The :Difficulty-controller: must update the :Player-skill-band: from each human :Move-regret: and drive the opponent toward a setpoint.

  Input: player_move — the human's chosen move, and the position before it
  Parameters: offset ∈ ℝ (additive regret bias), gain ∈ ℝ⁺
  Initialize: player_mean ← seed_mean, temperature ← STRENGTH_TEMP_MAX/2   # per game; seed optional

  Loop over each move pair in the game:
      When it is the human's turn:
          r ← :Move-regret: of player_move
          player_mean ← EMA(player_mean, r)
      Otherwise When it is the computer's turn:
          setpoint ← player_mean + offset                              # transient
          pick the root move at the current temperature
          r_ai ← :Move-regret: of the computer's played move
          temperature ← clamp(temperature + gain · (r_ai − setpoint), STRENGTH_TEMP_MIN, STRENGTH_TEMP_MAX)
      log r or r_ai together with the raw mover-value for validation
  Assert: temperature stays within [STRENGTH_TEMP_MIN, STRENGTH_TEMP_MAX] every move.

  Given a stream of human moves at a steady skill, When several computer moves follow, Then the computer's mean :Move-regret: should converge near player_mean + offset.
  Given offset > 0, Then the computer should end slightly stronger than the player; Given offset < 0, Then weaker.

- The raw mover-value must be logged beside :Move-regret: each move so the regret signal can be validated without a second control path.
- :Difficulty-controller: state resets at game start but may be warm-started from seed_mean; cross-session persistence is out of scope here.


================================================================================
FILE: spec\elo-calibration.spec.md
================================================================================

---
description: 'The temperature -> absolute-Elo dial: calibrate one trained net to any target strength against the anchor'
import:
  - elo-measurement
  - dynamic-difficulty
---

***definitions***

- :Temperature-elo-curve: is the persisted, monotone mapping from a policy's :Strength-temperature: to its absolute :Measured-elo: against the :Elo-anchor:. It is built per net checkpoint by measuring the policy at a grid of temperatures (argmax at the low end, flatter sampling at the high end) and fitting a monotone-decreasing curve — higher temperature = weaker = lower Elo.
- :Absolute-strength-dial: is the inverse lookup: given a target absolute Elo inside the curve's measured range, it returns the :Strength-temperature: whose measured Elo is closest (interpolated). This lets ONE trained net serve a chosen strength anywhere on the human rating curve (e.g. play a 1400 opponent) — an ABSOLUTE calibration, distinct from :Difficulty-controller:'s RELATIVE setpoint (player mean-regret + offset).
- :Chained-self-anchoring: is how the WEAK end of the curve is placed. A temperature that plays ~900 scores ≈0 against SF@1320 (the anchor's floor) — a shutout bound, not a point (per elo-measurement). So the directly-measurable range bottoms out well above the human range we want to serve. Fix: place mid-strength temperatures directly against the :Elo-anchor:, then place weaker temperatures by matches against the ALREADY-PLACED mid temperatures, chaining Elo downward off self-play rather than off the anchor.
- :Approximate-elo-curve: is a heuristic default :Temperature-elo-curve: present the instant a net is loaded, BEFORE any measurement — a smooth monotone placeholder (temperature → Elo gap below the anchor) that carries an `approximate` flag. It exists so the :Absolute-strength-dial: and the :Estimated-elo-readout: work with ZERO warm-up: a human can start an adjusting or fixed-strength game immediately and see estimated Elo. It is never claimed as measured — every surface that shows it labels it approximate — and it is replaced by the measured curve when (and only when) calibration runs (per the run-contract req below). Calibration is a distinct step from model TRAINING: training produces the net; calibration measures the temperature→Elo map of that net. Neither is a prerequisite for the other, and neither is a prerequisite for playing.
- :Estimated-elo-readout: is the per-turn surfacing of two numbers while a :Difficulty-controller: opponent (auto or fixed) is active: the opponent's Elo from its current :Strength-temperature:, and the human's Elo from their :Player-skill-level: regret (via `player_elo`). Both are read off whichever curve is active and are labeled approximate vs measured accordingly.

***implementation reqs***

- `elo_calibration.py` owns building, persisting, and inverting the curve; this file is its owning spec (previously absent). It reuses `measure_sf` / the elo-measurement machinery to place each grid point.
- Constant: TEMPERATURE_GRID, CALIBRATION_GAMES_PER_POINT — the temperature grid and games measured per point.
- The curve is persisted per checkpoint (e.g. `models/temp_elo.json`); it is re-measured ONLY on gate-clear checkpoints, not every run — a grid of ~8 temperatures × ~20 games does not fit the 5-minute run contract.
- The calibrator MUST expose an `approximate` flag and construct an :Approximate-elo-curve: as its default table, so a curve is always present without measurement. `calibrate()` clears the flag once it has measured the grid; loading a persisted curve restores whichever flag was saved.
- No implicit calibration: game start and difficulty setup MUST NOT trigger a measurement ("warm-up"). Running the real calibration is an explicit, opt-in action (a settings action), never a prompt gating normal play.

***test reqs***

- A grid with noisy raw measurements, to assert the persisted curve is projected monotone-decreasing before use.

***functional specs***

- :Temperature-elo-curve: must be monotone-decreasing in temperature.
  - Given temperatures t1 < t2, Then curve(t1) >= curve(t2); raw measurement noise is projected to the nearest monotone curve before persisting.
- Each point must be absolute Elo, not relative.
  - Given a grid temperature, When its point is measured, Then it is :Measured-elo: against the :Elo-anchor: (absolute), never a regret offset against a human.
- :Absolute-strength-dial: must invert within range and clamp (flagged) outside it.
  - Given a target Elo within the curve's range, Then the dial returns the interpolated :Strength-temperature:.
  - Given a target outside the range, Then it clamps to the nearest endpoint and flags the request as out-of-range (a shutout point is a bound, per elo-measurement).
- The weak end must be placed by :Chained-self-anchoring:, not against the anchor directly.
  - Given a temperature too weak to score against SF@1320 (a shutout bound), When it is placed, Then its Elo is derived from a match against an already-placed mid-strength temperature, chaining downward off self-play.
  - Given the curve extends below the anchor floor, Then it can serve human-range targets (e.g. 900) that direct anchor measurement cannot reach.
- Re-measurement must respect the run contract.
  - Given an ordinary ≤5-minute run, Then the curve is NOT re-measured; Given a gate-clear checkpoint, Then the full temperature grid is re-measured and persisted.
- The absolute dial must compose with, not replace, the relative controller.
  - Given the :Absolute-strength-dial: sets a baseline operating strength and the :Difficulty-controller: then tracks the seated human, Then the absolute curve fixes the operating point and the relative controller adjusts around it. (This is the temperature->Elo proxy the 1200 thread required.)
- The dial must be usable with no warm-up via the :Approximate-elo-curve:.
  - Given a net is loaded but never calibrated, When a human starts an adjusting or fixed-strength game, Then an :Approximate-elo-curve: is already present and the opponent's strength selection and the :Estimated-elo-readout: work immediately.
  - Given the active curve is approximate, When any Elo is shown, Then it is labeled approximate (never presented as measured).
  - Given the human declines or never runs calibration, Then play and the :Estimated-elo-readout: still function for the whole game (the warm-up is optional refinement, not a prerequisite).
- The :Estimated-elo-readout: must be surfaced each turn while an adjusting/fixed opponent is active.
  - Given a :Difficulty-controller: opponent (auto or fixed) is enabled, When a board is shown, Then the opponent's Elo (from :Strength-temperature:) and the human's Elo (from :Player-skill-level:) are displayed together, each labeled approximate or measured per the active curve.
  - Given the opponent is at full strength (:Difficulty-controller: disabled), Then no per-turn readout is required (there is no relative skill signal being tracked).


================================================================================
FILE: spec\rl-categorization.spec.md
================================================================================

---
description: 'What kind of RL this system is — qualified classification across the standard axes, per training stage'
import:
  - learned-model
  - search-mcts
  - annealing-schedule
  - teacher-distillation
  - value-target
  - self-play-leela
---

***definitions***

- :Headline-system: is the chess agent as a whole — a learned dual-head network, PUCT/MCTS planning with the known game rules, and an annealed handoff from teacher priors to self-play.
- :Value-critic: is the learned value head V(s). No head is trained by policy gradient at any stage, so the system is never actor-critic in the A2C/PPO sense.
- :Known-model-planning: is search with the exact game rules at decision time (the "model" is the rulebook, not learned).
- :Supervised-warm-start: is the Stage-1 bootstrap (`distill_sf.py`, spec: teacher-distillation) that regresses V(s) onto a Stockfish :Teacher:'s score and the :Policy-head: onto MultiPV soft targets over generated positions, *before* any self-play. It is **supervised value/policy regression / knowledge distillation**, not RL: a fixed target oracle, MSE/cross-entropy losses, no rewards, no bootstrapping off the net's own estimate, no environment interaction during the fit.

***implementation reqs***

- Every classification below MUST be stated *with its qualification*; the bare label is misleading and is the reason this file exists.

***functional specs***

- The :Headline-system: should be described stage-wise; the closest one-line summary is **"AlphaZero/Leela-lite bootstrapped by Stockfish distillation, with every handoff annealed and Elo-gated"** — historically labeled "Deep Q-Learning", which it is not (no `max_a Q(s,a)`, no per-action value head).
- Training is **three staged phases**, and they are different learning paradigms — do not conflate them:
  - **Stage 1 — :Supervised-warm-start: (teacher-distillation).** OFFLINE, off-policy (positions from the teacher's own strong-game trajectories — the :Position-source: — sampled via temperature over its MultiPV, NOT random playouts), **supervised regression / distillation**. Not RL, not TD, not DQN/SARSA/actor-critic — no reward, no return, no bootstrap. Its only job is to make both heads competent fast (self-play from scratch cannot cross ~1200 in minutes).
  - **Stage 2 — annealed off-policy refinement (`chess_ai.py`).** ONLINE, off-policy, value-based **search-bootstrapped λ-return** (:Lambda-return:, β-annealed via :Bootstrap-share:) with replay + target net + search planning + annealed shaped reward; teacher data mixed per :Demo-share:. Off-policy-safe by **tree-backup** (the bootstrap is the search's improved on-policy value), not by importance sampling. TD(0) is the β=1 special case (`neural_network.py:274`); pure Monte-Carlo is the β→floor limit.
  - **Stage 3 — expert iteration (self-play-leela).** RL proper: **policy iteration with search as the improvement operator**. Policy trained by **MCTS visit distillation** (cross-entropy), value by the same :Lambda-return: with β at its floor (approaching Monte-Carlo outcome regression, light search bootstrap retained).
- **Explicit yes/no classification** (the axes to answer directly):

  | Axis | Stage 1 (distillation) | Stage 2 (refinement) | Stage 3 (expert iteration) |
  |---|---|---|---|
  | Reinforcement learning at all? | **No** — supervised | **Yes** | **Yes** — policy iteration via search |
  | Online / offline | **Offline** (fixed generated set) | **Online** (self-generated) | **Online** (self-play) |
  | On-policy / off-policy | Off-policy (n/a target) | **Off-policy**; bootstrap tree-backup-safe, λ-tail recency-guarded (:Replay-window:) | same; Retrace(λ) behind flag for data staler than :Replay-window: |
  | Value / policy / actor-critic | Value+policy regression | **Value-based** (V(s) critic) | Policy+value, policy via **visit distillation** |
  | Actor-critic / A2C / advantage? | **No** | **No** — no policy grad, no advantage | **No** — expert iteration, not policy gradient |
  | SARSA? | **No** | **No** — not on-policy action-value | **No** |
  | Q-learning (action-value, max_a)? | **No** | **No** — state-value V(s), no `max_a Q` | **No** |
  | DQN? | **No** | **DQN-*family* tricks only** (replay, target net) | **No** — no TD target at all in pure form |
  | Model-based? | No (static labels) | Model-free learning **+** :Known-model-planning: | Model-free learning **+** :Known-model-planning: |
  | Bootstrapping? | **No** (direct label) | **Yes** — :Lambda-return: bootstrapped from the search value (tree-backup); TD(0) is the β=1 limit | β at floor ⇒ approaches Monte-Carlo outcome z; light search bootstrap retained |
  | Reward | none (regression) | dense shaped, annealed → sparse terminal ±1 | sparse terminal z; Dirichlet noise is exploration, not reward |
  | Closest lineage | knowledge distillation | TD-Gammon / AlphaZero value learning | **AlphaZero / Leela Zero** |

- Named-algorithm mapping: the system is at no stage SARSA, PPO/A2C, or literal action-value Q-learning. The one true action-value DQN in the repo family is the separate Connect4 `value_based/dqn_connect4.py`.


================================================================================
FILE: spec\terminal-interface.spec.md
================================================================================

---
description: 'The terminal human-vs-computer front-end: move entry, in-game commands, and the per-turn board readout'
import:
  - dynamic-difficulty
  - elo-calibration
---

***definitions***

- :Move-entry: is how the human submits a move at the prompt: a UCI string of 2 characters (a from-square, e.g. `e2`, which selects a piece and previews its legal moves) or 4 characters (a complete move, e.g. `e2e4`). Length disambiguates entry from a :Game-command: — every command token is a single letter or a full word, never a 2- or 4-character square string, so the two input classes never collide.
- :Game-command: is a non-move action typed at the same prompt. Each command has a canonical full word and a first-letter shortcut, and BOTH are accepted equivalently: `(h)int`, `(u)ndo`, `(s)ave`, `(l)oad`, `(r)esign`, `(c)ancel`. The shortcut is the word's first letter; the set is collision-free (distinct first letters) and disjoint from :Move-entry: (single letters are never legal squares).
- :Board-readout: is the header printed above the board every time it is rendered: the side to move, the position eval from the fast evaluator (the advantage/disadvantage number), the :Player-score-breakdown:, and — when a :Difficulty-controller: opponent is active — the :Estimated-elo-readout: (opponent + player Elo, labeled approximate or measured).
- :Player-score-breakdown: is a per-side table in the header showing each player's Pieces score (summed material value) and Position score (weighted positional terms: mobility, square control, king safety, pawn structure, space, coordination), plus their sum. It uses the same component weights as the fast evaluator, so White's total minus Black's total equals the position eval (excluding the turn-dependent check bonus). It gives the human a side-by-side view of where the advantage comes from, decomposing the single eval number into material vs. position for each color.
- :Side-selection: is the new-game setup prompt where the human picks their color before play begins (`Play as white or black? (w/b, default: w)`). White is the DEFAULT: empty input, or any input that does not explicitly request black, yields White; Black is chosen only by an explicit black token (a value starting with `b`). This holds uniformly across every entry path (fresh game, load-game, play-from-FEN).

***implementation reqs***

- `terminal_board.py` owns the terminal front-end: `process_input` parses :Move-entry: and :Game-command:; `display_board` renders the :Board-readout:.
- Command dispatch MUST match each :Game-command: against its full word OR its first-letter shortcut (e.g. `command in ('hint', 'h')`), so the on-screen legend and the parser stay in sync.
- The command legend printed to the human MUST show the shortcut form (e.g. `(h)int`) so the available shortcuts are discoverable without documentation.
- `menu.py` owns :Side-selection:. Every prompt that maps a color choice to `human_color` MUST resolve to `chess.WHITE` unless the input explicitly requests black — i.e. select `chess.BLACK` only on an explicit black token and default to `chess.WHITE` for empty or unrecognized input. No path may fall through to Black.
- The :Estimated-elo-readout: MUST be rendered inside `display_board`'s header — once per turn, on the persistent board view — NOT as a transient line that scrolls away after a move.
- The :Player-score-breakdown: MUST be computed by `evaluate_by_player` in `evaluation.py` and rendered inside `display_board`'s header directly below the position eval. `evaluate_by_player` MUST reuse the fast evaluator's material values and component weights so that `white['total'] - black['total']` reconstructs `fast_evaluate_position` (up to the check bonus), keeping the breakdown and the headline eval consistent.

***test reqs***

- A dispatch table asserting that each full word and each single-letter shortcut route to the same action, and that a 2- and 4-character UCI string is parsed as :Move-entry:, not a command.
- An invariant test over several positions (startpos, a mid-game position) asserting `evaluate_by_player(board)['white']['total'] - ...['black']['total']` equals `fast_evaluate_position(board)` for positions not in check, and that each side's `material + position == total`.

***functional specs***

- Every :Game-command: must accept its full word and its first-letter shortcut identically.
  - Given the input `hint` or `h`, Then a hint is shown; likewise `u`=undo, `s`=save, `l`=load, `r`=resign, `c`=cancel.
  - Given a 2- or 4-character legal UCI string, Then it is treated as :Move-entry:, never as a command (no shortcut shadows a square).
- :Side-selection: defaults to White.
  - Given the :Side-selection: prompt, When the human presses Enter with no input, Then they play White.
  - Given any input that does not explicitly request black (e.g. `w`, `white`, a stray token), Then they play White.
  - Given an explicit black token (a value starting with `b`, e.g. `b` or `black`), Then they play Black.
  - This behavior is identical across the fresh-game, load-game, and play-from-FEN entry paths.
- The :Board-readout: must show the :Player-score-breakdown: every turn.
  - Given the board is rendered, Then the header shows, for both White and Black, a Pieces score, a Position score, and their total.
  - Given a position, When the breakdown is computed, Then `white['total'] - black['total']` equals `fast_evaluate_position` for that position (excluding the turn-dependent check bonus), so the per-player totals stay consistent with the single eval number.
- The :Board-readout: must show the estimated Elo every turn while an adjusting/fixed opponent is active.
  - Given a :Difficulty-controller: opponent (auto or fixed) is enabled, When the board is rendered, Then the header shows the :Estimated-elo-readout: alongside the position eval.
  - Given the opponent is at full strength (:Difficulty-controller: disabled), Then only the position eval is shown and no Elo readout is required.
  - Given the active curve is approximate, When the readout is shown, Then it is labeled approximate (never presented as measured).
