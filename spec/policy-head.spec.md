# Merge 12 — Policy head: the proposer-evaluator loop in move space (expert iteration)

Why (2026-07-11 verdict chain): the 809-dim linear VALUE cannot store move-ranking knowledge
for materially-equal positions — proven three ways (donor-arm purist ceiling ~982; zero
population frozen at 11.17 and material at 13.94 with exploration maxed and the jitter
surrogate's gradient ~0). The missing object is an explicit POLICY: a second head that
memorizes *which moves the search chose*, i.e. storage for exactly the knowledge the value
head drops. Operator's frame: the same proposer→evaluator→selection pattern as the jitter
surrogate, one level down — "model of the model," in move space.

Grounding: Expert Iteration (Anthony/Tian/Barber), Deep Pepper (arXiv 1806.00683), AlphaZero
policy-head design — all council-validated (round #1). Compliance: the policy trains on OUR
search's choices over OUR games — self-distillation; no external labels, ever. NOT the old
actor-critic arm: that was policy-GRADIENT from noisy outcomes (died at the sharpness wall);
this is SUPERVISED learning on search decisions — a different, stable target.

## :Policy-head: (architecture)

- π(m | s) ∝ exp(θ · x(afterstate(s, m))) — a second linear head over the SAME 809 features,
  evaluated per legal move on its afterstate. Zero new feature machinery; whitened space like
  the value head (ZCA applies).
- The purist artifact this project has been missing: **raw policy = argmax_m θ·x(afterstate)**
  — a no-search player whose strength is a direct read of stored move knowledge.

## :ExIt-targets: (training signal)

- Every generation decision where the search's best move was PLAYED (on-policy records) is a
  supervised example: cross-entropy of π against the chosen move over the legal-move set.
- Records must carry the ROOT position (fen) + chosen move (uci). PARGEN record v2:
  (root_fen, chosen_uci, leaf_fen, white_value, predicted) — rsearch `play_games` addition;
  the python serial path already has root/choice in hand.
- Training batch: for each (root, chosen), compute afterstate features for ALL legal moves
  (bounded: mean ~30, reuse encode_features; SGD on CE). Off-best softmax samples and ε-moves
  are EXCLUDED (they are exploration, not expert choices).
- θ updates piggyback the existing cycle (same optimizer step cadence as the value head,
  separate learning rate QLEARN_ALPHA_PI, Optuna-tunable under the ≤30-min cap).

## :Policy-behavior: (closing the loop)

- Generation move choice becomes policy-guided where it pays: softmax over
  (θ-logits + σ·v1/τ) — the policy proposes, the value glances, temperature anneals as today.
  Flag `QLEARN_PI_BEHAVIOR` (0 = value-only, current behavior; part of study/arm identity).
- Search integration (later, measured separately): π as move-ORDERING for rsearch (better
  ordering = more cutoffs = cheaper depth — composes with Merge 10's efficiency thesis).

## Measurement scales (extends the canon's three)

- Raw policy (no search, argmax π) — the NEW purist metric of record.
- Raw value (1-ply argmax V) — kept for continuity with 982/490 history.
- d2-greedy and deep-inference rungs — unchanged.

## Pre-registered verdicts

1. The policy head trains without destabilizing the value head (crowns still climb on the
   canon recipe donor-seeded smoke arm).
2. WIN condition (operator's bar): raw-policy rung ≥ 1000 on the ladder scale from a
   from-scratch population (zero or material lineage) — the storage hypothesis confirmed.
3. Falsification: if raw-policy plateaus at the same wall (~950-1000) despite CE loss
   converging, the wall is deeper than storage (capacity of LINEAR anything) → the
   hidden-layer arm (nnue-eval revival) is next, with this result as its justification.

## Acceptance

1. py_compile + rsearch parity battery green; PARGEN v2 records verified (root fen matches
   chosen move's legality; excluded samples absent).
2. CE loss decreases on a 200-game smoke; raw-policy rung runs end-to-end on the ladder.
3. Single-variable arm: canon recipe + policy head vs canon recipe, same seed, same games —
   compare raw-policy vs raw-value rungs at matched game counts.
