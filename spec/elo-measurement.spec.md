---
description: 'Measured strength against a real Stockfish anchor — the honest progress signal and gate authority'
---

***definitions***

- :Elo-anchor: is a real Stockfish binary (in `engines/`) pinned to a fixed strength via UCI_LimitStrength/UCI_Elo; it is the only externally calibrated reference, so every strength claim is stated relative to it. Stockfish enforces a hard floor of UCI_Elo = 1320 — strength below the floor is inferred by the logistic relation, never by configuring a weaker anchor.
- :Measured-elo: is the Elo estimate produced by an alternating-color match against the :Elo-anchor:, with ply-capped games adjudicated by Stockfish evaluation, and the match score converted through the logistic relation (`measure_sf.py`).
- :Elo-gate: is a strength threshold (1200, then 1600, then teacher-strength) that must be cleared by :Measured-elo: before annealing progress or a training-stage transition may advance past it.
- :Measurement-game: is a game played purely for measurement: every training-time noise source (root Dirichlet noise, move temperature, shaped reward) is off and the agent plays argmax.
- :Measurement-power: is the requirement that GAMES_PER_MEASUREMENT be large enough that the 95% confidence interval on :Measured-elo: is NARROWER than the :Elo-gate: step it must resolve. The per-game score SD is √(p(1−p)/n); at n=6 it is ~0.13 — wider than the gap between adjacent strength levels — so gate decisions at n=6 are coin flips and any single-lever conclusion drawn from them is unfounded. Adequate power is ~30–50 games at a fast time control (games are cheap relative to a labeling grind).
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
