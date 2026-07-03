---
description: 'Measured strength against a real Stockfish anchor — the honest progress signal and gate authority'
---

***definitions***

- :Elo-anchor: is a real Stockfish binary (in `engines/`) pinned to a fixed strength via UCI_LimitStrength/UCI_Elo; it is the only externally calibrated reference, so every strength claim is stated relative to it. Stockfish enforces a hard floor of UCI_Elo = 1320 — strength below the floor is inferred by the logistic relation, never by configuring a weaker anchor.
- :Measured-elo: is the Elo estimate produced by an alternating-color match against the :Elo-anchor:, with ply-capped games adjudicated by Stockfish evaluation, and the match score converted through the logistic relation (`measure_sf.py`).
- :Elo-gate: is a strength threshold (1200, then 1600, then teacher-strength) that must be cleared by :Measured-elo: before annealing progress or a training-stage transition may advance past it.
- :Measurement-game: is a game played purely for measurement: every training-time noise source (root Dirichlet noise, move temperature, shaped reward) is off and the agent plays argmax.
- :Measurement-power: is the requirement that GAMES_PER_MEASUREMENT be large enough that the 95% confidence interval on :Measured-elo: is NARROWER than the :Elo-gate: step it must resolve. The per-game score SD is √(p(1−p)/n); at n=6 it is ~0.13 — wider than the gap between adjacent strength levels — so gate decisions at n=6 are coin flips and any single-lever conclusion drawn from them is unfounded. Adequate power is ~30–50 games at a fast time control (games are cheap relative to a labeling grind).

***implementation reqs***

- `measure_sf.py` is the sole measurement authority; self-reported training signals (loss curves, validation MSE, sign accuracy) never gate anything.
- Constant: ANCHOR_ELO, ELO_GATES, GAMES_PER_MEASUREMENT, and the adjudication centipawn threshold — developer-tuned measurement rules.
- The Stockfish binary lives under `engines/` and is discovered by glob, not PATH.

***test reqs***

- The harness must reproduce a known-strong engine's superiority over the anchor: `engine.py` (measured ~1720; beat SF@1320 4-0 at 0.3s/move) must score decisively above 0.5 vs SF@1320.

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
