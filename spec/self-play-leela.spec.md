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
- :Self-play-bootstrap: is the leaf evaluator the self-play search STARTS from, along the :Prior-lineage:. Two valid entry points: (a) the HAND HEURISTIC via an eps-blend leaf — board_eval = eps·tanh(pst/…) + (1−eps)·:Learned-value:, eps annealing 1→0 so the heuristic guides early self-play and the net takes over (pure prior-lineage, NO teacher in the loop); (b) a DISTILLED net snapshot (eps=0 from a checkpoint, e.g. the Stockfish-distilled tower — warmer start). SF is never in the self-play loop under either; it is only the :Ladder: gate. The two bootstraps are compared as a :Ladder:-measured experiment — each one's per-iteration ladder curve — to see whether unaided heuristic self-play catches the distilled start (and whether either catches SF). FIRST MEASURED RESULT (`benchmark.py`, 8 iters, net-minimax d1, n=12): NEITHER bootstrap climbed toward the search reference — A (alpha-beta pst @0.3s) = 1.00 vs heuristic-1ply while B1/B2 stayed flat at 0.0–0.5; value loss fell (net fits its own outcomes) but external strength did not rise. Diagnosis: the d1 improvement operator is too shallow to AMPLIFY (expert iteration climbs only when the search is stronger than the raw policy), and self-play from a weak net feeds itself weak-outcome data. Implication: on this hardware the net's value is better spent as a hole-free leaf eval INSIDE deeper search (:Search-profile: at higher depth / the NNUE synthesis) than as a standalone self-play player; useful self-play needs deeper per-move search, which is throughput-bound.
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
