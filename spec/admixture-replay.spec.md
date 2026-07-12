# Merge 15 — Admixture replay: GRPO + Elo surprise + temperature-controlled elite sampling

Supersedes Merge 14 (archived UNTESTED as an arm; its MECHANICS are included here and stay
in qlearn.py). Operator design, in their words: "make admixtures of better batches (class
balanced), mix and match, use temperature to control the sampling of the types of games we
generate to push the envelope higher."

## The loop (why the envelope pushes itself)

Generate (behavior τ controls game diversity) → score every game by **Elo surprise**
`s − E[s | ratings]` (:Rating-surprise:, Merge 14 mechanics: Bradley-Terry vs declared rung
ratings, SF@1320 anchor, agent rating online, draws vs stronger rungs pay positive) →
**GRPO group-whiten** the chunk (:GRPO-group:, RL_v2 `_whiten_group` semantics, zero-variance
⇒ zeros) → **resample the chunk ∝ exp(advantage / T_replay)** (:Replay-temperature:) with a
**stratified floor** (≥1 game of each outcome class survives — class balance) → train.
The baseline is relative and moving (group mean + rating expectation), so "above average"
climbs with the policy: the ratchet needs no absolute target.

## Knobs

- `QLEARN_SURPRISE=1` — Elo-surprise outcome term (graded games; mirror stays raw).
- `QLEARN_GRPO=1` — group-whitened advantage replaces z_out.
- `QLEARN_REPLAY_T` — the admixture dial: low = CEM/self-imitation (elite-only end),
  high = uniform; 0 = off. **Optuna-searched** (`QLEARN_REPLAY_TUNE=1`, dim
  replay_t [0.2, 3] log, prior 1.0) — new manifold knob, never hand-picked.
- Faithful-mode note: online per-game SGD means resampled elite games take proportionally
  more update steps — the intended admixture effect under KC_FAITHFUL.

## Provenance (:Provenance: law)

Clean stack: clean random seed + self-ZCA (own games) + declared rung-rating constants +
rules/RNG. PASSES. Rung ratings and SURPRISE_K are declared constants in qlearn.py.

## Pre-registered

1. Smoke: chunk resample preserves size; every outcome class present post-resample; runs
   clean on graded ladder. (Merge 14 mechanics smoke: PASSED 2026-07-12, 24 games,
   finite loss, metrics row.)
2. Study (|surprise|grpo|replay-temp|): trial 0 = proven parms + T=1.0; best-of-3 reported;
   operator launches the final run.
3. Arm verdict at matched games vs the raw-z graded control: surprise+GRPO+admixture curve
   ≥ control, and draw-shuffling rate falls (decisive-rate metric rises). Falsified ⇒
   revert to raw z; the trivium survives either way.

## Held (re-proposable with evidence)

A-HPO hysteresis (arXiv 2605.30201) — operator: nothing fancier than GRPO+Elo. WDL
distributional head — natural Merge 16 if this arm shows life (linear 3-class head, draws
informative by construction, E[s] falls out of the distribution).
