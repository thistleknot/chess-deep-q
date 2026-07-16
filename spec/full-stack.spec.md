# Merge 16 — The full stack: every organ ON (operator: "combine all this")

Supersedes Merge 15 (archived as an arm; ALL mechanics live in qlearn.py and are included
here). One regime, every validated organ enabled simultaneously — the most complete agent
the linear net can carry:

| Organ | Mechanism | Knob |
|---|---|---|
| Model | the RULES (perfect, given) via rsearch d2 planning | QLEARN_RSEARCH_DEPTH=2 |
| Critic | trivium value targets (λ-return : search : outcome, tuned anneal) | QLEARN_TRIVIUM* |
| Score | Elo-surprise s − E[s\|ratings] (BT vs declared rung ratings, SF@1320 anchor, online agent rating) | QLEARN_SURPRISE=1 |
| Advantage | GRPO group whiten per chunk (RL_v2 `_whiten_group`, zero-var ⇒ zeros) | QLEARN_GRPO=1 |
| Admixture | replay-temperature resample ∝ exp(adv/T), stratified W/D/L floor | QLEARN_REPLAY_T (Optuna dim) |
| Replay | :Magic-deck: — persistent advantage-ranked buffer, decaying priorities ("reshuffle"), DECK_MIX draws per chunk | QLEARN_DECK=256, MIX=0.5, DECAY=0.9 (declared) |
| Opponents | graded ladder + matchmaking (draws vs stronger rungs pay positive) | QLEARN_OPP=graded |
| Actor | ABSENT by design (no linear-capacity precedent; returns with capacity) | — |

Provenance (:Provenance: law): clean random seed + self-ZCA (own games) + declared constants
(rung ratings, K=32, deck knobs) + rules/RNG. PASSES.

## Study (|surprise|grpo|replay-temp|deck|)

3 trials, s100 × e2, trial 0 = proven parms + replay_t 1.0; deck knobs fixed declared
(never searched — infrastructure); replay_t is the searched manifold knob. 30-min cap.
Context baseline: the M15 (no-deck) study's trial 0 posted 892 — the strongest clean-regime
trial objective to date (vs 822/788/756 prior probes).

## Pre-registered

1. Study best ≥ 892 (deck should not hurt at trial scale; if it does, deck reverts to 0 for
   the final run — logged, not silently kept).
2. Final run (OPERATOR-launched, console/UI-visible): success = d2-scale crowns exceeding
   the pure-lineage ceiling family (~8.8–13.9 across all prior from-scratch arms); the
   decisive-rate metric must RISE (the anti-shuffling claim of surprise scoring).
3. Falsified ⇒ organs peel back one at a time (deck → replay-T → GRPO → surprise), each
   peel a logged single-variable step; the trivium core survives regardless.
