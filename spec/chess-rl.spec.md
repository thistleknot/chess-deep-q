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
- :Measured-disposition: (this iteration) is the honest state of the RL goal, recorded so the spec cannot drift back into unearned optimism: FOUR RL configs have plateaued at "beats random, loses to heuristic-1ply" (distillation → ~1040, self-play @d1 → flat, off-policy bootstrap → weak, A2C+beam @400ep → 0/20), and NNUE v1 is FALSIFIED. The strong player in the repo — the ~1672 alpha-beta engine (`engine.py` pst) — is CLASSICAL SEARCH, NOT RL: a **baseline to SURPASS, offered to the user as an optional opponent, never a substitute for the learned agent.** The repo's premise is **learning to apply RL to chess EFFICIENTLY** — strength is only the yardstick; RL is the point. Any "ship the strong engine" framing is a category error against that premise and MUST NOT re-enter the spec. The diagnosis (see `self-play-leela.spec.md`): every attempt used a WEAK improvement operator. The remaining, never-run RL move is the batched-PUCT :Amplification-experiment: — a strong operator with a usable (batched) net throughput. Its per-iteration :Ladder: curve is the definitive verdict: rising ⇒ RL viable here; flat ⇒ shallow-net RL is walled on this hardware, proven not conjectured.
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
