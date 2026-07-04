---
description: 'Linear afterstate-value RL — TD-Gammon-style linear function approximation over hand features, trained by gradient descent on self-play outcomes, multiprocessing self-play, deep alpha-beta — the µs-cheap, provably-stable, efficient RL path'
import:
  - prior-evaluator
  - value-target
  - rl-categorization
  - elo-measurement
---

***definitions***

- :Linear-value: is V(s) = wᵀφ(s) — TRUE linear in the weights (NO squash), φ = the hand feature vector (`gbdt_features.features`: material + PST + castling, White-absolute), w the LEARNED weights. Linearity is load-bearing: Sutton & Barto §9.4 proves the linear case has a SINGLE global optimum, so any convergent method reaches it — wrapping v̂ in a tanh/sigmoid would make it NONLINEAR in w and FORFEIT exactly that guarantee (a subtle bug corrected: the value stays linear; bound the *target*, not the model, if bounding is wanted). The hand heuristic `pst` is itself a FIXED wᵀφ (hand-set weights); :Linear-value: LEARNS w by gradient descent — RL-tuning the heuristic's own weights. µs-cheap dot product, CPU, plugs into `AlphaBetaEngine(eval_fn=...)` unchanged.
- :Afterstate-target: is the RL signal: Q(s,a) = wᵀφ(s·a) (:Afterstate-action-value:), and w is regressed toward the self-play OUTCOME z / :Lambda-return: (White-absolute), SWAPPING the supervised SF-cp label (`train_linear`/`train_gbdt`) for the committed outcome. Update = the linear SGD form (S&B eq. 9.5): w ← w + α·(target − wᵀφ(s))·φ(s) — gradient Monte-Carlo with target z; **semi-gradient TD(λ)** (S&B eq. 9.9, bootstrapped target) is the documented extension. This is exactly TD-Gammon: linear FA + TD(λ) on self-play outcomes. Honest caveat (S&B §9.2): minimizing the value error VE is a PROXY — "the best value function [for finding a better policy] is not necessarily the best for minimizing VE" — so the :Linear-kill-check: gates on PLAY (beat pst), not on fit.
- :Feature-set-design: is the principle for choosing φ, since a linear model's ceiling IS its features (S&B §9.5). The eval and the search DIVIDE LABOR: features must encode what SEARCH cannot cheaply see. INCLUDE — search-COMPLEMENTARY, slow-horizon, static structure: material + PST, mobility, king safety (attackers / castled / center), pawn structure (doubled / isolated / PASSED), space, coordination, bishop pair. EXCLUDE — search-REDUNDANT tactics: threatened / hanging pieces, immediate captures, checks, the check-bonus — the d2 search already computes these (a threat IS the opponent's reply), so hand-coding them pays twice. ADD — §9.5.1 INTERACTION products for nonlinearity while staying LINEAR IN THE WEIGHTS (single-optimum preserved): king_exposure × enemy_queen, bishop_pair × openness, passed_pawn × endgame_phase. The first 29 dims MUST equal `gbdt_features.features` so `linear.npz` PARTIALLY warm-starts (pst-baseline start, new features learned from zero). Implemented in `rich_features.py`; `gbdt_features.py` stays untouched (its other consumers).
- :Parallel-selfplay: is multiprocessing self-play (:Process-separated-labeling: generalized): N worker OS processes each play alpha-beta(:Linear-value:, depth D) games on CPU cores — NO GPU, so near-linear core speedup with zero contention (the µs eval is WHY this is minutes, not hours; the resnet+PUCT path was GPU-bound and GIL-serialized). Games open with random plies for coverage; workers return (φ, outcome) trajectories to the trainer, which does the GD update and broadcasts the new w.
- :Linear-kill-check: is the ship gate: the RL-tuned :Linear-value:, in alpha-beta at depth D, MUST beat the hand-`pst` baseline at EQUAL depth on the :Ladder: (n ≥ 30, CIs). Honest scope — a LINEAR model over material+PST CANNOT represent nonlinear interactions (bishop pair, king safety, closed-position piece values); it TUNES the baseline's weights, it does not transcend the representational class. Success = RL-tuned weights + search beat hand-set weights + search; the ceiling is ~pst-level, strength coming from the deep search the µs eval enables. MEASURED (8 iters, 150 games, d2, gradient-MC): FAILED — the value FIT converged cleanly (VE mse 0.78→0.12, S&B §9.4 single-optimum confirmed) BUT the eval PLAYS WORSE: linear-RL vs pst @d2 = 0/30 (score 0.03, −585 Elo). This is S&B §9.2 empirically: minimizing VE is a proxy that DIVERGED from play — regressing weak self-play OUTCOMES erodes the sound hand-`pst` weights. Disposition: RL-tuning a linear eval on self-play outcomes does NOT beat hand-tuning; `pst` is near the linear-feature ceiling. (A semi-gradient TD(λ) target would still be gated by the same §9.2 proxy gap — not pursued.)

***implementation reqs***

- Constants: LINEAR_LR, LINEAR_LAMBDA, SELFPLAY_WORKERS, SELFPLAY_DEPTH, LINEAR_RANDOM_PLIES.
- File: NEW `train_linear_rl.py` — MP self-play (alpha-beta + :Linear-value:) → GD on outcomes → `models/linear_rl.npz`; warm-start w from `models/linear.npz`. Reuses `engine.AlphaBetaEngine`, `gbdt_features.features`, `measure_ladder` for the :Linear-kill-check:.
- :Linear-value: MUST emit White-absolute scores (same frame as `pst_eval`), so `AlphaBetaEngine(eval_fn=...)` is unchanged.

***test reqs***

- The :Afterstate-target: gradient reduces training MSE on a held-out set (a learning check).
- The :Linear-kill-check: comparison: :Linear-value: vs `pst_eval` at equal depth on the :Ladder:, n ≥ 30 with CIs.

***functional specs***

- :Linear-value: must plug in as an alpha-beta `eval_fn` unchanged.
  - Given a board, When :Linear-value: is called, Then it returns wᵀφ(board) (White-absolute) and `AlphaBetaEngine(eval_fn=...)` runs unmodified.
- :Parallel-selfplay: must be process-parallel and GPU-free.
  - Given SELFPLAY_WORKERS workers, When self-play runs, Then each is a separate OS process playing alpha-beta(:Linear-value:) games on CPU, returning (φ, outcome) trajectories; no GPU is used.
- Training must swap the supervised label for the self-play outcome.
  - Given a self-play trajectory, Then w is updated by GD toward the outcome z / :Lambda-return: (NOT the SF cp label), warm-started from `models/linear.npz`.
- The gate is equal-depth vs the baseline.
  - Given the RL-tuned :Linear-value: does NOT beat `pst` at equal depth on the :Ladder: (CIs overlapping/below), Then it does not ship; the hand baseline stands.
