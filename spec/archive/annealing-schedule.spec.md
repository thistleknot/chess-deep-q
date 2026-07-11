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
