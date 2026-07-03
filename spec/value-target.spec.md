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

***implementation reqs***

- The return math lives in pure, stateless functions in `value_targets.py` (`td0`, `mc_return`, `nstep_return`, `lambda_return`, `retrace_weights`), decoupled from any trainer; consumed by the Stage-2 refinement trainer and the Stage-3 self-play trainer.
- λ is derived from the schedule's :Bootstrap-share: β as λ = 1 − β; there is no separate λ constant.
- Constant: GAMMA — **pinned to 1** for the value target (chess is episodic with a bounded horizon and no natural discount, matching AlphaZero). With γ=1, λ=1 is exactly the terminal outcome z; any γ<1 would make λ=1 a *discounted* z, so the "collapses to z" claim holds only at γ=1.
- Constant: REPLAY_WINDOW — the :Replay-window: size (last-N self-play games) that keeps the uncorrected λ-tail near-on-policy.
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
