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
- :Advantage-shrinkage: is a SIGNIFICANCE filter on the advantage A = G_t^λ − V(s_t): a transition enters the policy update only if its advantage is statistically distinguishable from 0 given its uncertainty — |A| / σ_A > z_α, equivalently the interval A ± z_α·σ_A excludes 0 (the regression-coefficient t-test read off a whitened advantage; "act only on significant changes"). It DROPS insignificant transitions (a prioritized-replay filter), not zeroes them — zeroing a kept sample saves no compute (the backward pass still runs), whereas dropping focuses gradient on the capability boundary (:Value-of-information:) and denoises Bellman-residual noise. Mechanism, named precisely: this is a HARD significance gate — keep-at-full-value or DROP — i.e. subset selection / L0, NOT soft-threshold shrinkage (L1-prox would shrink each kept |A| by ~z_α·σ_A while keeping every sample, saving no compute); the term "shrinkage" names the sparsifying effect on the signal, not an L1 shrink of retained values. Whitening MUST be ZCA (zero-phase), not PCA, so within-trajectory correlated advantages are decorrelated IN THE ADVANTAGE BASIS and each transition's significance test is independent. It is SELF-ANNEALING: as V converges σ_A shrinks, so a FIXED α automatically admits progressively finer real advantages — this supersedes any hand-tuned magnitude dead-zone ε (deleted). σ_A is estimated cheaply as the batch advantage spread (= standard advantage normalization → significance RELATIVE to the batch, not an absolute p-value), or better from the λ-return's n-step variance / the :Search-window-reuse: δ-spread. Multiple-comparisons applies (at α over a B-batch, ~α·B pass by chance): control the false-discovery rate (Benjamini-Hochberg) across the batch, not a per-transition α, when it bites. STAGE-2+ only — advantage exists only once the λ-return refinement runs; Stage-1 distillation (supervised MSE) has none. NOT a precision/bit trick: A is a transient fp32 scalar; the memory win lives in :Encoding-packing: (learned-model).
- :Distillation-anchor: is an optional small, persistent SUPERVISED term blended into the self-play value loss: DISTILL_ANCHOR_ALPHA · MSE(V(s), tanh(SF_cp(s)/400)) over a held Stockfish-labelled anchor set, ADDED to the self-play outcome/λ-return loss. It tethers the value head to Stockfish's calibrated evals where they exist, guarding the self-play stages against drift / reward-hacking — the value analog of DEMO_SHARE_FLOOR (data) and the shaped-reward floor (reward). DISTILL_ANCHOR_ALPHA anneals from high early (lean on the teacher) toward a small FLOOR (never 0 — a permanent anchor), the same prior-lineage shape as :Bootstrap-share:. It is a REGULARIZER, not the objective: self-play outcomes remain the primary value signal and are the only channel that can EXCEED the teacher. OFF by default (α=0) in the pure heuristic-vs-distilled bootstrap comparison, which measures UNAIDED self-play; ON as the anti-drift anchor when chasing the :Surpass-teacher-gate:.

***implementation reqs***

- The return math lives in pure, stateless functions in `value_targets.py` (`td0`, `mc_return`, `nstep_return`, `lambda_return`, `retrace_weights`), decoupled from any trainer; consumed by the Stage-2 refinement trainer and the Stage-3 self-play trainer.
- λ is derived from the schedule's :Bootstrap-share: β as λ = 1 − β; there is no separate λ constant.
- Constant: GAMMA — **pinned to 1** for the value target (chess is episodic with a bounded horizon and no natural discount, matching AlphaZero). With γ=1, λ=1 is exactly the terminal outcome z; any γ<1 would make λ=1 a *discounted* z, so the "collapses to z" claim holds only at γ=1.
- Constant: REPLAY_WINDOW — the :Replay-window: size (last-N self-play games) that keeps the uncorrected λ-tail near-on-policy.
- Constant: DISTILL_ANCHOR_ALPHA (start/floor) — the annealed :Distillation-anchor: weight; the anchor set is the held Stockfish-labelled :Cumulative-dataset:. Floor > 0 (permanent tether), α=0 disables it (the pure-self-play comparison).
- Constant: ADV_ALPHA — the :Advantage-shrinkage: significance level (z_α). The filter ZCA-whitens advantages and is applied at replay sampling, so dropped transitions cost no gradient step; the default σ_A estimator is the batch advantage spread (advantage normalization). No annealed magnitude ε — the significance test self-anneals via σ_A.
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
- :Advantage-shrinkage: must DROP by significance, not zero by magnitude, and must whiten with ZCA.
  - Given |A| ≤ z_α·σ_A (the confidence interval includes 0), Then the transition is excluded from the update batch — not kept with A zeroed (a kept-and-zeroed sample still runs the backward pass).
  - Given correlated within-trajectory advantages, Then they are ZCA-whitened (not PCA) before the per-transition significance tests, so the tests are independent in the advantage basis.
  - Given V converging (σ_A shrinking), Then a fixed ADV_ALPHA admits progressively finer advantages — there is no separate magnitude-ε anneal.
  - Given a large batch filtered at per-transition α, Then false-discovery control (Benjamini-Hochberg) governs, not a raw per-test α.
  - Given Stage 1 (supervised distillation), Then :Advantage-shrinkage: does not apply (no advantage exists until the λ-return refinement).
- :Distillation-anchor: must be a floored regularizer, not the objective.
  - Given self-play value training with the anchor ON, Then the loss is (self-play outcome/λ-return MSE) + DISTILL_ANCHOR_ALPHA·MSE(V, SF-label); Given rising :Gated-progress:, Then DISTILL_ANCHOR_ALPHA anneals toward its floor (never 0) so the net stays tethered to calibrated evals while self-play outcomes drive improvement past the teacher.
  - Given the pure bootstrap comparison, Then the anchor is OFF (α=0) so unaided self-play is what the :Ladder: measures.
