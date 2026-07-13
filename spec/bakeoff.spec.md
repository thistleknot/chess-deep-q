# :Bake-off: — the feature-set / organ testing approach (operator method, spec'd)

The standing procedure for choosing between CANDIDATE COMPONENTS (feature encodings,
mechanism "organs", architectures): parallel 3-trial Optuna studies, one candidate per
study, everything else pinned to the organ-free baseline. Optuna is the judge ("optuna
would be sufficient to identify the appropriate feature set — 3 trials per").

## Protocol

1. **One candidate per study.** The study's config = the BASELINE (organ-free canon: graded
   ladder, KC-faithful TDLeaf d2, proven trivium anneal, raw z) + exactly the candidate.
2. **Identical infrastructure everywhere** (s100 × e1 × elo12 × b20 under the 30-min cap;
   these are controls, never varied between candidates). Identical seed PROTOCOL: each
   candidate's provenance-pure seed via `QLEARN_TUNE_SEED`; trial 0 = the proven parms
   (auto-enqueued), so every study contains the same reference point.
3. **3 trials per candidate**, parallel launches permitted (contention stretches wall-clock,
   not validity — the games are still the games).
4. **Compare study BESTS on one axis** (`/compare` page, data/bake_*.log + organ_*.log);
   single-trial noise is ~±100, so candidates within ~100 of each other are TIES — break
   ties by cost (dimensionality, wall-clock), never by vibes.
5. **Dependency-honest decomposition**: organs that consume another organ's output are
   tested as MARGINALS on their dependency, never "alone" when alone is inert (e.g. GRPO
   and replay-T consume the surprise score ⇒ arms are surprise / surprise+GRPO /
   surprise+replayT / DDQN). An "isolated" test of an inert flag is a false control.
6. Winners graduate to a 30-epoch run on tuned parms; losers are archived with their
   numbers. The final composition run (winning set + arch + surviving organs) happens ONCE,
   at the end — never as the starting point. (The week's lesson, operator-diagnosed:
   imagining the downstream full kit first is the anti-pattern; this spec is its inverse.)

## Current matrix (2026-07-12)

- Feature sets: kc-zca 942 ★ | pca-320 941 (tie, 2.5× fewer dims) | kc-raw 878 |
  nk-512 782 | k809-2048 expansion — running.
- Architectures: linear (B0, running) vs mlp-64 (mlpb, running) — the shutdown-clause arm.
- Organs (parallel studies running): surprise / surprise+GRPO / surprise+replayT / DDQN.
- Graduated: kczca 30-epoch run on the 942 parms — running.

- :Relay: (spec/relay.spec.md) — run-timescale organ: diagnosed-restart waves with paired controls; acceptance = beats control-only relay over the same budget (Merge 19).

## Organ close-outs (2026-07-13, SME-corroborated)

- **GRPO: closed permanently — measured negative (913/879 ≤ 965 baseline) now EXPLAINED.**
  (a) Mechanism: GRPO's group-relative z-score is a critic-FREE Monte-Carlo baseline,
  designed for the sparse-terminal-scalar / no-value-function setting (DeepSeekMath
  motivates it explicitly as removing the critic). Our pipeline already carries a
  lower-variance search-bootstrapped critic (trivium compound target) — a group baseline
  on top adds outcome-level noise and discards per-move credit. (b) Placement defect:
  we routed the group z-score into VALUE regression; a whitened z-score is not a valid
  value target (it destroys the absolute scale a critic must learn). The faithful
  placement is a policy head (sibling-rollout leave-one-out), which this architecture
  does not have — so no retry is warranted. If a low-sim policy-target lane ever opens,
  the published operator for that regime is Gumbel AlphaZero (Danihelka et al. 2022),
  not GRPO.
- **Capacity verdict caveats (re-audit, Merge 20):** the fast triple already ran the
  Adam+replay recipe (headers show no KC-FAITHFUL token) and the net output was always
  tanh-squashed (WDL-like space since Merge 2) — but it was data-starved for the MLP
  (48 games ≈ 3k positions vs ~49k MLP-64 params) and used plain ReLU at a linear-tuned
  step size. Remaining recipe deltas under test in bake4: clipped ReLU + TPE over the
  mlp step-size range (paired with a linear control in the same regime). Data volume
  stays a flagged confound that only a big-corpus trainer route resolves.

## Merge 20 verdicts (2026-07-13, bake4 — 3 studies, 3 trials each, s100/e1/elo12)

- **mlp64 crelu+adam 783 ≈ linear+adam control 781** (identical regime, Δ2 ≪ ±100 band)
  → tie → control wins by cost. The recipe-artifact hypothesis is DEAD: with clipped
  ReLU, Adam+replay, and TPE'd step size all discharged, MLP still ties linear. The
  capacity verdict SURVIVES its re-audit at ~10⁶ positions; the last standing
  explanation for MLP ≤ linear is DATA VOLUME (documented NNUE-class gains live at 10⁸⁺).
- **kpst 834 < 865** (benchmark 965 − band; trials 708 / 586 / 834, best at α 6e-4 —
  2× the proven α, consistent with the 4× per-feature dimension shift). REFUTED AT THIS
  CORPUS, but NOT claimable as "king-conditioning worthless": the H-b occupancy
  discriminator (data/kpst_occupancy.md) measured **71.5% of positions in one bucket**
  (94.8% in the opening slice; buckets 1 and 3 combined < 1%; boundary crossings 1.19%
  of plies — TD targets straddling disjoint weight sets). Dilution makes 834 the
  EXPECTED outcome at 10⁶. Per pre-registration: do NOT ladder to 8 buckets on this
  corpus (doubles dilution).
- **Side observation**: linear under Adam+replay (781) scores ~180 below the same
  config under KC-faithful online SGD (965) — the faithful online recipe is strongly
  regime-superior for linear at this scale; capacity arms were not handicapped by it.
- **Q8 battery** (data/mixed_eval_battery.md): rollout 6/9 > mixed 4/9 > static 3/9
  sign-agreement vs SF anchor — AlphaGo's λ=0.5 preference reverses under a biased
  value head; mate-in-1 leaf holes fully repaired by rollouts; the queen-fork position
  survived ALL evaluators (rollout policy = greedy over the same V inherits the move-
  preference hole). Sixth independent representation diagnostic.
- **Joint disposition**: king-conditioning and nonlinearity both fail at 10⁶ positions
  and both have documented gains only at 10⁸⁺ — data volume is the binding
  PRECONDITION, not feature choice. Both questions fold into the bullet route
  (replicate-before-invent: jw1912/bullet on self-generated labels, result weight 0),
  where generation throughput is fixed simultaneously. Bullet route = separate merge,
  operator go required.
