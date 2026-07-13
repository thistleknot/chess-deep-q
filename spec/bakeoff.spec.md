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
