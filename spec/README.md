# chess-deep-q spec — RL from first principles

The goal was **to understand RL by building it up one reviewable merge at a time** — strength
as the yardstick, every feature spec'd, small, traceable. The build-up succeeded and produced
a canon.

**Start here: [`trivium.spec.md`](trivium.spec.md) — the enshrined lesson.** Sparse-depth
trivium RL (λ-return + d2 search value + outcome, Optuna-tuned anneal, ZCA, self-play volume)
holds the 1428–1672 goal band beyond doubt (1484, CI 1434..1542, 200 games) from a net that
never saw an external opponent in training. The canon carries the exact recipe, the
measurement scales, and the governing rules.

Operating protocols, still live:
- [`expectations.spec.md`](expectations.spec.md) — self-grading rubric (Below/Met/Exceeded)
- [`intervention-queue.spec.md`](intervention-queue.spec.md) — web council / sidecar advisers
- [`console-pargen.spec.md`](console-pargen.spec.md) — training-console surface contract

Other active specs (current, non-archived):
- [`bakeoff.spec.md`](bakeoff.spec.md) — feature/organ A/B screening protocol
- [`h2h-instrument.spec.md`](h2h-instrument.spec.md) — duel-based Elo measurement ruler
- [`pathfind-population.spec.md`](pathfind-population.spec.md) — zero-seed population search
- [`repo-layout.spec.md`](repo-layout.spec.md) — package layout (executed)
- [`schedules.spec.md`](schedules.spec.md) — LR/λ/γ/mix dial panel
- [`bullet-route.spec.md`](bullet-route.spec.md) — NNUE/bullet corpus pipeline
- [`search-arms.spec.md`](search-arms.spec.md) — decision-time search screen
- [`search-distill.spec.md`](search-distill.spec.md) — search-bootstrapped self-distillation /
  champion promotion (current champion recipe)
- [`full-stack.spec.md`](full-stack.spec.md) — all-organs-on regime (Merge 16, supersedes
  the relative-reward → admixture-replay lineage, now archived)

Everything else is dispositioned: [`dispositioned.md`](dispositioned.md) indexes every
absorbed/superseded/falsified/parked spec, with the files preserved under `archive/`.
