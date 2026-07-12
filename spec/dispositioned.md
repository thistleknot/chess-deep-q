# Dispositioned specs — index of everything superseded by the canon

Canon: [`spec/trivium.spec.md`](trivium.spec.md). Live operating protocols:
[`expectations.spec.md`](expectations.spec.md) (grading rubric),
[`intervention-queue.spec.md`](intervention-queue.spec.md) (web council / sidecar),
[`console-pargen.spec.md`](console-pargen.spec.md) (console surface contract).
Everything below lives in `spec/archive/` with its disposition.

## Absorbed into the canon (mechanisms live in code, spec text historical)

- `tdleaf.spec.md` — TDLeaf(λ), :Backed-bootstrap:, :Ramp-filter:, :Confirmed-crown:,
  :Informative-patience:, :Trivium-anneal:, :Graded-opponents: — all folded into the recipe.
- `knightcap-full.spec.md` — KC-faithful mode (donor fidelity); the replicate-first pillar.
- `parallel-generation.spec.md` — PARGEN native self-play batches (Merge 9).
- `eval-features.spec.md` — the 809 donor feature set (Merge 6).
- `rust-search.spec.md` — rsearch engine (Merge 8); v3.5 LMR/aspiration honestly falsified
  (tree already skeletal — LESSONS #21); depth is an inference converter.

## Superseded

- `pathfinding.spec.md` — :Out-of-bounds: exploration framing → superseded by the
  expectations rubric + council protocol.
- `q-learning.spec.md`, `actor-critic.spec.md` — Merge 2/3 lineage; AC dispositioned at the
  actor-sharpness wall; Q-learning evolved into the canon recipe.
- `decision-time-planning.spec.md`, `self-improvement-loop.spec.md`, `environment.spec.md`,
  `observability.spec.md` — early-era scaffolding, superseded by the console + canon.

## Prior archive (pre-campaign era, already under `spec/archive/`)

`annealing-schedule` `chess-rl` `dynamic-difficulty` `elo-calibration` `elo-measurement`
`entrypoint` `learned-model` `linear-value-rl` `nnue-eval` `prior-evaluator`
`rl-categorization` `search-mcts` `self-play-leela` `teacher-distillation`
`terminal-interface` `training-loop` `value-target` — the exploratory era that the
replicate-first pivot ended. `nnue-eval` may be revived by the queued capacity arm
(Merge 10 / purist representation queue).

## Archived UNTESTED (operator call, 2026-07-12 — all effort to Merge 14)
- Merge 10 (incremental eval / efficiency) — spec'd in LESSONS #21 context, never built.
- Merge 12 (policy head, spec/policy-head.spec.md) — spec committed, demoted by council
  round #7 (no linear-capacity precedent), never implemented.
- Merge 13 (Giraffe-lineage MLP capacity) — council-ordered, never implemented.
These remain re-openable; specs/ledger entries preserved. Active line: Merge 14 (GRPO + Elo
surprise, spec/relative-reward.spec.md).
- Merge 14 (rating-surprise + graded, spec/relative-reward.spec.md) — mechanics IMPLEMENTED
  and smoke-passed, arm never run; superseded by Merge 15 (spec/admixture-replay.spec.md)
  which includes its machinery. Archived as an arm 2026-07-12.
- Merge 15 (admixture replay, spec/admixture-replay.spec.md) — mechanics implemented; its
  3-trial study COMPLETED (best 892 = proven parms + replay_t 1.0, strongest clean-regime
  trial to date); arm superseded by Merge 16 full stack (spec/full-stack.spec.md) which
  includes all its machinery + the magic deck. Archived 2026-07-12.
