# THE TRIVIUM RECIPE — the enshrined lesson (canonical spec)

**Status: CANON.** This spec governs the project. Everything else in `spec/` is either an
operating protocol (expectations, intervention-queue, console surface) or archived under
`spec/archive/` with its disposition recorded in `spec/dispositioned.md`.

## The lesson

**Sparse-depth trivium RL works — and brute training depth was never needed.** A linear eval
over 809 donor features, trained by TDLeaf on compound targets — the *trivium*:
`G = a·λ-return + b·search-value + c·outcome` with the weights **annealed on an
Optuna-tuned schedule** — climbs from scratch on pure self-play with **nothing deeper than
2 ply anywhere in training**. λ (the eligibility-trace horizon, the n-step-advantage analog)
substitutes for depth; the tiny d2 glance keeps targets sound; the outcome term anchors early
learning and anneals away. Depth belongs at *inference* (play time), where it is a pure
converter: the same net that trains at d2 measures 1484 claims-grade at d7.

Every ingredient was validated single-variable against a faithful donor baseline
(replicate-first, then extend):

| Ingredient | Single-variable evidence |
|---|---|
| Trivium anneal (Optuna-tuned, ~21 trials) | matched-depth rung 1237 vs 1166 un-annealed; static ⅓⅓⅓ ignites-then-fades (falsified); 7-trial re-probe found nothing better |
| ZCA whitening of the 809 features | matched faithful-arm strength on ~¼ the games |
| d2 sound targets (not d4) | E1: equal strength at half the clock — depth beyond soundness pays nothing |
| Self-play volume (PARGEN native) | 15k games → the 1484 beyond-doubt net; restart reproduced its peak (32.56) in 8k |
| Measurement discipline | confirmed crowns (S&B 015 outer-loop), informative patience, 200-game claims rungs |

## Provenance (measured, claims-grade)

- **1484 (1434..1542), 98W-92D-10L @200 games vs SF@1320** — pure-self-play net + rsearch d7.
  CI floor above the 1428 band floor: the 1428–1672 goal band is held **beyond doubt** by a
  net that never saw an external opponent in training. (`data/archive/`, MLflow, tag
  `pre-triv-restart`, commit f3c9871.)
- Restart reproduction (this recipe, fresh seed, clean board): six consecutive confirmed
  crowns to 32.56 in 8,000 games — the prior campaign's 15,000-game peak, at ~half the games.

## The recipe (exact, console defaults as of commit 5eda481)

TDLeaf(λ) generation at **d2** (native `rsearch3`), KC-faithful (λ=0.7, γ=1.0, online SGD,
RAMP filter), 809 donor features (`enc=kc`), **ZCA** (`models/zca.npz`, pristine whitened
seed), **trivium** `0.285,0.341,0.374 → 0.516,0.341,0.143` @ warmup `0.481`
(λ-return, search, outcome), α `0.0003`, PARGEN parallel self-play (ε=0.1, 12 threads),
1000-game epochs × 30, informative patience 4, confirmed crowns on, epoch-Elo 24.

## Measurement scales (never conflate — the 2026-07-11 correction)

1. **d2-greedy scale** (live console Elo, crown metric, Optuna objective): the net playing as
   it trains, with its 2-ply glance. The operator's 900→1100+ arc lives here.
2. **Raw purist scale** (1-ply argmax, no look-ahead): compression stress-test. Ceiling so
   far ~903 — the open question. Flat verdict #3 fires the representation queue
   (capacity/hidden-layer arm, Kanerva — one per arm, council-reviewed).
3. **Deep-inference rungs** (d5–d7 ladder, 60g scouting / 200g claims-grade): where band
   claims are made. Report Elo-equivalents, name the scale, every time.

## Governing rules (unchanged, load-bearing)

**:Provenance: (added 2026-07-11):** any run claiming "from scratch" must pass the purity
law in spec/pathfind-population.spec.md — every input derivable from rules + RNG + declared
constants + hyperparams + feature definitions; NOTHING computed from a trained model's
weights or play (seeds, labels, schedules, preprocessing statistics like ZCA all included).
Pre-run provenance checklist is mandatory and logged.


Single-variable arms with pre-registered falsification · replicate-before-invent ·
formula/manifold change ⇒ Optuna re-tune, **≤30 min wall, new studies only on operator
request** · infrastructure vars are controls, never search dims · operator starts final runs
from the console · web council convenes at crown/verdict checkpoints and on every
Below-Expectations event · no engine-label distillation ever (own-search self-distillation is
compliant) · treasures (`*_best.pt`) are append-only and committed with their numbers.
