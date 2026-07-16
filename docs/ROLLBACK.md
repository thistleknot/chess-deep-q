# Rollback point — 2026-07-11, pre trivium-restart (tag `pre-triv-restart`)

State of the campaign at the operator's "restart fresh with the trivium approach" directive.
Everything below is restorable from this tag + the committed model checkpoints.

## Confirmed numbers (the treasures)

| Artifact | What it is | Measured |
|---|---|---|
| `models/qlearn_kc7_best.pt` | KnightCap-lineage champion (donor init + faithful recipe) | **1441 (1392..1495) claims-grade @200g** vs SF@1320 at rsearch d7; rungs 1212(d5)/1337(d6)/1460(d7) |
| `models/qlearn_vol_best.pt` | volume net — PURE self-play, 15k games, ZCA + tuned trivium anneal | **1552 (1458..1691) @60g at d7** (campaign high); purist 1-ply 903; 200g claims confirmation was in flight at tag time (lands in `data/rl_trend.jsonl`) |
| `models/qlearn_kc7k_best.pt` | combined-winners arm (ZCA + d4 + tuned anneal) | 1384 @ d7; matched-depth 1237 |
| `models/qlearn_kc8_best.pt` | flagship (same recipe, long horizon) | own rung 1121 — run variance, closed |
| `models/qlearn_wseed.pt` | PRISTINE whitened seed (donor init in ZCA space, never RL-trained) | seed for all ZCA lanes |
| `models/zca.npz` | ZCA whitening (Z, mu) for the 809 features | head back-conversion: `w = Z @ w'`, `b = b' − w·mu` |

Purist lane (metric of record): 903 (573..1023) — flat, pivot count 2 of 3
(next representation arm if #3 flat: Kanerva, then RBF, sequentially).

## Tuned hyperparameters (study 4c4e03dd, trivium-anneal manifold)

alpha 0.0003, KC-faithful (lam 0.7, gamma 1.0, online SGD), trivium anneal
outcome 0.374→0.143 @ warmup 0.481, search weight 0.341
→ env triples: `QLEARN_TRIVIUM=0.285,0.341,0.374` `QLEARN_TRIVIUM_END=0.516,0.341,0.143`
`QLEARN_TRIVIUM_WARMUP=0.481`.

## The shelved path (how to resume it)

Full-depth rung program shelved per operator directive (LESSONS #21: tree already skeletal;
node cost is the lever). To re-measure any net at depth: load checkpoint, convert head if
whitened (formula above), `rsearch3.Searcher(list(w), b).search(fen, depth)`, wrap with
`measure_elo.measure(agent, label, games, merge=N)`. Queued-but-cancelled: d10 rungs on
kc7_best/kc7k (2.6s/move measured viable).

## What replaces it

Fresh trivium lane driven from the console: fresh Optuna study (regime = TDLeaf d2 +
KC-faithful + ZCA + trivium anneal + PARGEN native self-play, seeded from wseed), then the
operator loads Optuna best and starts the final run from the UI. Console gains the missing
knobs (trivium/zca/rsearch/faithful/pargen/lineage) in the commit AFTER this tag — `git
checkout pre-triv-restart` restores the console exactly as it was before that surface change.

## In-flight at tag time

- 200g claims run on the volume net (background; appends "VOLUME net (pure self-play) +
  rsearch d7 (200g CLAIMS CONFIRMATION)" to `data/rl_trend.jsonl` and MLflow when done).
- Ladder history: `data/rl_trend.jsonl`; ledger: `data/experiments.md`; canon: `LESSONS.md`
  (1–21); MLflow: `sqlite:///mlflow.db`, runs tagged with git commits.

## UI reset (post-tag, 2026-07-11)

Console board wiped for the trivium-restart campaign: full 207-row ladder history moved to
`data/archive/rl_trend_pre-triv-restart.jsonl` (+ old live metrics alongside). Every number
in the tables above lives in that archive; restore = move the files back. Trainer rung rows
now prefix `QLEARN_TAG` (trial3/final) so tuner trials are distinguishable on the ladder.
