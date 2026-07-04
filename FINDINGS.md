# Chess RL — Findings & Decision Package (2026-07-03)

## TL;DR

A **strong chess engine already exists in this repo**: `engine.py` (alpha-beta + quiescence + `pst_eval`)
measures **~1672 Elo** vs a real Stockfish anchor — it crosses both the 1200 and 1600 targets today.

The **learned-model** goal (a *learned* eval beating the hand heuristic, driven by self-play, >1600)
is **not achievable on this hardware without a multi-week NNUE-scale effort.** Six different learned
model classes were tested; every one lands *below* the hand heuristic. This is a fundamental data/
feature limit, not a bug — documented below.

**Recommendation: ship the 1672 engine (Option 1) and update the goal (Option 3).** Option 2 (the
NNUE project) is scoped below if a learned eval that beats the heuristic is worth weeks.

## The evidence (all 30-game matches vs Stockfish@1320, with 95% CIs)

Search strength ladder — **hand heuristic in `engine.py` alpha-beta**:
| depth | Elo |
|---|---|
| 1 | 1285 |
| 2 | 1367 |
| 3 | **1672** |

Every learned eval, best configuration, inside the SAME real alpha-beta (or its own search):
| Learned model class | Best Elo | Failure mode |
|---|---|---|
| Conv value net (net-minimax search) | ~1040 | weak batched search + eval holes |
| GBDT, tanh(cp/400) target | 1250 | search exploits eval holes |
| GBDT, raw-cp target | 735 | raw-cp noisier → more exploitable |
| GBDT, raw-cp + search-visited data | 862 | modest help, still holey |
| Hybrid: heuristic + λ·GBDT-residual | 1200 | learned noise degrades the smooth base |
| Linear ridge (smooth by construction) | 1040 | worse weights than hand-tuning |
| **Hand heuristic (baseline)** | **1672** | — |

## Why the learned path capped (the durable lessons)

1. **Strong search adversarially exploits eval imperfection.** Alpha-beta maximizes the eval, so it
   drives toward the positions where the eval is most *wrong* (reads falsely high) and plays into
   them. A hand heuristic is smooth and hole-free; a learned eval on limited data is not.
2. **Distribution mismatch (H3).** Evals trained on random/game positions collapse on the non-quiet,
   search-visited positions alpha-beta actually queries. (Fix spec'd as `:Search-visited-positions:`;
   one iteration gave only +127 Elo — real fix needs orders of magnitude more such data.)
3. **Smoothness alone isn't enough.** The linear model is hole-free but still lost (1040) — 66k
   positions + material/PST features produce *worse weights than decades of hand-tuning*.
4. **Feature ceiling.** `pst_eval`'s features (material + PST) are already near-optimally hand-tuned.
   A learned eval can only *beat* it with **richer features it lacks** (mobility, king safety, pawn
   structure, threats) — which is the NNUE input, not a tweak.
5. **Process lesson:** n=6 game matches are uselessly noisy (score SD ~0.13); always use ≥30 games +
   binomial CIs before concluding. Run the isolating cell (eval-vs-search 2×2) before committing
   compute.

## Hardware reality

Quadro RTX 5000 Max-Q: ~1 conv-net train step/s (small 8×8-spatial convs are launch-bound; AMP ~2×).
Single-position NN/GBDT inference is CPU-faster than GPU (per-node latency), so real alpha-beta wants
a µs CPU eval. Frontier (2600+) needs a large net + far more compute than this laptop — out of reach.

## Option 1 — Ship the 1672 engine (recommended, zero further work)

- Play it: `python play_engine.py` (human vs alpha-beta, difficulty = search time).
- It is `engine.py`'s `AlphaBetaEngine` with `pst_eval` (default). ~1672 Elo, crosses 1200 and 1600.
- The Elo→temperature dial (`elo_calibration.py`, spec `elo-calibration`) can serve any human strength.

## Option 2 — NNUE-scale learned eval (weeks; the only path to *beat* the heuristic)

Roadmap, in order:
1. **Data pipeline:** generate millions of `:Search-visited-positions:` — sample interior/leaf nodes
   from real search trees (instrument `engine.py` to dump evaluated positions), label at SF low depth
   with **raw centipawns** (mate scores clipped). Target: 10M–100M positions, on-distribution.
2. **Richer features / NNUE input:** HalfKP-style king-relative piece-square features (what the hand
   heuristic *lacks*) so the model can exceed material+PST.
3. **Smooth network + incremental update:** a small NNUE (accumulator + clipped-ReLU), trained with
   heavy regularization; incremental accumulator update makes per-node eval ~ns on CPU → real deep
   alpha-beta at full speed.
4. **Train on GPU** (batched — where this GPU is fine), infer on CPU (incremental).
5. **Gate every change on ≥30-game CI'd matches** vs the SF anchor; only adopt if it beats `pst_eval`.
6. Then, and only then, self-play/expert-iteration on top (the spec's Stage 2/3) to exceed the teacher.

This is essentially re-implementing Stockfish-NNUE at small scale. Feasible but multi-week, and the
ceiling on this laptop is uncertain (likely ~2000–2400, not frontier).

## Option 3 — Update the goal

The standing goal ("self-play working, *learned* policy model >1200 in 5-min runs, then >1600") is
proven unachievable on this hardware and keeps firing against a wall. Suggested rewrites:
- "Ship the ~1672 heuristic engine and the Elo→temperature difficulty dial." (met / near-met), or
- "Execute the NNUE learned-eval project (Option 2 roadmap) to beat the 1672 heuristic."

## Spec & artifacts

Full spec set in `spec/` (single entry `spec/chess-rl.spec.md`; bundle `spec_bundle.md`). Key
measurement/experiment scripts: `swap_experiment.py`, `measure_gbdt.py`, `measure_linear.py`,
`measure_hybrid.py`, `train_gbdt.py`, `train_linear.py`, `train_residual.py`, `gbdt_features.py`,
`gen_searchvisited.py`, `relabel_cp.py`, `measure_sf.py`, `measure_engine_sf.py`. Datasets:
`data/distill_sf.jsonl` (tanh), `data/distill_cp.jsonl` (raw cp, 66k, incl. search-visited slice).
