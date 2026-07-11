# Merge 11 — Pathfinding: zero-seed population search over weight space

The research question (operator, 2026-07-11): **can the system FIND the weights itself** —
no donor seed, no distilled model — by treating training as *pathfinding through weight
space* (RRT*/A*: spread, branch what's promising, prune what isn't, never bet on one
trajectory)? Not an architecture arm; the capacity/hidden-layer arm stays queued separately.

Grounding (council round #3, data/council.md): PBT (Jaderberg 2017) + interval caveat
(Multiple-Frequencies PBT: too-frequent evolution = greedy collapse), optimistic-init
wash-out under function approximation (DOIE lineage), KnightCap's own material-only start
(cs/9901002 — the donor paper's from-scratch experiment).

## The mapping

| Pathfinding | Training realization |
|---|---|
| Node | a lane's checkpoint at a sync point |
| Expansion | 2 epochs (2×1000 games) of the CANON trivium recipe per lane, unchanged |
| Spread | 4 lanes: 1 canonical control + 3 with jittered ε / α / trivium-start / warmup |
| f(n) | confirmed-crown strength (noise-gated) + slope over the last generation |
| Rewire | at sync: prune bottom 2, fork the top lane's weights into freed slots with fresh jitter |
| Local-minima escape | population diversity + :Slope-guard: (below) |

## :Zero-seed: (build_zero_seed.py)

- Raw space: w = N(0, 1e-3) noise (symmetry breaking), **plus side-to-move optimism**
  `w[stm] = 2c, b = −c` with `c = 0.25` — the mover always believes +c (S&B §2.6 optimism,
  made zero-sum-symmetric; a global positive bias would be White-optimism/Black-pessimism).
- Pre-registered wash-out caveat: under linear FA the optimism deflates within early epochs
  (council refutation) — ε/τ exploration is the durable mechanism; optimism is the ignition
  nudge, not the engine.
- Whitened to ZCA space exactly like wseed (`w' = solve(Z, w_raw)`, `b' = b + w_raw·mu`);
  ZCA is a coordinate change, not knowledge. No `strength` key. → `models/qlearn_zero_seed.pt`.

## :Population: (pathfind.py)

- 4 lanes `a..d`, lane `a` = control (canonical proven config: α 0.0003, trivium
  0.285,0.341,0.374 → 0.516,0.341,0.143 @ 0.481, τ anneal 0.7→0.05); lanes `b..d` jittered
  per generation from a seeded RNG: τ_start/τ_floor × [0.5, 2], α × [0.5, 2], trivium
  c_start ± 0.1 (clamped, a = 1 − b − c), triv_warmup ± 0.15. **ε = 0 in all lanes.**
- :Softmax-tau: (rsearch v3.6 `rsearch4`, `QLEARN_PARGEN_SOFTMAX=1`): native generation
  samples moves ∝ exp(σ·v₁/τ) over 1-ply afterstate values, τ = the trainer's own annealed
  temperature. Exactly TWO exploration drivers (external-critique audit, council ledger):
  optimism (deflation dynamic; state-uniform, so it cannot rank moves) + softmax (direction:
  ~uniform over near-equal inflated values, sharpening as weights differentiate = natural
  explore→exploit anneal). τ is jitter/reheat-controlled, never Optuna-searched. Off-best
  samples record their own afterstate + 1-ply value (python softmax-path contract; no data
  starvation — measured 91 recs/game on the zero seed).
- :Reheat: (operator's stall rule): at each sync, if the population best failed to improve,
  jittered lanes' τ scale ×1.5 (cap ×4) — local optimum ⇒ temperature up; any improvement
  cools back to ×1. Logged as `REHEAT ×k` tree rows. (v2 refinement: also trigger on
  td_sigma tightening — read from lane metrics.)
- :Stale-reheat: (in-lane, operator: "as patience increases we do this factoring"):
  `QLEARN_STALE_REHEAT=k` scales the behavior τ by (1 + k·stale), cap ×4 — every informative
  failure raises exploration; a kept crown resets stale, so τ cools on progress by
  construction. Population lanes run k=0.5. Not an Optuna dim (exploration-knob rule).
- :Explorer-lane: (operator: "leave one for exploration"): the last lane's config is pinned
  hot (τ 1.4→0.3, canon otherwise) and NEVER re-jittered toward exploitation — when forked
  from the top node it keeps exploring from the best-known weights (RRT expansion from best).
- Each lane = detached `qlearn.py 1000 2` subprocess with the canon env
  (TDLEAF+KC_FAITHFUL+RAMP+ZCA+PARGEN, `QLEARN_PARGEN_THREADS=3`, per-lane
  CKPT/METRICS/TAG=`zero<A..D>`, RESUME=1, CONFIRM=1). qlearn.py is UNTOUCHED; lanes feed
  the ladder via :Crown-rung:.
- :Sync: (every generation = 2 epochs/lane): read each lane's `_best.pt` `strength`; rank.
  - :Slope-guard: the lane with the best strength-DELTA over the generation is exempt from
    pruning even if bottom-2 absolute (FIRE-PBT long-term-potential caveat).
  - Prune the (unprotected) bottom 2 → overwrite their ckpt+_best with the top lane's
    weights, `strength` and optimizer state STRIPPED (cross-regime-bar lesson, LESSONS #20
    discipline: forks are new lineages, not resumed bars) → fresh jitter next generation.
  - Append every node/prune/fork to `data/pathfind_tree.md` (generation, lane, config,
    strength, action).
- Defaults: 6 generations (≈48k population-games, ~6h at PARGEN throughput); all CLI-overridable
  (`python pathfind.py [gens] [epoch_games] [epochs_per_gen] [lanes]`). Full run is
  OPERATOR-STARTED (standing rule); smoke = `pathfind.py 1 200 1 2`.

## Pre-registered verdicts

1. **Ignition**: some lane's confirmed crown ≥ its first crown + 10 by **10k games PER LANE
   (generation 5)**, else verdict "trivium cannot ignite from optimistic zero" → council
   convenes; fallback rung = MATERIAL-ONLY seed (1,4,4,6,12 — the donor paper's own
   from-scratch start, paper-faithful, still no trained/distilled weights).
   *(AMENDED 2026-07-11 during gen 3, disclosed: originally "10k population-games" — an
   authoring error equal to 2.5k/lane, a window in which no arm in campaign history moved
   +10. Evidence at amendment time: pop best 5.92→5.92→9.27, trending but not +10.)*
2. **Pathfinding win**: winner's purist/crown within CI overlap of the donor-seeded arm
   (purist 982 (894..1046), crown 40.89) = the weights were FOUND, not inherited. Beating it
   = headline. Either way the winner gets purist + d7 scout + (if in-band) 200g claims.
3. Negatives are enshrined with the same prominence as wins (ledger + canon note).

## Acceptance

1. `py_compile` clean; zero-seed sanity: net loads, V(startpos) ≈ +c for the mover (raw
   space), first crown ≈ floor.
2. Smoke (`1 200 1 2`): 2 lanes spawn, sync ranks, prune/fork writes correct files, tree log
   rows appear, ladder rows carry `zeroA/zeroB` tags.
3. Full run: 4 SF instances coexist; population games/sec ≈ single-lane 12-thread rate;
   monitors + council checkpoints as in the triv arm.
