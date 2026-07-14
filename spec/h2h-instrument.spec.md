# :H2H-instrument: — paired head-to-head differential ruler (charter: instrument + volume)

## Why (and the declared deviation)

The pooled-SF objective quantizes for sub-floor nets (SF half-points over ~36 games:
0/36→585, 1/36→709, 1.5/36→782 — all arms ≥s50 pinned at one grid point). The charter
suggested "graded-rung pooled or ≥100 SF games/point"; both measure two floored
absolutes (p≈0.02 vs the anchor) and subtract. DEVIATION, declared: the question is a
DIFFERENTIAL, so the instrument plays the two arms AGAINST EACH OTHER at p≈0.5, where
per-game information is maximal. ~300 games → ±~40 Elo band at zero SF cost.

## Definitions

- **Duel** = N games between ckpt A and ckpt B. Openings: K=6 random plies from a
  seeded RNG; each opening played TWICE with colors swapped (paired design — opening
  bias cancels). Move policy: depth-2 width-8 search with root softmax at τ=0.02 (the
  KC-faithful dither) over the net's batched value_fn.
  **Pinned negatives (2026-07-13) — both constants are forced, re-pass G1/G2b on any
  policy change**:
  1. 1-ply greedy FAILED G2b — champion (12.87 crown, 5-way confirmed) scored 0.487 vs
     a 585-class net / 200g. Value quality does not express at 1 ply (strength=search,
     7th corroboration).
  2. d2 pure ARGMAX (τ=0) FAILED by repetition collapse — census: 29/30 games ended in
     FIVEFOLD_REPETITION (deterministic policies cycle-lock; decisive rate 10-12%, G2b
     +27..+37 with bands spanning 0 at 200-600g). The τ=0.02 dither breaks cycles; it
     is NOT exploration.
- **Adjudication**: rules result; at PLY_CAP=160, material count (1/3/3/5/9), diff ≥ 1
  pawn → win, else draw. (Declared. Was ≥2: the 2-pawn window drew 88% of G2b(d2) games
  — champion +37 with band −12..+85, gate unreadable. Between same-class sub-floor nets
  material at the cap is the skill signal; 1 pawn trades a small truth distortion in
  drawn-in-truth pawn-up endings for ~3× effective sample.)
- **Duel size**: gates and verdict rungs run 600 games (band ≈ ±28 at p≈0.55) — duel
  Elo is COMPRESSED relative to SF-anchored Elo (draw-heavy same-class play), so the
  band must be sized to the compressed differential, not the absolute scale.
- **Verdict numbers**: score s (draws=0.5), Elo diff via measure_elo.elo_diff(s, n),
  95% Wilson band on s mapped through elo_diff, decisive rate (instrument health).

## Contracts

- **Require**: both ckpts pst-encoded (enc=pst), arch read from ckpt metadata; same
  RNG seed per battery (openings identical across rungs → rungs comparable).
- **Guarantee**: paired symmetry — the same net dueling itself scores EXACTLY 0.500
  (null calibration gate G1).
- **Maintain**: no SF process, no training, no ckpt writes; each duel ≤ 15 min or runs
  as a background lane.
- **Assert (gates before any verdict is read)**:
  - G1 null: lin50 vs lin50 → s=0.500 exactly, Elo 0.
  - G2b known-separation: p7 champion vs lin25 → must separate beyond the Wilson band,
    champion up. (The original G2 pair lin50-vs-lin25 was CIRCULAR — its "known" gap
    of ~197 came from the old quantized ruler this instrument exists to replace; the
    champion's superiority is confirmed independently of that ruler.)
  - G3 health: decisive rate ≥ 20% (below that, the band is too wide to meet the
    charter's success test — widen N or revisit adjudication before reading verdicts).

## Validation record (2026-07-13) — INSTRUMENT ACCEPTED

- G1 null (τ build): lin50 self-duel, 100g → 0.490, band spans 0.5 ✓
- G2b: champion vs lin25, 600g → 0.681 = **+132 Elo (95% +102..+161)** ✓ (compare the
  old ruler: same pair unreadable at +27..+37 with bands spanning 0)
- G3 health: decisive 97% ✓ (was 10-12% pre-dither)

## Verdict protocol (charter)

Duels mlp_v vs lin_v at v ∈ {25, 50, 100}. SUCCESS = |Elo diff| > band at v=100 (either
direction), slope read across rungs. Arms inseparable at 4× volume (|diff| ≤ band with
G3 healthy) → STOP-LOSS: capacity closed PERMANENTLY, re-diagnosis delivered.

## Rung results (2026-07-13, 600g each, seed-0 training runs)

| rung | mlp − lin | 95% band | decisive |
|---|---|---|---|
| 25  | −1  | −28..+27 | 98% |
| 50  | +52 | +24..+80 | 98% |
| 100 | +23 | −5..+50  | 98% |

Onset slope 25→50 significant (+53 ± ~40); no growth 50→100 (−29 ± ~40); pooled 50+100
(1200g): **+37 (+17..+57)** — excludes zero. First direct capacity separation ever
measured in this repo. Caveat: n=1 TRAINING run per rung — duel bands cover measurement
noise only, not training-seed noise.

**Confirmation leg (pre-registered before launch):** retrain both s50 arms at
QLEARN_SEED=1, duel 600g (`rung50b`). mlp-up, band excludes 0 → capacity effect
CONFIRMED (constant small effect ≈ +35 duel-Elo at ≥2× volume; charter closes SUCCESS).
Band spans 0 → pooled across seeds decides. Sign REVERSED beyond band → seed variance
dominates, rung-50 result downgraded to unconfirmed, more seeds or stop-loss review.

## CHARTER VERDICT (2026-07-13) — SUCCESS branch, confirmed

rung50b (seed 1): **+30 (95% +2..+58)**, mlp up, band excludes zero → CONFIRMED per the
pre-registered rule. Two independent training seeds, same sign, both exclude zero
(seed 0: +52 +24..+80; seed 1: +30 +2..+58; pooled 1200g ≈ +41).

**The instrument-grade answer to "does capacity pay with volume":**
1. Capacity PAYS — a small, replicated ~+30..+50 duel-Elo edge for mlp64-crelu over
   linear, measurable from 50 games/epoch. The prior "capacity closed: dead tie"
   verdict was INSTRUMENT BLINDNESS, now formally overturned at the differential level.
2. Volume does NOT (yet) — no measured growth 25→100 games beyond the onset; the
   "more data unlocks more capacity" scaling story is UNSUPPORTED in the tested decade.
   The 10⁷⁺ regime remains empirically open (and the canon prescribes no doctrine —
   spec/sota-notes.md).
3. Scope limits: effect measured in the Adam+replay diagnostic regime on the duel
   scale; transfer to the KC-faithful production recipe and to SF-anchored strength is
   untested. The champion and leaderboard are unaffected.
