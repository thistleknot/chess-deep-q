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

## Verdict protocol (charter)

Duels mlp_v vs lin_v at v ∈ {25, 50, 100}. SUCCESS = |Elo diff| > band at v=100 (either
direction), slope read across rungs. Arms inseparable at 4× volume (|diff| ≤ band with
G3 healthy) → STOP-LOSS: capacity closed PERMANENTLY, re-diagnosis delivered.
