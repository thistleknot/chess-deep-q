# :Bullet-route: — self-play corpus → NNUE-small → deep search → claims (goal-1600)

Governing spec for the goal-1600 program (operator goal 2026-07-13: RL model self-plays
to 1600+ Elo). Drafted as RECONCILIATION: corpus_gen.py, nnue_eval.py, the vol1600
bullet arm, and the mlp volume lane shipped ahead of this spec (contract deviation,
operator-flagged); this document now governs; future changes are spec-first again.

## Pipeline order (operator-corrected, NEVER invert)

1. SCREEN — studies/duels at toy scale measure POTENTIAL only (3-trial Optuna,
   600-game duels). Screening can never produce a claims-grade net (studies top
   ~950–1200; the 1540 record came from a 26-epoch full run).
2. GRADUATE — the strongest screened candidate gets the full-scale run, launched by
   the agent under explicit delegation (task #32). Trainer-lane arm: 30 epochs ×
   1000 games. Bullet arm: 10× corpus + retrain (training is ~23 s; DATA is the
   commitment axis).
3. VIEW — the graduated run's final Elo is the operator's deliverable.
4. CLAIM — the claims ladder runs on the GRADUATED net: 60g deep rung → 200g claims
   vs SF@1320; goal closes at 95% CI floor ≥ 1600.

## Definitions

- **Corpus** = bullet text format `FEN | score | result`, one line per TDLeaf SEARCH
  LEAF from native self-play (rsearch4.play_games, both sides = generator net,
  d2/d2, ε=0.1, ply cap 160). score = white-relative cp from the generator's own
  backed search value via cp = round(800·atanh(v)) (declared scale 400:
  sigmoid(cp/400) ≡ (v+1)/2), clip ±3000. result = (z+1)/2, z White-absolute.
- **Filter** (council verdict, data/council.md): drop in-check leaves (measured 9.6%).
- **Arm 1 (vol1600)** = jw1912/bullet stock simple.rs recipe replicated whole:
  (768→128)×2 dual-perspective SCReLU, AdamW, SCALE 400, WDL 0.75, StepLR. Declared
  departures: our data; steps scaled so one superbatch ≈ one corpus epoch.
- **Arm 2 (control)** = bullet-linear on the same corpus (pending; verdicts for any
  bullet net are RELATIVE to this control — regime confound rule).
- **Arm 3 (kingbucket)** = factorised input buckets (bullet progression example 3 /
  factorised.rs) — parked until arms 1–2 read out.
- **Evaluator** = nnue_eval.NNUEEval over the checkpoint's raw.bin (f32, SavedFormat
  order l0w/l0b/l1w/l1b; Chess768 map, stm-relative dual accumulators, SCReLU
  = clamp(0,1)²; value() returns White-absolute sigmoid-input units).

## Contracts

- **Require**: generator net = purity-clean self-play lineage (volume net: kc-809
  linear, ZCA back-conversion identity-gated); corpus fields all self-generated
  (purity law — SF is opponent/anchor only, never a label source).
- **Guarantee**: every trained arm is measured by the validated H2H instrument
  (spec/h2h-instrument.spec.md gates G1/G2b/G3) before any absolute rung is bought.
- **Maintain**: <15-min interactive gates; anything longer is a background lane;
  replicate-before-invent for every bullet recipe change (stock first, single-variable
  departures, declared).
- **Assert (gates, all measured 2026-07-13 for arm 1)**:
  1. ZCA identity gate: whitened-net value == raw-weight value on ≥3 positions (PASS,
     models/zca.npz).
  2. Corpus gate: ≥3 result classes, legal FENs, sane cp spread, decisive rate
     reported (PASS: 1.34M positions, 54% decisive games).
  3. Evaluator convention gate: corpus-label correlation split by side-to-move, both
     halves r ≥ 0.5 (PASS: +0.74/+0.57); sign agreement ≥ 80% at |cp|>300 (PASS:
     98.3%). OOD hand batteries are NOT valid gates for nets trained on narrow
     self-play distributions (pinned: 3-queen battery false-alarmed).
  4. Duel gate: arm vs TEACHER (its label source) 600g — student ≥ teacher-band =
     distillation sound; student > teacher beyond band = capacity gain at this data
     scale. (In flight: h2h nnue_vs_teacher.)

## Known data caveats (carried, not hidden)

- Corpus result skew: White 9% / draw 63% / Black 28% (ε-handicapped agent vs clean
  opponent asymmetry) — the net's priors inherit it (startpos value ≈ −0.58 in
  sigmoid units). Rebalance is an arm-2+ dial, not a silent patch.
- Generator strength (≈1500) unisolated in literature — mitigated by teacher-relative
  and control-relative verdicts only.

## Acceptance

Bullet route graduates to step 2 (10× corpus) only if a bullet arm beats BOTH its
teacher and the bullet-linear control beyond the duel band. Otherwise the graduated
run goes to the best trainer-lane candidate (mlp volume lane vs volume-net incumbent).
