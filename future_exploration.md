# Future exploration — parked 2026-07-15 on the amap-champion win

State at rest: **champion = the operator's amap-897 feature net** (d9 scout
49W-1D-0L/50, floor 1724 > gate 1605; commit 9578615 on iter-search-first).
Campaign law learned: representation beats algorithm — features 3-for-3
confirmed, mechanisms 0-for-7. Resume by priority order below.

## 1. kc ⊕ amap hybrid — the highest-prior untested cell
937 dims = kc-809 (KnightCap terms: king safety, pawn structure, mobility —
the OLD champion's edge) ⊕ attack-maps-128 (the NEW champion's edge). Both
blocks independently confirmed winners; both already computed inside the Rust
eval, so the port is ~20 lines (new weight-length mode in rsearch4, encoder in
encoders.py, seed, console option). Pipeline: screening duel vs amap control
(15-min train + 3-min sharded duel) → if band excludes zero, seed-confirm →
champion-recipe school run (~2.5 h) → pooled d9 scout → gate at floor 1724.

## 2. Stronger anchor rungs — instrument before strength
SF@1320 saturates above ~1700 (floors pile up at score>0.9); wins beyond the
current champion are statistically invisible. Add SF@1500 and SF@1700 rungs to
the pooled ladder (UCI_Elo settings; same 50-game law). Do this BEFORE any new
strength arm or its result can't be measured.

## 3. Deathmatch (free curiosity): old champion vs new
kc-1670 (models/champion_backup_kc1670.pt) vs amap champion, head-to-head d9,
50 games on the duel ruler — the direct band for "how much better is my net",
immune to anchor saturation. rs:<depth>: spec form in head2head.

## 4. E-queue canon features (screening tier only)
E6 coverage-scalars first (operator's retreat concept compressed: ~24 floats,
dilution-proof); then outposts/passed-pawn detail as explicit terms. Each: one
50-game screen vs the current champion encoding, kill fast per the law.

## 5. Played-buffer lane port (small defect, known)
human_replay.py still asserts enc=="kc" — port to amap/no-ZCA so "Learn from my
games" (menu 5) works against the new champion. ~15 lines.

## Parked deep doors (big lifts, only on explicit operator go)
- Native-targets mellowmax (:Backup-temperature: at scale) — Rust-side soft
  backup; its python-beam form tied, unconfirmed stacked (0/7 mechanism prior).
- Native/GPU MCTS for NONLINEAR evals — only pays if we ever move past linear
  (bullet-NNUE at 10× corpus is the companion door).
- d10+ ladder on the amap champion (depth-amplification check at the new eval).

## Standing rules that govern any resumption
20g explore / 50g full measure (H2H_CAP); screens kill, never confirm; one
factor per cell + round-robin standings for combinations; everything through
lane.py (registry, conflict guard, measured ETA — quote ETAs from `lane.py
eta`, never intuition); instrument tests never write to rl_trend; thermal guard
stays on (75% cores, turbo 90%).
