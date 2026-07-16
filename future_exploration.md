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

## 2. dmap — destination maps — SCREENED & PARKED (2026-07-15 evening)
VERDICT: dmap vs amap **−35 (−130..+61) tie**; amapd (amap⊕dmap) vs amap
**+0 (−95..+95) dead tie** — both 50g bands span zero, no seed-confirm per
pre-registration (spec :Dmap-screen:). Mobility neither beats nor adds to
coverage at this resolution; non-pawn destinations overlap amap's attack bits
too heavily. Residual idea if ever re-opened: divergent-bits-only variant
(dmap XOR amap — pawn pushes + blocked-attack squares only). Original concept
below for the record.
**For every piece, the set of squares it is available to move into.** This is
the missing half of the coverage idea: amap says which squares we ATTACK;
dmap says where our pieces can actually GO.
Lineage: the operator raised this in the ORIGINAL coverage discussion — it was
captured compressed as E5 in the spec's E-queue (quantized legal-move COUNT per
piece-square, 2026-07-14). dmap is the uncompressed form: the destination
SQUARES themselves, not just how many. E5 stays as the dilution-proof fallback
if the maps lose on info-per-dim. They differ exactly where chess
knowledge lives — pawn pushes (moves, not attacks), pins and blockers
(attacked-but-unreachable), castling, and squares covered by an enemy piece.
Two encodings to screen:
- **dmap-128**: union of legal destinations per side (2×64), amap-symmetric —
  the direct sibling of the confirmed winner; total 897 dims again.
- **dmap-768**: destinations keyed by piece type × side (12×64) — richer, but
  dilution risk per the info-per-dim finding; screen only if 128 shows signal.
Cost note: amap was free (the eval already computes attack unions); legal
destinations need movegen at every leaf — use PSEUDO-legal destination bits
first (near-free from the same attack pass + pawn pushes), full-legal only if
the screen pays. Pipeline: same as any feature lane — 20g explore / 50g screen
vs the amap champion encoding, seed-confirm, then champion recipe. Natural
combination cell: amap ⊕ dmap (attack + mobility, 1025 dims) via the
factorial-tournament method.

## 3. Stronger anchor rungs — instrument before strength
SF@1320 saturates above ~1700 (floors pile up at score>0.9); wins beyond the
current champion are statistically invisible. Add SF@1500 and SF@1700 rungs to
the pooled ladder (UCI_Elo settings; same 50-game law). Do this BEFORE any new
strength arm or its result can't be measured.

## 4. Deathmatch (free curiosity): old champion vs new
kc-1670 (models/champion_backup_kc1670.pt) vs amap champion, head-to-head d9,
50 games on the duel ruler — the direct band for "how much better is my net",
immune to anchor saturation. rs:<depth>: spec form in head2head.

## 5. E-queue canon features (screening tier only)
E6 coverage-scalars first (operator's retreat concept compressed: ~24 floats,
dilution-proof); then outposts/passed-pawn detail as explicit terms. Each: one
50-game screen vs the current champion encoding, kill fast per the law.

## 6. Played-buffer lane port (small defect, known)
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
