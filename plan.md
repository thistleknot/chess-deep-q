# Disposition backlog — COMPLETE (2026-07-22 sweep, ≤2 cores)

All P1–P7 dispositioned. Verdict-first, differentiating evidence, better-performer named.

## CLOSED (do not re-test)
- Sparse coding (Kanerva) beats hand-features → **NO** — parked vs amap.
- Hand-features (amap) load-bearing as a set → **YES** — champion.
- Search-distillation of own d5-search into the linear eval beats the champion → **YES, +240 H2H** (promoted).
- Expert iteration compounds unbounded → **NO** — converges (A2≈A3≈A4).
- SWA averaging → **NO** — dilutes.
- Capacity/nonlinear pays (old) → small YES (+41); (P2, decisive) → **NO** at matched depth.
- vs-SF ladder trustworthy for strong agents → **NO** — draw-floods (H2H is the ruler).
- DQN target network needed → **NO** — frozen-teacher distillation already is one.

## P1–P7 (tonight)
- **P1 deeper labels → NO.** d5→d7 gain collapsed +232→+47 (n.s.); linear search-consistent by d7.
- **P2 more capacity → NO.** halfKP NNUE lost 0.050 (−512 Elo, band excl 0.5) vs linear at matched d4.
- **P3 native NNUE port → GATED OUT.** Won't port an eval that loses to the linear.
- **→ EVAL AXIS CLOSED.** Distilled linear amap champion is the endpoint (label depth + capacity both fail).
- **P4 absolute Elo → 1840 (1721..1958)** adjudicated + SF@2500. Confirms the ladder can't see the +240 H2H gain (both champions ~1840 vs SF); H2H is the eval-improvement ruler.
- **P5 feature audit → 86/897 beat the null**; coverage-maps survive at ~10× the PST rate (53/128 vs 33/769) — explains why amap won. Diagnostic only.
- **P6 search-arms (MCTS×minimax-λ) → NO/parked** (prior: 0-16 vs native-d9 champion; alpha-beta wins).
- **P7 Prism → not a strength lever** (warm-start dominates init; Mod-Wheel fits only the human-games lane).

## Net
Champion (distillA2, search-distilled linear amap; ~1840 absolute / +240 H2H over predecessor) stands
as the endpoint on every axis tested tonight. Remaining real levers need MORE COMPUTE (deeper-than-
deployment labeling at scale) or a different modality — not feasible at ≤2 cores. Full evidence:
spec/dispositioned.md, data/search_distill_campaign.md.
