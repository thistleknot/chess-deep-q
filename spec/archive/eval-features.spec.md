# Merge 6 — eval capacity: the linear PST must die

Grounding: [Giraffe](https://arxiv.org/abs/1509.01549) (representation was the lever: piece
lists, attack/defend maps, mobility → ~IM on one machine); [KnightCap](https://arxiv.org/abs/cs/9901002)
(each feature stage bought a strength jump); S&B `056_feature-construction`–`059_tile-coding`
(one-hot piece-square is the crudest coding; features must let V generalize across positions).
Measured motivation: every training lane plateaus at raw ~850–950 on the 769-dim linear PST —
depth amplifies its holes (session memory: depth-amplifies-holes); a PST cannot represent
mobility, king safety, or hanging pieces AT ALL.

Gate: implement AFTER rung 1 (graded TDLeaf study 30fbaead) reports — its result picks Phase A
vs Phase B first, it does not change that one of them happens.

## Phase A — capacity via hidden layer (zero new encode cost, exists behind a flag)

- `QLEARN_ARCH=mlp` (ValueNet already implements it; HIDDEN=64). What failed before (E1
  collapse at α=1e-3) is a STEP-SIZE fact, not a capacity verdict: S&B §9.6 / skill 086 →
  α range for mlp+TDLeaf trials stays [1e-4, 3e-3] with prior mean 3e-4 (already wired).
- Study identity handles it (`-mlp-h64` in PROTO). Seed: NOT resumable from the linear
  champion (shape mismatch → fresh net by design); trials run fresh under graded ladder —
  the ladder starts them at `random`/`heuristic`, which is exactly the curriculum a fresh
  net needs (fresh-net floor-flatness argument applies to SELF-play, not graded).
- Accept if: study best beats the linear graded-TDLeaf study best by >1 SE of the pooled
  objective. Then a full run + 200g rung.

## Phase B — `encode_features` (donor-reconciled: KnightCap eval.h, 577 coefficients)

Donor: [tridge/KnightCap](https://github.com/tridge/KnightCap) `eval.h`/`eval.c` — the actual
coefficient vector their TDLeaf tuned. Load-bearing groups transplanted (bitboard-computable
in python-chess, no per-square python loops); the groups we CANNOT cheaply compute (outposts,
x-rays, trapped pieces, per-bucket mobility vectors) are consciously dropped, not approximated.

- New `encode_features(board) -> float32[NFEAT]` in cem_loop.py, alongside (NOT replacing)
  `encode`; consumers take an `encode_fn` parameter (`:Compat:` below).
- Feature map (NFEAT = 769 + 40 = 809):
  1. [0:769]   the existing one-hot planes + turn (PST signal stays);
  2. [769:771] bishop pair per side (donor BISHOP_PAIR);
  3. [771:779] mobility: attacked-square counts per N/B/R/Q per side, /14 (donor I*_MOBILITY);
  4. [779:787] SAFE mobility: attacks to squares NOT attacked by the opponent, per N/B/R/Q
     per side, /14 (donor I*_SMOBILITY — their split, kept);
  5. [787:791] hung pieces: count/5 and value-sum/9 of own pieces attacked-and-undefended,
     per side (donor IHUNG_VALUE + HUNG_PIECE_FACTOR + THREAT);
  6. [791:793] king-ring attack: enemy-attacked squares in own king's ring /8, per side
     (donor KING_ATTACK_COMPUTER/OPPONENT);
  7. [793:797] castling rights K/Q per side (donor CASTLE_BONUS);
  8. [797:803] pawn structure: doubled/8, isolated/8, passed/8 per side (donor DOUBLED_PAWN,
     ISOLATED_PAWN, IPAWN_ADVANCE/UNSTOPPABLE_PAWN);
  9. [803:807] rooks on open / half-open files per side /2 (donor ROOK_ON_[HALF_]OPEN_FILE);
  10.[807:809] connected rooks per side (donor CONNECTED_ROOKS).
- Cost budget: ≤ 90µs/position ABSOLUTE (measured 81.5µs; the earlier "5×" ratio was pegged to
  a stale 15µs encode estimate — encode is actually ~9µs, and the features' Python floor is
  ~75µs of real work). Search cost impact: ~300 encodes/move → ~+25ms/move ≈ 2× generation
  slowdown, accepted. If ever blown, drop the most expensive group, not the merge.
- :Compat:: checkpoints carry `nin`; resume guards already reject shape mismatches;
  search_policy takes `encode_fn` (default `encode`); qlearn selects by `QLEARN_ENC`
  (`pst` default | `kc`) — Merge 4/5 lanes untouched.
- Pipeline: TDLeaf+graded Optuna study (fresh nets — NIN differs from champion; the graded
  ladder IS the curriculum for a fresh net) → full run → 200g rung. Accept if ≥1300 with
  d3 k16 w12 (rung-2 goal).

## Non-goals

- No NNUE incremental-update machinery (we batch-eval leaves; incrementality pays at
  engine-grade node counts, not ours).
- No board-plane CNNs / AlphaZero nets — compute budget rules them out (1712.01815).
