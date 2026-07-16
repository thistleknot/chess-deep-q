# Merge 8 — rsearch: native full-width alpha-beta + quiescence under Python

The last untested KnightCap recipe line (deep, full-width, quiescence-resolved targets) is
~1000× too slow in Python. Fix the constant factor, not the algorithm: a Rust extension
(PyO3/maturin, both now installed) holding the 809-feature linear eval and a real search.

## Scope (v1, deliberately small)

- Crate `rsearch/` (cdylib, pyo3): move generation via the `cozy-chess` crate (replicate,
  don't reinvent); NO transposition table, NO parallelism in v1.
- **Eval**: reimplement `encode_features`'s 40 donor features + 769 planes as an incremental
  dot product against weights passed in from Python at construction. MUST match
  `cem_loop.encode_features` semantics exactly (same normalizations, same indices).
- **Search**: iterative-deepening negamax alpha-beta, FULL WIDTH, depth `d` (target 4–6),
  quiescence at the horizon (captures until quiet, stand-pat), tanh applied at the leaf.
- **API**: `Searcher(weights: list[float])`;
  `searcher.search(fen: str, depth: int) -> (best_uci: str, value: float, leaf_fen: str,
  predicted_reply_uci: str | "")` — leaf FEN (not features) crosses the boundary; Python
  re-encodes with `encode_features` for training pairs (one call per move, cheap, and keeps
  the python encoder the single source of truth for training data).

## Acceptance

1. Eval parity: for 200 random positions, Rust pre-tanh score == numpy `w · encode_features`
   within 1e-4 (same weights).
2. Search sanity: finds mate-in-1/2 on test positions; depth-1 result == python 1-ply argmax.
3. Speed: ≥ 50× python beam per position at d4 (measure; target ≥ 10k nodes/s incl. eval).
4. Arm (the KnightCap-scale replication): ~300 games with d4 full-width+quiescence targets,
   faithful-mode training rules, graded+reach opponents → rungs. ALSO: 60g inference rung of
   the existing champion/kc7 eval driven by rsearch at d5 — search-side Elo readout for free.

## v2 (:Depth-per-second: — the verdict demanded it: d6 ≈ 1s/move is the scaling wall)

- **Transposition table**: Zobrist hashing (piece-square keys + side + castling + ep file),
  fixed-size 2^22-entry table, replace-always, stores (key, depth, score, bound, best move).
- **Move ordering**: TT move first, then MVV-LVA captures, then killer moves (2/ply), then
  history-scored quiets.
- **Null-move pruning**: R=2, skipped when side to move is in check or has only K+P material.
- Target: effective branching ≤ 3 → d8-d9 at ≤ 1s/move. Same API, same eval, parity tests
  and mate tests must still pass; d1 unchanged by construction.
- Deploy constraint: the .pyd is file-locked by running trainers — `cargo build` to verify,
  `maturin develop` only between runs.

## Non-goals (v2)

Parallel search, aspiration windows, LMR, NNUE-style incremental eval — next tier, only if
d8-d9 still isn't enough after the trained eval lands.
