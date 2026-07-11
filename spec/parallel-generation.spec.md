# Merge 9 — parallel generation: ε for coverage, enumeration for the min, λ for the horizon

Decision (operator asked for the figured-out answer, 2026-07-10): the plateau decomposes into
three problems with non-substitutable tools —
1. STATE COVERAGE is undersampled at ~10³ games → ε-greedy on OUR moves + raw game volume
   (sampling fixes variance);
2. ADVERSARIAL RESOLUTION is a min-operator → computed exactly by full-width d2 + quiescence
   (~1ms native); sampling opponent replies estimates the ε-smoothed MEAN, which is
   optimistically biased by exactly the blunder mass and gets WORSE with more ε (measured:
   every trajectory-target arm plateaued ~880; KnightCap's RAMP discards favorable samples
   for this reason);
3. LONG-HORIZON CREDIT → λ traces over real trajectories (already faithful, λ≈0.85–0.9 here
   since the local min is exact and cheap).

## rsearch v3: `play_games(weights, bias, n_games, threads, depth, eps, opp_mix, ply_cap)`

- Plays `n_games` CONCURRENTLY (std::thread pool over the machine's cores) fully native.
- Behavior: OUR moves = argmax of full-width depth-`depth`+qsearch with prob 1-ε, uniform
  random legal with prob ε (coverage). Per-move record: (leaf_fen, white_value, pred_reply,
  predicted_flag) — same TDLeaf semantics as the faithful trainer.
- Opponents per game drawn from `opp_mix`: "self" | "random" | "heuristic" (native 1-ply PST
  mover ported from measure_ladder semantics). SF-rung games remain python-side (UCI processes
  don't belong in the thread pool); reach/diet composition handled by the python caller.
- Returns per game: (result z, [per-OUR-move records], agent_white). Python re-encodes leaf
  FENs with encode_features (single source of truth) and feeds the EXISTING build_targets
  (RAMP-capable) + online/batch update paths.
- Target: ≥ 20 games/s aggregate at d2 on the workstation (vs ~0.3/s python serial).

## Trainer integration (qlearn.py)

- `QLEARN_PARGEN` (games per parallel batch, 0=off): generation loop swaps per-game play_game
  for one native batch per training cycle; everything downstream (RAMP, λ-returns, ratchet,
  :Confirmed-crown:, purist rungs) unchanged.

## Acceptance

1. Determinism-free sanity: batch of 64 games returns legal records (python-chess validates
   every leaf FEN + reply move), outcomes ∈ {-1,0,1}, records only for agent moves.
2. Throughput: measure games/s at d2, 16 threads; must beat serial python ≥ 30×.
3. Arm (the hybrid): d2-exact targets, λ=0.9, ε=0.1, mix self/heuristic/random + python-side
   SF reach games; equal wall-clock vs the d4 incumbent; METRIC OF RECORD = purist 1-ply rung
   (spec/expectations.spec.md); the lane gap tracks compression debt.

## Non-goals

GPU batching (the linear eval is ~1µs native — the GPU adds latency, not throughput; revisit
only with a real net); SF engines inside the thread pool; tournament self-play leagues.
