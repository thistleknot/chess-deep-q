---
description: 'Merge 1 — the self-improvement loop: generate games, keep only decisive (checkmate-under-cap) ones in a buffer, fit a value on those clean ±1 outcomes, act greedily, regenerate. CEM-style policy iteration.'
import:
  - environment
---

***definitions***

- :Self-improvement-loop: is the iterate-and-measure cycle over :Chess-env: (Merge 0): (1) generate G games with the current :Policy:; (2) keep only games that reached checkmate within the :Turn-cap: into the :Buffer:; (3) fit :Value: on the buffer; (4) lower ε and regenerate. It is the least-common-denominator SELF-IMPROVING RL — no bootstrapping, no policy gradient — just "keep the games that won, learn to value their positions, play by that value, repeat." Algorithm family: Cross-Entropy Method (the decisive games are the elite set), generalized policy iteration, self-imitation / filtered behavioral cloning.
- :Buffer: is the accumulated set of (board-encoding, z) pairs drawn ONLY from decisive games (checkmate within :Turn-cap:), z the white-absolute outcome ∈ {+1, −1}. Draws are discarded, so EVERY label is ±1. This is the load-bearing reason filtering matters: it removes the draw-dominated z = 0 mush (~89% of random games) that leaves a value with nothing to separate. The buffer accumulates across iterations (retained off-policy experience) — the "buffer" the loop is named for.
- :Value: is V(board) computed from the RAW board state only — a piece-square one-hot encoding, the least-engineered board representation, with NO hand-crafted eval features — fit by regression toward the :Buffer:'s ±1 labels. A linear V over the piece-square one-hot is exactly a learnable piece-square table.
- :Policy: is ε-greedy over afterstate :Value: (white maximizes, black minimizes the white-absolute V), reusing :Chess-env:'s known deterministic transitions. ε starts at 1 (:Optimistic-start: — pure random, the Merge 0 baseline) and decays toward a floor across iterations, never to 0.
- :Turn-cap: is the decisive-game filter: only games checkmating within N full moves (default 60) are kept. Shorter decisive games carry cleaner credit — fewer random plies between an early position and the terminal result.
- :Checkmate-rate-signal: is the loop's leading convergence indicator: the fraction of each iteration's G games that reach checkmate within the :Turn-cap:. THE load-bearing falsifiable hypothesis of this rung: it RISES across iterations as the improving policy steers into mate more often. Flat ⇒ the rung is walled (random-mate labels too noisy to steer from); rising ⇒ self-improvement is real, measured not assumed.

***implementation reqs***

- The loop imports :Chess-env: from Merge 0 UNCHANGED and adds :Value:, :Policy:, :Buffer:, and the iteration controller in its own module. It MUST NOT modify env reward semantics.
- Per iteration, report: :Checkmate-rate-signal:, :Buffer: size, training loss on the buffer, and a cheap strength proxy (win-rate of the greedy policy vs the random policy over a fixed sample). Elo against the Stockfish anchor is a separate, heavier measure — not run every iteration.
- Only decisive (checkmate-under-cap) games enter the :Buffer:; draws and cap-outs are counted in the :Checkmate-rate-signal: denominator but discarded from training.

***functional specs***

- Given iteration 0 with ε = 1, Then generation is pure random play (the Merge 0 baseline) and the :Buffer: seeds from that iteration's decisive games only.
- Given a completed iteration, When :Value: has been fit on the :Buffer:, Then ε decreases and the next iteration regenerates with the improved :Policy:.
- Given a decisive game (checkmate within :Turn-cap:), When it is added to the :Buffer:, Then every one of its positions is labeled with the SINGLE white-absolute outcome z ∈ {+1, −1}.
- Given a drawn or capped game, Then it contributes to the :Checkmate-rate-signal: denominator but adds NOTHING to the :Buffer:.
- Given the :Checkmate-rate-signal: across iterations, Then a rising trend is the acceptance evidence that the loop learns; a flat trend falsifies self-improvement at this rung and motivates the next (bootstrapping / near-terminal credit assignment).
