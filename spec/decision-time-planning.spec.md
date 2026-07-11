# Merge 4 — Decision-Time Planning over the Learned Critic (S&B §8.8–8.9)

Motivation: the critic learns (piece values converge) but the 1-ply policy can't convert knowledge
into wins — 11 paired Merge-3 experiments, all flat (data/experiments.md). Search is the converter;
this repo already measured 1428–1672 with search over a tuned eval.

***concepts (committed)***

- :Search-policy: — depth-2 negamax over the critic's White-absolute V with batched leaf evals.
  Root exploration = softmax over search values at the existing τ; measurement = argmax. Mate-in-1
  is detected exactly (every root move is pushed once); non-terminal leaves use the critic.
- :Search-width: — only the top-K root moves by 1-ply value get reply expansion (`QLEARN_SEARCH_WIDTH`,
  default 8); the rest keep their 1-ply value. Bounds cost at ~K·|replies| leaf evals per move.
- :Search-flywheel: — search-played games feed the SAME λ-return targets; critic improves → search
  improves (expert iteration). Runs under the existing anchor/gate/ratchet/lineage process unchanged.
- :Depth-hazard: — cap at d2–d3: repo history shows depth AMPLIFIES eval holes (1467@d4 → 926@d11).
  Deepen only after coverage (mate bank, curriculum) fills the holes.

***implementation reqs***

- `search_policy.py`: `move_values(board, value_fn, width)` → (moves, White-absolute values);
  `search_move(board, value_fn, tau, rng, width)` → move. `value_fn(X)` is any batched evaluator.
- `qlearn.py`: `QLEARN_BEHAVIOR=search` swaps behavior + greedy/measurement policies to search;
  bootstrap targets stay 1-ply greedy (unchanged semantics). Behavior is part of the study identity.
- Acceptance: smoke decisive-rate ≫ 0.10 and tolerable speed; study differentiates; deep run beats
  the 626 pooled baseline. Milestones: decisive >0.5 → first SF win → pooled >1000 → 1320 → 1600.
