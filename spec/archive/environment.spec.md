---
description: 'Merge 0 — the chess MDP scaffold: board state, random legal-move policy, episode rollout, terminal reward. No learning. The reviewable base every RL rung builds on.'
import: []
---

***definitions***

- :Chess-env: is the environment and the ONLY owner of state transitions and reward. Interface mirrors the classic RL env (`reference/all-rl-algorithms/01_simple_rl.ipynb` `GridEnvironmentSimple`): `reset() -> board` and `step(move) -> (board, reward, done)`. Transitions are deterministic and the model is known (the game rules), which is what lets a later rung score afterstates directly.
- :State: is a `chess.Board`. The env is deliberately AGNOSTIC to how a learning rung encodes the board (raw board, hand-features, planes) — encoding is a learning-rung concern, not the env's.
- :Action: is one legal move from the current board's `legal_moves`.
- :Random-policy: is uniform choice over the legal moves. It is Merge 0's entire behavior policy AND the exploration floor every later rung starts from (ε = 1 → decay). "Start with the random" is literal: rung 0 plays only this.
- :Terminal-reward: is the white-absolute game score z ∈ {+1 white delivers mate, −1 black delivers mate, 0 draw}, and 0 at every non-terminal step. Draw covers stalemate, insufficient material, threefold/fivefold repetition, the 75-move rule, AND the :Ply-cap:. Reward comes ONLY from the game result — no evaluator, no search, no Stockfish.
- :Ply-cap: is a hard cap on plies per episode (default 200) that ends the game as a draw. It guards against the non-terminating wander that random play produces, so episodes always return.
- :Optimistic-start: is the standing exploration contract the first LEARNING rung inherits (Sutton & Barto §2.6): initialize value estimates artificially HIGH and start ε = 1, so every untried position looks attractive and the agent explores broadly before it exploits. Declared here so Merge 0's defaults are set up for it; the value machinery itself lands in Merge 1.

***implementation reqs***

- `:Chess-env:` lives in `chess_rl.py` as `ChessEnv`. It defines `reset`, `step`, and the terminal-score rule, and NOTHING else — no policy, no value, no learning. A learning rung imports it and adds its own policy/value on top; a rung MUST NOT change reward semantics.
- Config defaults (episode count, ε schedule, :Ply-cap:, learning rate) are module-level constants overridable by argv/env, so a run can be dialed without editing code.
- Merge discipline: each RL algorithm is its own spec + its own reviewable change on top of this env. No rung folds two algorithms into one merge. The former all-in-one system is archived under `spec/archive/` and is NOT imported.

***functional specs***

- Given a fresh episode, When `reset()` is called, Then the board is the standard start position and the ply counter is 0.
  - The returned object is the live `:State:`; the caller reads legal moves from it.
- Given a legal move, When `step(move)` is called, Then the board advances exactly one ply and `reward` is 0 unless the resulting position is terminal or the ply count has reached the :Ply-cap:.
- Given a terminal position (or the :Ply-cap: reached), When `step` returns, Then `done` is True and `reward` equals the :Terminal-reward: (white-absolute), and the episode ends.
- Given ε = 1 (Merge 0), Then move selection is the uniform :Random-policy: over legal moves. A later rung lowers ε over episodes toward a small floor; it never removes the floor entirely.
- Given that chess positions essentially never recur within or across episodes, Then a tabular value/Q table cannot learn on `:Chess-env:` — it would never revisit a key and would stay at its initial estimate. THEREFORE the first learning rung MUST use a generalizing function over :State: features, not a lookup table. This is the load-bearing constraint that rules out literal tabular Q-learning and fixes the LCD as function-approximation TD/MC from the very first learning merge.
