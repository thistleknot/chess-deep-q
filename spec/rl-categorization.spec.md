---
description: 'What kind of RL this system is — qualified classification across the standard axes, per training stage'
import:
  - learned-model
  - search-mcts
  - annealing-schedule
  - teacher-distillation
  - value-target
  - self-play-leela
---

***definitions***

- :Headline-system: is the chess agent as a whole — a learned dual-head network, PUCT/MCTS planning with the known game rules, and an annealed handoff from teacher priors to self-play.
- :Value-critic: is the learned value head V(s). No head is trained by policy gradient at any stage, so the system is never actor-critic in the A2C/PPO sense.
- :Known-model-planning: is search with the exact game rules at decision time (the "model" is the rulebook, not learned).
- :Supervised-warm-start: is the Stage-1 bootstrap (`distill_sf.py`, spec: teacher-distillation) that regresses V(s) onto a Stockfish :Teacher:'s score and the :Policy-head: onto MultiPV soft targets over generated positions, *before* any self-play. It is **supervised value/policy regression / knowledge distillation**, not RL: a fixed target oracle, MSE/cross-entropy losses, no rewards, no bootstrapping off the net's own estimate, no environment interaction during the fit.

***implementation reqs***

- Every classification below MUST be stated *with its qualification*; the bare label is misleading and is the reason this file exists.

***functional specs***

- The :Headline-system: should be described stage-wise; the closest one-line summary is **"AlphaZero/Leela-lite bootstrapped by Stockfish distillation, with every handoff annealed and Elo-gated"** — historically labeled "Deep Q-Learning", which it is not (no `max_a Q(s,a)`, no per-action value head).
- Training is **three staged phases**, and they are different learning paradigms — do not conflate them:
  - **Stage 1 — :Supervised-warm-start: (teacher-distillation).** OFFLINE, off-policy (positions from the teacher's own strong-game trajectories — the :Position-source: — sampled via temperature over its MultiPV, NOT random playouts), **supervised regression / distillation**. Not RL, not TD, not DQN/SARSA/actor-critic — no reward, no return, no bootstrap. Its only job is to make both heads competent fast (self-play from scratch cannot cross ~1200 in minutes).
  - **Stage 2 — annealed off-policy refinement (`chess_ai.py`).** ONLINE, off-policy, value-based **search-bootstrapped λ-return** (:Lambda-return:, β-annealed via :Bootstrap-share:) with replay + target net + search planning + annealed shaped reward; teacher data mixed per :Demo-share:. Off-policy-safe by **tree-backup** (the bootstrap is the search's improved on-policy value), not by importance sampling. TD(0) is the β=1 special case (`neural_network.py:274`); pure Monte-Carlo is the β→floor limit.
  - **Stage 3 — expert iteration (self-play-leela).** RL proper: **policy iteration with search as the improvement operator**. Policy trained by **MCTS visit distillation** (cross-entropy), value by the same :Lambda-return: with β at its floor (approaching Monte-Carlo outcome regression, light search bootstrap retained).
- **Explicit yes/no classification** (the axes to answer directly):

  | Axis | Stage 1 (distillation) | Stage 2 (refinement) | Stage 3 (expert iteration) |
  |---|---|---|---|
  | Reinforcement learning at all? | **No** — supervised | **Yes** | **Yes** — policy iteration via search |
  | Online / offline | **Offline** (fixed generated set) | **Online** (self-generated) | **Online** (self-play) |
  | On-policy / off-policy | Off-policy (n/a target) | **Off-policy**; bootstrap tree-backup-safe, λ-tail recency-guarded (:Replay-window:) | same; Retrace(λ) behind flag for data staler than :Replay-window: |
  | Value / policy / actor-critic | Value+policy regression | **Value-based** (V(s) critic) | Policy+value, policy via **visit distillation** |
  | Actor-critic / A2C / advantage? | **No** | **No** — no policy grad, no advantage | **No** — expert iteration, not policy gradient |
  | SARSA? | **No** | **No** — not on-policy action-value | **No** |
  | Q-learning (action-value, max_a)? | **No** | **No** — state-value V(s), no `max_a Q` | **No** |
  | DQN? | **No** | **DQN-*family* tricks only** (replay, target net) | **No** — no TD target at all in pure form |
  | Model-based? | No (static labels) | Model-free learning **+** :Known-model-planning: | Model-free learning **+** :Known-model-planning: |
  | Bootstrapping? | **No** (direct label) | **Yes** — :Lambda-return: bootstrapped from the search value (tree-backup); TD(0) is the β=1 limit | β at floor ⇒ approaches Monte-Carlo outcome z; light search bootstrap retained |
  | Reward | none (regression) | dense shaped, annealed → sparse terminal ±1 | sparse terminal z; Dirichlet noise is exploration, not reward |
  | Closest lineage | knowledge distillation | TD-Gammon / AlphaZero value learning | **AlphaZero / Leela Zero** |

- Named-algorithm mapping: the system is at no stage SARSA, PPO/A2C, or literal action-value Q-learning. The one true action-value DQN in the repo family is the separate Connect4 `value_based/dqn_connect4.py`.
