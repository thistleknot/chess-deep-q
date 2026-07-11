# chess-deep-q spec — RL from first principles

A clean restart. The goal is **to understand RL by building it up one reviewable merge at a time**, not to
ship a strength number. Strength is only the yardstick; the point is that every feature is spec'd, small,
and traceable — a proper merge request, not an AI-generated pile.

**Root spec: [`environment.spec.md`](environment.spec.md)** — Merge 0, the chess MDP scaffold (board +
random legal moves + terminal reward, no learning). Every learning rung imports it and adds exactly one
algorithm.

## Run it

No setup — `python-chess` is the only dependency. From the repo root:

```bash
python chess_rl.py            # Merge 0: 2000 uniform-random games, prints the W/L/D baseline
python chess_rl.py 500 200    # args: [episodes] [ply_cap]

python measure_elo.py 20      # anchor the current agent to Stockfish@1320 -> Elo, append data/rl_trend.jsonl
python dashboard.py           # render dashboard.html (Elo, W/D/L, chart) and open it in the browser
```

Observability (`measure_elo.py` + `dashboard.py`, spec: `observability.spec.md`) is cross-cutting — pulled
forward so every rung is watchable. At Merge 0 the agent is random, so it reads ~-280 Elo (floored, score ≈0
vs SF@1320): the honest bottom the ladder climbs from.

Merge 0 has no agent to run — it plays random moves and reports the floor every learning rung must beat
(~W 5% / L 6% / D 89%). The learning rungs (`chess_rl.py` will grow an agent at Merge 1) come next.

> The repo-root `README.md` documents the **archived** system (the old distillation/menu engine), not this
> restart. This file is the entry point for the from-scratch RL work.

## The merge ladder

Each rung is its own spec + its own PR-sized change on top of the previous one. We do not skip rungs and we
do not fold two algorithms into one merge.

| Merge | Spec | What it adds | Learns? |
|-------|------|--------------|---------|
| **0** | `environment.spec.md` | `ChessEnv`: reset/step, random legal-move policy, white-absolute terminal reward, ply cap | no — the scaffold |
| **1** | `self-improvement-loop.spec.md` | Cross-Entropy Method / policy iteration: keep only decisive (checkmate-under-cap) games in a buffer → clean ±1 labels → fit value on raw board state → act greedily → regenerate. Watch checkmate-rate rise. | yes (CEM, no bootstrap) |
| **2** | `q-learning.spec.md` | afterstate TD(λ) Q-learning: keep all games, credit each move by `r+γV(next)` (λ=0 ⇒ one-step Q-learning), uniform replay buffer, live dashboard + SF-anchored Elo. Fixes Merge 1's draw-collapse. | yes (bootstrap) |
| 3+ | _(later)_ | n-step / TD(λ), then policy-gradient (REINFORCE → PPO/GRPO), per `reference/all-rl-algorithms` | — |

## Why not tabular Q-learning directly

Chess states essentially never repeat, so a tabular Q-table never revisits a key and never learns — it stays
at its initial value forever. So the least-common-denominator that actually learns here is **function-
approximation** TD/MC (a value function over features), not a lookup table. This constraint is pinned in
`environment.spec.md` functional specs and shapes every rung.

## Reference

`reference/all-rl-algorithms/` (cloned) — the algorithm-by-algorithm notebooks we climb: `01_simple_rl` →
`02_q_learning` → `03_sarsa` → `06_reinforce` → … plus `cheatsheet.md`. Paired with *Deep Reinforcement
Learning in Action* (ch. 2 bandits/optimistic-init, ch. 3 DQN).

## Archive

The former all-in-one system (17 specs: distillation, PUCT/MCTS, NNUE, self-play, Elo calibration) is
preserved under [`archive/`](archive/) with git history intact. It is **not imported** and does not govern
new work — it's reference for what was tried, not a spec to extend.
