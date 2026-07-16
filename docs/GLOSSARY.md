# Glossary — chess RL from first principles

Plain definitions for the terms used across this repo's RL build. Grouped by topic, not alphabetical, so it
reads as a learning path. `[cited]` marks a claim a specific source backs; repo pointers show where the idea
lives in code. Companion to `spec/README.md` (the merge ladder) and `reference/all-rl-algorithms/` (the
algorithm notebooks). Primary texts: **S&B** = Sutton & Barto, *Reinforcement Learning: An Introduction*
(2018); **DRLiA** = Zai & Brown, *Deep Reinforcement Learning in Action*.

---

## 1. The RL problem (the vocabulary of the MDP)

- **MDP (Markov Decision Process)** — the formal frame for RL: states, actions, a transition rule, and a
  reward. "Markov" = the next state depends only on the current state + action, not the whole history. Chess
  is an MDP (the board is a sufficient state).
- **State (s)** — the situation the agent is in. Here: a `chess.Board`. See `chess_rl.py`.
- **Action (a)** — a choice the agent makes. Here: a legal move.
- **Reward (r)** — the scalar feedback. Here: 0 every move, then a terminal ±1 (white mate / black mate) or 0
  (draw). Reward comes ONLY from the game result — no hand-tuned shaping. `ChessEnv._terminal_z`.
- **Policy (π)** — the agent's rule for choosing an action given a state. Can be random, ε-greedy, or greedy.
- **Return (G)** — the total (optionally discounted) reward from a state to the end of the episode. What the
  value function tries to predict.
- **Discount (γ, gamma)** — how much future reward is worth vs immediate (0.9–0.99 typical). γ<1 makes a
  *faster* mate worth more than a slow one.
- **Episode / trajectory** — one full game from start to terminal.
- **Value function V(s)** — expected return from state s under a policy. "How good is this position?"
- **Q-function Q(s,a)** — expected return from taking action a in state s. Value *of a move*, not just a
  position.
- **Afterstate value** — value of the position *after* a move is made. Chess has a known model (you know the
  resulting board), so scoring afterstates with V and taking the best sidesteps needing a Q-head over ~4672
  actions. This repo uses afterstate V. `spec/q-learning.spec.md :Afterstate-value:`.
- **White-absolute frame** — this repo's sign convention: value/reward are always from White's point of view
  (+ good for White, − good for Black), regardless of whose turn it is. Keeps one consistent sign everywhere.

## 2. How agents learn (methods, simplest → richer)

- **Monte Carlo (MC)** — learn by averaging *actual* returns over complete episodes. No bootstrapping; waits
  for the game to end. λ=1 limit of TD(λ). Merge 1 used this. `value_targets.mc_return`.
- **Bootstrapping** — updating an estimate using another estimate instead of waiting for the final outcome:
  `V(s) ← r + γV(s′)`. The core trick of TD/Q-learning; it assigns credit move-by-move instead of smearing
  one terminal reward over 60 moves.
- **TD(0) (Temporal Difference)** — one-step bootstrap: target = `r + γV(s′)`. [S&B §6]. `value_targets.td0`.
- **TD(λ) / eligibility traces** — a dial between TD(0) (λ=0) and Monte Carlo (λ=1). λ blends short bootstraps
  and long real returns. [S&B §12]. `value_targets.lambda_return`. The α *and* λ knobs in the dashboard.
- **Q-learning** — off-policy TD control: `Q(s,a) ← Q(s,a) + α[r + γ·maxₐ′Q(s′,a′) − Q(s,a)]`. The `max`
  learns the *optimal* policy regardless of how the data was generated. [S&B §6.5, p.131]. Merge 2.
  Converges to optimal w.p.1 if every state-action pair keeps being visited and α shrinks over time.
- **SARSA** — the on-policy sibling of Q-learning: bootstraps off the action *actually taken* next, not the
  max. More cautious. [S&B §6.4, p.130].
- **On-policy vs off-policy** — *on-policy* learns the value of the policy it's currently following (SARSA,
  REINFORCE, PPO) → can't freely reuse old data. *Off-policy* learns about the optimal/greedy policy while
  behaving differently (Q-learning, DQN) → CAN reuse old data (this is why a replay buffer is allowed).
  [cheatsheet.md; S&B §5–6].
- **Cross-Entropy Method (CEM)** — keep the "elite" episodes (highest return / here: the games that
  checkmated), train the policy to imitate them, regenerate, repeat. Simple self-improvement. Merge 1 used
  this; it draw-collapsed. [cited — Szita & Lőrincz 2006; opening algo in Lapan's *Deep RL Hands-On*].
- **REINFORCE / policy gradient** — learn a *policy* directly by pushing up the probability of actions from
  high-return episodes (vs learning a value). On-policy, high variance. `reference/…/06_reinforce.ipynb`.
- **Generalized Policy Iteration (GPI)** — the generate→evaluate→improve→regenerate cycle underlying almost
  all RL. The "loop" the dashboard runs.
- **Expert iteration** — improve a policy by distilling a stronger search (e.g. MCTS visit counts) into it,
  then searching again. AlphaZero's loop. (Archived system used this; see `spec/archive/`.)
- **DAgger (Dataset Aggregation)** — run the current agent, have an expert (Stockfish) label the states it
  actually visits, add to the dataset. Fixes distribution shift. (Used in the archived distillation work.)

## 3. Exploration (how the agent tries new things)

- **ε-greedy (epsilon-greedy)** — with probability ε pick a random move (explore), otherwise pick the best
  known (exploit). ε starts at 1 (all random) and decays toward a floor (never 0).
- **Optimistic initialization** — set value estimates artificially HIGH so every untried option looks good,
  driving broad early exploration. [S&B §2.6, p.34]. `spec/environment.spec.md :Optimistic-start:`.
- **Dirichlet noise / temperature (τ)** — AlphaZero's exploration recipe at the search root (noise) and in
  move sampling (temperature), instead of uniform ε-greedy. [cited — Silver et al. 2017]. Not used in the
  current simple merges.

## 4. Deep-RL training machinery

- **Function approximation** — using a learned function (linear model or neural net) to generalize value
  across states, instead of a lookup table. Required for chess (states never repeat, so a table never gets a
  second visit). Note: it voids Q-learning's tabular convergence *theorem* → convergence becomes empirical.
- **Experience replay / replay buffer** — store transitions `(s,a,r,s′)` and train on random past samples.
  Two payoffs: sample efficiency (reuse each game) and stability (random sampling breaks the correlation of
  consecutive moves). [DRLiA §3.3, pp.75–76]. `qlearn.py ReplayBuffer`.
- **Catastrophic forgetting** — in online training, consecutive correlated updates overwrite each other so
  the model never converges. The problem replay solves. [DRLiA §3.3.1].
- **Target network** — a periodically-frozen copy of the value net used to compute the bootstrap target, so
  the target doesn't chase itself. Stabilizes DQN. [DRLiA §3.4]. (Optional in `qlearn.py`; off by default.)
- **Prioritized Experience Replay (PER)** — sample transitions by `|TD-error|` ("how surprising / how much
  to learn") rather than uniformly. A *speed* optimization, NOT required for convergence. [cited — Schaul et
  al. 2015/2016; tabular ancestor "prioritized sweeping," S&B §8.4].

## 5. Chess-engine terms

- **PST (Piece-Square Table)** — a per-(piece-type, square) positional bonus added to material (knight in the
  center = good, etc.). `engine.py _PST` / `pst_eval` (material + PST, white-absolute centipawns). A *linear*
  value over the piece-square one-hot **is** a learnable PST.
- **Material** — raw point count of pieces (P=1, N/B=3, R=5, Q=9). The crudest eval.
- **Centipawns (cp)** — eval unit: 100 cp = one pawn of advantage.
- **Alpha-beta** — classical game-tree search with pruning; the repo's ~1672-strength `AlphaBetaEngine` uses
  it. Strong but NOT RL — a baseline to surpass, never the deliverable.
- **MCTS / PUCT** — Monte Carlo Tree Search; PUCT is its AlphaZero variant balancing a policy prior against
  exploration. `reference/…/17_mcts.ipynb`; archived `spec/archive/search-mcts.spec.md`.
- **NNUE** — "Efficiently Updatable Neural Network," a fast incrementally-computed chess eval. The archived
  distillation work trained one (`nnue_model.py`).
- **Stockfish (SF)** — the strong open-source engine used here as the labeling oracle and the Elo anchor.
  `engines/stockfish/…exe`.

## 6. Measuring strength

- **Elo** — a relative rating; a 400-point gap ≈ the stronger player scores ~10×. Two players' Elo difference
  maps to an expected score.
- **elo_diff(s)** — score→Elo conversion: `400·log10(s/(1−s))`. `measure_ladder.py:29`.
- **SF anchor / calibrated Elo** — Stockfish limited to `UCI_Elo=1320` is a fixed reference; the agent's Elo
  ≈ `1320 + elo_diff(score vs SF@1320)`. The one calibrated yardstick. `qlearn_eval.calibrated_elo`.
- **Proxy strength** — cheap, instant strength signals used every iteration: win-rate vs the 1-ply PST mover
  and vs a random mover. `qlearn_eval.proxy_strength`.

## 7. Hyperparameter search

- **Hyperparameter** — a setting you choose *before* training (α, γ, λ, ε schedule), as opposed to weights the
  model learns.
- **Grid / random search** — try every combination (blows up) / sample blindly (never learns). The baselines
  TPE beats.
- **Optuna** — the Python hyperparameter-optimization framework used here. `tune_qlearn.py`.
- **TPE (Tree-structured Parzen Estimator)** — Optuna's default Bayesian sampler. It splits past trials into
  "good" and "bad," fits a density over the hyperparameters in each (`l(x)` good, `g(x)` bad), and samples
  next where `l(x)/g(x)` is highest — i.e. settings common in winners, rare in losers (∝ Expected
  Improvement). First ~10 trials are random to seed it; then it exploits. "Tree-structured" = handles
  conditional params. [cited — Bergstra et al. 2011, NeurIPS].
- **Objective noise** — the run-to-run variance of a trial's score. You size a trial's game count so the
  effect between hyperparameter settings exceeds this noise (why `tune_lambda`/`tune_qlearn` pick their game
  counts, NOT the checkmate rate).

## 8. This project's own terms

- **Merge ladder** — the from-scratch plan: Merge 0 = env + random baseline; Merge 1 = CEM/filtered
  self-imitation (falsified); Merge 2 = afterstate TD(λ) Q-learning + dashboard. `spec/README.md`.
- **Eval holes** — positions a learned eval scores falsely-high because training never showed them. The
  binding constraint in the archived distillation work: depth *amplifies* holes, coverage *fills* them.
- **Draw-collapse** — Merge 1's failure: a greedy value trained on outcomes steers toward draws, draws get
  filtered out of the buffer, so the loop starves its own training signal. Fixed by Merge 2 (keep all games,
  credit per move).
- **V^random** — the value of a position *under random continuation*. Nearly useless (close to noisy
  material), because a random game's outcome barely reflects the position's true value. Why Merge 1's
  outcome-labels were weak.
