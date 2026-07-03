## Syllogism

**Most plausible throughline:** chess is a sparse-reward, high-branching domain, so pure from-scratch RL is usually too inefficient. A fixed evaluator or scripted prior can get the agent into the space of sane play. But a fixed prior also caps discovery if it remains dominant. Therefore the practical recipe is: initialize with a prior, learn from outcomes, preserve controlled exploration, and progressively hand control to the learned policy, optionally with search on top.

---

# Crystallized skill: applying RL to a system such as chess

## Question

How should you apply reinforcement learning to a structured strategy system like chess when you already have a known board evaluator, heuristic policy, or engine-like prior?

## Findings

1. **Do not start with pure random play unless you have extreme compute.**
   In chess-like domains, sparse terminal rewards make unguided RL painfully inefficient.

2. **Use the prior in one of three places.**
   - **Policy initialization**: imitate the prior's move choices or use it to bias early action selection.
   - **Reward shaping**: convert board-evaluation deltas into intermediate reward.
   - **Search bias**: use the prior to steer search or rollout allocation.

3. **Keep the prior and the learned policy conceptually separate.**
   - **Prior policy/value** = fixed heuristic, engine score, or scripted baseline
   - **Learned policy/value** = model being updated from experience

4. **Annealing means reducing prior control, not increasing randomness.**
   Early on, the prior keeps the agent from nonsense. Later, the learned policy should take over. Exploration remains, but it should increasingly happen around the learned policy, not around the prior.

5. **Early training is often effectively off-policy.**
   If behavior comes from a prior, replay buffer, demonstrations, or mixed controller, the learner is consuming trajectories not generated solely by its current policy.

6. **If you never reduce prior influence, you can trap the model in heuristic local maxima.**
   Example: an evaluator that overvalues material may suppress long-horizon sacrifices or positional compensation.

7. **Exploration should be structured, not uniform if possible.**
   Better options than blind random legal moves:
   - temperature-scaled sampling
   - noise over policy probabilities
   - search-based exploration
   - occasional randomization in controlled phases such as openings

8. **Hybrid systems are usually the practical answer.**
   For chess, pure RL is rarely the cheapest path. Stronger setups usually combine:
   - supervised bootstrapping or heuristic bootstrapping
   - RL/self-play improvement
   - search
   - value learning

---

## Digest page

### 1. Question

How do you apply RL to chess when you already have a board-scoring function or heuristic policy?

### 2. Findings

- **A prior is useful because pure RL in chess is sample-inefficient.**
- **The prior can act as teacher, reward source, or search guide.**
- **The learned model must eventually outrun the prior, so prior influence should decay over training.**
- **Annealing the prior does not imply more random play; it implies more reliance on the learned policy.**
- **Exploration still remains necessary after annealing starts.**
- **Early mixed-policy training is naturally off-policy.**
- **A practical system is staged: bootstrap -> explore -> self-play refine -> optionally search-augment.**

### 3. Entities involved

- **Chess environment**
- **Board evaluation function**
- **Prior policy**
- **Learned policy network**
- **Learned value network**
- **Replay buffer**
- **Exploration mechanism**
- **Self-play loop**
- **Search layer, e.g. MCTS**
- **Training schedule / annealing schedule**

### 4. Lessons

- **Use priors to eliminate stupid early behavior, not to freeze the final policy.**
- **Separate "what generated the move" from "what gets updated."**
- **If the prior remains load-bearing too long, the learner will inherit its blind spots.**
- **Exploration should move from broad and guided to narrow and policy-centered.**
- **Dense reward from evaluator deltas can accelerate learning, but it can also distort objectives if it replaces win probability rather than supporting it.**
- **In chess, search is often a multiplier on learned policy/value, not a substitute for them.**
- **If your training keeps repeating the same mistakes, revisit the data-generation policy rather than tuning the optimizer forever.**

### 5. Open questions

- How strong is the available prior: simple heuristic, classical engine, or learned evaluator?
- Is the target **strong play**, **fast learning**, **low compute**, or **interpretability**?
- Will the system act without search at inference time, or is search allowed?
- Should the objective optimize **game result only** or **game result plus style or efficiency**?
- What is the failure tolerance for reward shaping bias?

---

# Reusable skill protocol

## When to use

Use this when:

- you have a complex decision system like chess
- sparse win/loss reward is too weak for efficient learning
- you already possess a heuristic evaluator, legacy engine, or human/expert demonstrations

Do **not** use this exact pattern when:

- the prior is known to be badly misaligned with the true objective
- the action space is small enough that from-scratch RL is cheap
- search alone already solves the problem within your compute budget

---

## Procedure

### Step 1: define the role of the prior

Pick one or more:

- **Policy prior**
  - sample or imitate good moves from the prior
- **Value prior**
  - use evaluator scores as training targets or auxiliary targets
- **Reward shaping source**
  - reward improvements in board evaluation
- **Search prior**
  - bias which branches to explore first

### Step 2: keep interfaces separate

Maintain separate modules for:

- **behavior policy**: what chooses actions during training
- **learned policy/value**: what gets optimized
- **prior evaluator/policy**: fixed source of guidance

This avoids confusing “greedy from the prior” with “greedy from the model.”

### Step 3: bootstrap competence

Use one or more:

- behavioral cloning from engine or human games
- prior-biased move sampling
- evaluator-based shaped reward
- curriculum from simple positions to full games

Goal: get to **legal, coherent, non-self-destructive play** quickly.

### Step 4: preserve controlled exploration

Good choices:

- softmax with temperature over move scores
- adding noise to policy probabilities
- opening-phase higher exploration, endgame lower exploration
- search-time exploration bonuses

Avoid relying on uniform random legal moves except as a fallback baseline.

### Step 5: anneal prior influence

Shift gradually from:

- **prior-led behavior** -> **learned-policy-led behavior**

Possible knobs to anneal:

- probability of following prior moves
- weight of shaped reward relative to terminal outcome
- share of demonstrations in replay
- search prior strength

The point is not “become random.”  
The point is “let the learned policy own more of the behavior.”

### Step 6: switch emphasis toward self-play refinement

Once the learned model is competent:

- generate more data from the model itself
- reduce dependence on demonstrations or fixed heuristics
- optimize against stronger versions of itself
- measure whether it now exceeds the prior

### Step 7: monitor failure modes

Watch for:

- **prior lock-in**: model never surpasses heuristic habits
- **reward hacking**: model optimizes evaluator score but not winning
- **policy collapse**: too little exploration, repetitive openings
- **search dependence**: model weak without expensive inference-time search
- **distribution brittleness**: strong on familiar lines, weak on offbeat play

---

## Minimal architecture choices

### Small-budget setup

- board encoder
- policy head over legal moves
- value head for win probability
- shaped reward from evaluator deltas
- replay buffer
- off-policy learner
- optional shallow search

### Stronger setup

- supervised warm start on expert or engine games
- self-play generation
- policy + value network
- MCTS or search-guided target improvement
- scheduled reduction of prior influence
- evaluation versus prior and held-out opponents

---

## Decision rules

- If the agent cannot play coherent legal chess, **increase prior guidance**.
- If the agent plays coherent but derivative chess, **decrease prior guidance**.
- If it optimizes centipawns but loses games, **reduce shaping dominance and restore terminal objective weight**.
- If it repeats narrow lines, **increase structured exploration**.
- If the same pathology persists across runs, **change the data-generation policy**, not just hyperparameters.

---

## One-line skill summary

**For chess-like systems, use heuristics to bootstrap, not to govern forever: start with prior-guided learning, keep exploration structured, and anneal control toward self-play-trained policy/value models.**

---

## Lesson-store candidates

Initial confidence for each would be **0.6** if you were storing them.

- **In sparse-reward strategy domains, a prior should reduce early nonsense but not cap final behavior.**
- **Annealing a prior means shifting toward the learned policy, not toward randomness.**
- **Early training with demonstrations or heuristic behavior is usually off-policy relative to the learner.**
- **Evaluator-based reward shaping improves sample efficiency but can trap the agent in heuristic local maxima.**
- **Structured exploration beats uniform random moves in large legal-action spaces like chess.**