# RL theory ⇄ this repo — cliff notes

Squares the n-step / actor-critic / bootstrap thread against what we actually built (chess TD-Leaf).

## The mechanism we're running
- **n-step actor-critic (TD-Leaf), NOT MC/GRPO.** `da2c.py`: target = `lambda_return(V_search)`, actor `−logπ·(V_search−V_net)`. One trajectory, γ-bootstrap, zero cross-rollout averaging.
- **"n" has two axes here** (kept separate): **search-depth n** = the spearhead lookahead baked into `V_search`; **temporal n** = game moves (`DA2C_NSTEP=10`), blended by **λ_game≈0.85** (the eligibility trace / GAE-analog).

## The n<T rule (the transcript's trap) — we pass it
- n-step bootstrap **only fires when n < T**; n≥T collapses to MC / REINFORCE-with-baseline. Chess T≈40–80 moves, `DA2C_NSTEP=10` → **bootstrap engages**, not collapsed. (Guard: never set NSTEP ≥ game length.)

## Sparse reward (chess = terminal z only)
- **γ is the backcast operator** — `G_t` collapses to `γ^(T−t−1)·z`, nonzero, distance-scaled, one trajectory, **no averaging**. We run γ=1 (undiscounted, `DA2C_GAMMA=1.0`), correct for a win/loss game.
- Mid-game the n-step window rarely reaches terminal → target is **bootstrap-dominated**. Normally that's "only as good as V(s_{t+n})" (noise early). **TD-Leaf sidesteps it**: the bootstrap is `V_search` (deep sound search), not the raw net — a strong signal even with no reward in the window. This is *why* the search operator matters.

## "does bootstrap imply residual?" — yes
- Bootstrap = use your own prediction in the target ⇒ loss **is** the TD/Bellman residual: `(r + γV(s′) − V(s))²`. Our critic loss `(V_net − target)²` with target bootstrapped = the residual. The **advantage** `V_search − V_net` bootstraps by the same token. (Ties to S&B ch.11: TD minimizes the PBE, not the raw BE.)

## "average" — literal elsewhere, loose here
- Tabular MC: V(s)=sample-mean of realized returns. GRPO: `A_i=(R_i−mean)/std` over a group (flat per token). Both literal averages. **n-step actor-critic (us) has no average inside a step** — discounted sum + bootstrap; the only mean is over the minibatch.
- GRPO is the user's *other* (LLM/token-MDP) tool — different family, not this repo.

## Rollout vs replay
- **Rollout** (on-policy, fill→few updates→clear) = our DA2C. **Replay** (persistent, random reuse) = the **Q-learning lane** (deferred, off-policy, the sample-efficient teacher-reuse path). ≠ each other.

## Empirical anchor (this session)
- **DA2C self-improve DEGRADED the eval at BOTH depths**: distilled `nnue.pt` (d2 −58 / d3 −191) → self-improved `nnue_da2c.pt` (**d2 −301 / d3 −301**). Even with the sound spearhead + TD-Leaf + annealed demo-share, self-play **eroded** the distilled weights (S&B §9.2: regressing self-play outcomes drags weights off the sound point). **Distillation > self-play here** — the fifth+ confirmation of the through-line: on this hardware/data, learning the value from self-play OUTCOMES does not beat learning it from a strong teacher's LABELS.
- **Disposition (honest):** the distillation climb reached the milestone — **parity with pst at d2 (1367, first learned eval to match the hand eval)** — but the eval never SURPASSED pst, self-play degraded it, and the equal-time wall stands at **−255** (learned eval weaker than pst at depth, holes exploited by search). We did not break 1600. Ceiling: a learned eval that *matches* pst at shallow depth, not one that beats it.
