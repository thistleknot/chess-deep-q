# RL Findings — Applying RL to Chess Efficiently

**Premise:** learn how to apply RL to chess *efficiently* (one GPU, not TPU-pod scale). Strength is
the yardstick; RL is the point. The classical alpha-beta engine (~1672 Elo, `engine.py`) is a
**non-RL baseline to surpass, never the deliverable.**

## Disposition (the answer)

Five RL configurations, measured on the ladder (vs random, vs the 1-ply hand heuristic):

| Config | Approach | vs heuristic-1ply |
|---|---|---|
| Distillation | supervised SF-eval regression | ~1040, **loses** |
| Self-play @d1 | tabula-rasa net-minimax | flat, **loses** |
| Off-policy bootstrap | 1672-engine trajectories | weak, **loses** |
| A2C + sampled beam | `mcts_chess.py`, 400 ep | 0/20, **loses** |
| **AlphaStar-hybrid PUCT** | **heuristic baseline + PUCT search + self-play** | **16/16 draws = PARITY** |

The first four are tabula-rasa / weak-operator — all **plateau** (lose to the heuristic). The fifth
breaks the plateau: measured with its real search, **net+PUCT draws the 1-ply heuristic 16/16
(parity)** — the first RL agent in the repo that does not lose to it. (Beats random ~0.72 — the net
is still weak; the search does the defending.)

## Why — the efficiency lesson

1. **Tabula-rasa (AlphaZero) doesn't work here.** It needs TPU-pod scale; on one GPU the net-alone
   plateaus. Four independent confirmations.
2. **The AlphaStar recipe is the efficient path.** Heuristic baseline (eps-blended search leaf) →
   PUCT reconstructs afterstate action-values → off-policy self-play refines → anneal the baseline
   out. Playable from iteration 0, reaches parity.
3. **Strength lives in the search, not the net.** The net approximates the **state-value V(s)**;
   action-values are **afterstates Q(s,a) = V(s·a)** reconstructed by search. Measuring the net at d1
   (no search) is BLIND — it reported 0/16 losses for an agent that draws 16/16. **Always measure the
   agent with its search.**
4. **The original repo already had the shape.** The v1.4 DQN + MCTS-over-hand-eval was playable in
   minutes for exactly this reason: search + a heuristic baseline carries early play, the net refines.
   That is afterstate Q-learning with a heuristic bootstrap — the AlphaStar shape (AlphaStar bootstraps
   from human replays; here, from the hand eval).

## Status

- **Learning goal: MET.** "How to apply RL to chess efficiently" = the AlphaStar recipe: bootstrap
  from a heuristic baseline, measure with search, let the net augment (not replace) search.
- **Strength: parity with the 1-ply heuristic**, not yet surpassing. Surpassing is **compute-bound**
  (more iterations / higher playouts) — an engineering scale-up, not a paradigm question, and out of
  scope for the efficiency lesson.
- **Deliverable shipped:** `play_puct.py` / menu option 20 — the net+PUCT RL agent, playable. The
  classical engine (option 18) is the labeled non-RL baseline.

## Coda — linear value RL (TD-Gammon-style, S&B ch9)

Sixth config, added to test the *efficient* path (µs linear eval, multiprocessing self-play, minutes
not hours): V(s)=wᵀφ(s) over hand features, RL-tuned by gradient-MC on self-play outcomes (S&B §9.5).
The value **fit converged cleanly** (VE mse 0.78→0.12 — the linear single-optimum guarantee, S&B §9.4)
— **but the eval plays 585 Elo WORSE**: linear-RL vs `pst` at equal depth = 0/30. This is **S&B §9.2
made concrete**: *"the best value function [for a better policy] is not necessarily the best for
minimizing VE."* Regressing weak self-play outcomes erodes the sound hand-tuned weights. **RL-tuning a
linear eval on self-play outcomes does not beat hand-tuning** — `pst` is near the linear-feature
ceiling. Sixth confirmation of the through-line: strength lives in search + sound weights; fitting
shallow values on self-play outcomes doesn't surpass hand-tuning on this hardware.

## Coda 2 — linear feature engineering (ch9/11): every knob lost to hand-tuned pst

Chasing the linear eval further (S&B ch9 §9.5 features, §11 Bellman/PBE targets):

| linear variant | vs pst @d2 | ~Elo |
|---|---|---|
| **material+PST (29 feat, MC)** | **−223** | **~1140 — best** |
| + rich positional (42 feat, MC) | 0/30 | ~worst (overfit) |
| + PCA-reduce (15) + TD(0)/PBE | −708 | ~659 |

Adding search-complementary positional features **overfit** (42 weights on ~160 games corrupt the
material block, 0/30). PCA-reduction + a TD(0)/Bellman target **rescued the overfit** (−1600→−708) but
lost pst precision (15-comp cap → warm-start corr 0.89) and gave a great fit with worse play (mse ~0.1,
§9.2/§11: min-PBE ≠ strong policy). **Hand-tuned `pst` (~1140) beat every linear-RL variant.** The
lesson: decades of hand-tuning already sit near the **linear ceiling**; ~160 games of self-play RL
can't improve on it within the linear class. The only real levers left are **more data** or **leaving
the linear class** (deeper search / a bigger net at scale) — not more linear feature engineering.

## Coda 3 — the (τ, λ) hybrid: the interior λ is the sweet spot (first improvement on MC)

Reframed the search/target choices as two continuous dials (τ = tempered top-K sampling over the
sound-ordered moves; λ = `lambda_return` bootstrapped from the search value), tested only at extremes
before. λ sweep at τ=0.5 (material+PST) vs pst @d2:

| λ | vs pst @d2 | ~Elo |
|---|---|---|
| 0.0 (TD0) | −458 | ~909 |
| 0.5 | −241 | ~1126 |
| **0.9** | **−176** | **~1191 — best** |
| 1.0 (MC) | −241 | ~1126 |

A clean interior peak (‸): λ≈0.9 beats **both** endpoints and the prior material+PST-MC best (~1144) —
the bias-variance middle was real. **First linear config to improve on plain MC.** Modest (~+50 Elo,
small n, still 0/30 wins vs pst) — the λ tuning squeezes the interior; it doesn't break the linear/data
wall. Best linear agent to date: the two-dial hybrid (τ≈0.5, λ≈0.9).

## Coda 4 — Stage 2: the enriched-data NNUE (the distillation bet), measured

Rebuilt the label pipeline per fresh literature review (in-distribution teacher-trajectory positions,
quiet-filter, soft-MultiPV + WDL, ~3× augmentation, multiprocessing labeling 58→~120/s) and trained the
v1 NNUE arch on **156k enriched positions** (~450k augmented). Result vs pst @d2:

| eval | vs pst @d2 | ~Elo |
|---|---|---|
| v1 NNUE (66k random/one-hot) | −559 | ~808 |
| **Stage-2 NNUE (156k enriched)** | **−585** | **~782 — no improvement** |

Better data at this volume did **NOT** move the wall. Two root causes, the second more important:
1. **Volume** — 156k ≪ the millions the literature needs (arXiv:2412.17948 ≈ 44k games ≈ millions of positions).
2. **Coverage** — quiet + strong-SF-play trajectories are material-**BALANCED** (mean |cp|~170; only 8%
   have |cp|>750), so the eval never learns to value decisive material (queen-up read **+380**, not +900),
   and alpha-beta exploits that hole at depth. The `:Quiet-filter:` that correctly removes tactical noise
   ALSO strips the material-imbalanced-but-quiet positions (won endgames, pawn-up middlegames) the eval
   needs. Stockfish's own NNUE avoids this only via a HUGE corpus that contains them.

**COVERAGE FIX (`:Material-coverage:`, the unlock):** seeded 35% of trajectories from a material
imbalance (`perturb_material`: remove 1–3 pieces), regenerated to **469k** positions (decisive |cp|>750:
8%→13%). The eval learned to value material — queen-up **+380→+671**, start +160→**+41**, sign_acc
**0.746→0.815**, val_rmse 174→146cp — and PLAY jumped:

| eval | vs pst @d2 | ~Elo |
|---|---|---|
| v1 (66k random/one-hot) | −559 | ~808 |
| enriched, balanced (156k) | −585 | ~782 |
| **enriched + coverage (469k)** | **−223** | **~1144** |

**+362 Elo from the coverage fix** (shutout → 13/30 draws) — the first LEARNED eval to reach the linear
ceiling (~1140). Disposition FLIPPED from coda 2/3: the data lever WORKS; the earlier "learned evals
can't beat pst" was a DATA-coverage artifact, not a ceiling. Crucially, unlike the linear eval (feature-
ceiling'd), the NNUE is nonlinear and was still improving (sign_acc climbing, val_rmse dropping) — it has
HEADROOM past pst with more coverage+volume. Gap to pst is now −223, not −585. Next levers: more
coverage/volume (the metrics hadn't plateaued), then the depth advantage (NNUE at d3+ once fast, Stage 3).

## Coda 5 — the deployed agent (NNUE critic + phi-widening + tree reuse), measured

Built the integrated play agent (spec `:Deployed-agent:`): sound alpha-beta + NNUE `:Critic-leaf:` +
`:Full-width-floor:` (d3) + `:Phi-widening:` (forward-prune to a Fibonacci budget, captures/checks exempt
= refutation-preserving) + `:Tree-reuse:` (persistent TT). Verified: tactically SOUND (same move as
full-width on free-queen/trap positions), **93% node reduction at d5** (9,556 vs 137,565), tree reuse
works. But the EQUAL-TIME kill-check (`measure_phi.py`, 0.3s/move):

| match | score | ~Elo |
|---|---|---|
| NNUE+phi vs pst @0.3s | 0.15 (−301) | ~1127 |
| NNUE+phi vs NNUE-no-phi @0.3s | 0.45 (−35) | wash |

**The throughput wall dominates, cleanly isolated.** The search structure is correct, but at a TIME
budget the 281µs recompute-per-leaf NNUE caps depth, and phi's 93% node cut does NOT overcome the
per-eval cost — pst's µs eval simply searches deeper, so phi-widening is a wash vs plain alpha-beta and
both lose to pst. The eval-quality edge (equal-DEPTH: d3=1481) is ERASED at equal TIME. Disposition: the
architecture is validated and reusable; equal-time strength is gated on eval SPEED, not search structure.
**Stage 3 (the incremental `:Accumulator:` + int8, µs eval) is the binding unlock** — make the eval fast
and phi's node cut converts to depth, flipping the equal-time comparison. The eval-quality lever (the
distillation climb) keeps rising in parallel (d2 1144→1284→…).

## Coda 6 — the accumulator (Stage 3) shipped, but did NOT flip the equal-time wall (Phase C corrected)

Merged the incremental accumulator: bit-exact (3.2e-7cp), **7.8× faster** (407µs → 52µs eval), engine-parity
verified, deployed agent + measure both on it. But the equal-TIME retest (NNUE+phi FAST eval vs pst @0.3s):
**−338 (2W-1D-17L)** — statistically the SAME as the slow eval's −301, and phi-vs-no-phi is a dead wash (0.50).

**Phase C's disposition was incomplete and is corrected here.** Phase C attributed the equal-time loss to the
throughput WALL (slow eval caps depth). The fast eval falsifies that as the *sole* cause: matched on
speed, NNUE STILL loses, because it is weaker than pst **at every depth** (d2 −83, d3 −191). The binding
constraint at equal time is **eval QUALITY, not speed** — deeper search with a weaker eval does not beat
deeper search with pst's better (and also µs-fast) eval. The accumulator was NECESSARY infrastructure (a real
7.8×; it makes DA2C self-play cheap and lets a *good* eval play deep) but NOT sufficient. Caveat: measured
under heavy load (the climb), so both engines searched shallow — favoring pst's shallow-material strength;
an unloaded re-measure is fairer. DISPOSITION: the path to >1600-at-time requires the eval to **SURPASS pst**
(the persistent eval-quality problem) — the climb (distillation) + DA2C (self-improve) are the levers;
accumulator + phi + tree-reuse are the enabling infrastructure, not the strength.

## Reproduce

```
python puct_selfplay.py 12 96 64     # train  -> models/tower_puct.pt
python puct_eval.py 80 16            # measure net+PUCT vs the ladder
python play_puct.py                 # play the RL agent (or menu option 20)
```

**Specs:** `self-play-leela.spec.md` (`:Amplification-experiment:`, `:Search-measured-gate:`,
`:Search-profile:`), `rl-categorization.spec.md` (`:Afterstate-action-value:`, `:Heuristic-baseline:`),
`chess-rl.spec.md` (`:Measured-disposition:`).
