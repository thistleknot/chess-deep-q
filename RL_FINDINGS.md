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

## Reproduce

```
python puct_selfplay.py 12 96 64     # train  -> models/tower_puct.pt
python puct_eval.py 80 16            # measure net+PUCT vs the ladder
python play_puct.py                 # play the RL agent (or menu option 20)
```

**Specs:** `self-play-leela.spec.md` (`:Amplification-experiment:`, `:Search-measured-gate:`,
`:Search-profile:`), `rl-categorization.spec.md` (`:Afterstate-action-value:`, `:Heuristic-baseline:`),
`chess-rl.spec.md` (`:Measured-disposition:`).
