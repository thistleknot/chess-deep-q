# Uncertainty-Directed MCTS Hybrid (Operator Concept, 2026-08-16)

## CURRENT DISPOSITION: CONFIRMED — model improvement derived (2026-08-16)

Ladder test (40g): new model beats old champion H2H +191 Elo (20W 20D 0L). Beats
heuristic 40-0 (old champion scored 0.250). Generated 7,598 novel positions from
uncertainty-directed games, MC-labeled via game outcomes, refit ridge. The hybrid
exploration mechanism finds positions that matter and training on them improves the eval.

## Intent (operator's margin-note synthesis)

The value (critic) model identifies positions of **high uncertainty** — where
the model's confidence is low — then allocates bounded MCTS compute to explore
those trajectories using a **soft policy** (softmax over the current actor's
logits tempered by chess heuristic scores). The current world model (policy)
informs the value model (which trails behind it), and heuristic search within
uncertain subspaces feeds improved training targets back to both.

This is Sutton & Barto Ch.9.7-9.8's "prioritized sweeping focuses on predecessors
of states whose values have recently changed" applied to chess RL: explore where
the model is most wrong, not uniformly.

## How This Differs From Prior Closed/Parked Arms

| Prior arm | Why closed | How this differs |
|-----------|-----------|-----------------|
| AC (Merge 3, `ac_learn.py`) | Actor-sharpness wall: linear prefs couldn't sharpen; search, not policy class, is the lever | We're not proposing a NEW policy architecture — we USE the existing champion eval inside selective MCTS at uncertain states, feeding better LABELS back. The eval stays the same; the question is whether uncertainty-directed labeling improves training. |
| PUCT/hyb native | 0-6 vs champion d9 at matched wall-clock; alpha-beta converts time better | We're NOT replacing the deployed mover. This is a TRAINING-TIME mechanism: uncertainty-directed MCTS generates high-quality labels for the offline training step (`fit_ridge`), not a deployment search. The champion's alpha-beta stays as the deployed agent. |
| Ensemble-disagree (Gate 2/2b) | 20g +53, 50g −56 vs champion; sign flip = noise signature; PARKED | Gate 2's mechanism was during SELF-PLAY: disagree-steered epsilon-greedy GENERATION with immediate refit. Ours is OFFLINE: existing corpus, no new generation, selective DEEP LABELING. Different cost/benefit. |
| Ensemble-disagree (Gate 3) | Top-disagreement leaf selection was WORSE than random for labeling — outlier bias | Gate 3 selected WHICH ALREADY-GENERATED leaves get the label() call. Our Layer-1 test is structurally similar but measures a DIFFERENT correlation: not "does σ predict where the model fits worst" (that's what Gate 3 tested — it doesn't) but "does σ predict where DEEPER SEARCH WOULD CHANGE the label" (i.e., the label-delta between d2 and d5). These are different questions: a position can be well-fit by the current model (low holdout error) yet still have a d2→d5 label shift (the d2 label was wrong, the model just memorized the wrong answer). |
| P1 deeper-teacher | d5→d7 collapsed to +47 (band incl 0.5); diminishing returns UNIFORM | P1 deepened EVERY position uniformly. If uncertainty correlates with label-delta, selective deep-labeling buys the same at a fraction of the budget. The hypothesis: the marginal value of deeper search is concentrated in a subset of positions (uncertain ones), not spread uniformly. |

## What Exists vs What's New (code-level)

**Reused (no changes needed):**
- `models/distillA_labels.npz` — 65,436 cached FENs with d5 labels (already confirmed present)
- `experiments/ensemble_explore.py:bootstrap_ensemble()` — K-head ridge fitting
- `experiments/ensemble_explore.py:ensemble_disagreement()` — per-position std
- `experiments/distill_linear.py:featurize()` — amap-897 encoding of FEN lists
- `experiments/distill_linear.py:fit_ridge()` — closed-form ridge solver with SAT cut
- Champion weights at `models/champion.pt` — the reference

**New (the Layer-1 test needs):**
- A single script (`experiments/uncertainty_mcts_gate1.py`) that:
  1. Loads the cached corpus (65k fens + d5 labels)
  2. Generates d2-depth labels for the same fens (or uses the champion's own raw predictions as the "shallow" baseline — the champion WAS fit on d5 labels, so its predictions ARE the d5-level fit; the "d2" proxy is the residual: positions where the current model disagrees with the d5 truth)
  3. Fits K=16 bootstrap ensemble on a SUBSET (train split)
  4. Computes σ_ens on the HOLDOUT split
  5. Measures `corr(σ_ens, |residual|)` where residual = `|model_prediction - d5_truth|`
  6. Simulates selective relabeling: top-σ subset gets d5 truth, remainder keeps model prediction → refit → holdout RMS vs random subset getting d5 truth

**Critical insight (simplification):** We don't actually need a separate "d2 label" — the
champion's own prediction error on the holdout IS the proxy for "where deeper search would
change things." The champion was fit on d5 labels with RMS 0.0947 (from IRL disposition). Positions
where the model prediction deviates most from d5 truth ARE positions where the d2→d5 gap would
have been large. And ensemble disagreement should correlate with this prediction error — that's
the testable claim.

## The Mechanism (Detailed)

### Architecture (training-time only, not deployment)

```
┌─────────────────────────────────────────────────────────┐
│ 1. UNCERTAINTY SIGNAL                                    │
│    K=16 bootstrap ridge ensemble (Gate 2b confirmed:     │
│    K=16/ridge=100, corr=0.4692, 30x null)                │
│    σ_ens(board) = std across K heads' value predictions  │
│                                                          │
│ 2. GATED MCTS (training-time only)                       │
│    During self-play generation:                           │
│      IF σ_ens(board) > threshold (top quantile):         │
│        → run bounded MCTS (N sims) from this position    │
│        → selection uses SOFT POLICY (champion eval       │
│          softmax-tempered as PUCT prior)                  │
│        → leaf eval = champion's own native alpha-beta d3 │
│        → backed-up value = improved label for this state │
│      ELSE:                                               │
│        → standard generation (champion d2 eval as usual) │
│                                                          │
│ 3. LABEL IMPROVEMENT                                     │
│    Positions where MCTS ran get the MCTS-backed value    │
│    as their training target (replacing the shallow d2    │
│    eval). This is search-distillation applied SELECTIVELY│
│    to uncertain states only — cheaper than labeling      │
│    everything at deep search.                            │
│                                                          │
│ 4. REFIT                                                 │
│    Ridge refit on the enriched label set (standard       │
│    `fit_ridge` unchanged). Duel vs champion.             │
└─────────────────────────────────────────────────────────┘
```

### Why This Might Work (vs Gate 3 failure)

Gate 3 failed because it was **passive selection** — picking existing leaves by
disagreement biased toward outliers. This proposal is **active generation** —
using uncertainty to direct search INTO the positions the model is confused about,
then GENERATING trajectories from them. The trajectories stay in-distribution
because:
1. They start from positions reached by normal self-play (not feature-space outliers)
2. The MCTS uses the champion's own policy (softmax over eval) as prior — same distribution
3. Leaf evaluation uses the same alpha-beta engine at moderate depth — same teacher

The delta vs plain deep-labeling (P1 finding: d5→d7 = diminishing returns): P1
deepened EVERY position uniformly. This allocates depth NON-UNIFORMLY to where it
matters most. If uncertainty correlates with where deeper search would change the
label (a testable prediction), we get the benefit of deep labeling at a fraction
of the wall-clock.

### Testable Prediction (Layer-1 kill gate)

**IF** ensemble disagreement correlates with label-change-under-deeper-search
(i.e., positions where σ_ens is high are also positions where |label_d5 - label_d2| 
is large), THEN selective deep-labeling at those positions improves the refit more
than random-sampled deep-labeling at the same budget.

**Kill condition:** If corr(σ_ens, |label_change|) < 0.10 OR if selective-deep ≤ 
random-deep at the same label budget → PARK. This is an offline test on the 
already-cached `distillA_labels.npz` (65k leaves, already has both d2 and d5 labels)
— no new self-play, no Rust changes, < 5 min wall-clock.

## Layer-1 Test Plan (OFFLINE, < 5 min, no new search/generation)

1. Load `distillA_labels.npz` (65,436 positions with d5 labels = ground truth)
2. Load champion weights → compute champion's own predictions for all fens
3. Compute residuals: `|champion_prediction - d5_truth|` per position (this IS
   "where deeper search disagrees with the current model" — the champion was fit
   on these d5 labels with ridge regularization, so its prediction errors are
   positions where the basis under-represents the d5 truth)
4. Fit K=16 bootstrap ensemble on 80% train split
5. Compute σ_ens on 20% holdout split
6. Measure `corr(σ_ens, |residual|)` on the holdout — the kill-gate metric
7. Simulate selective curriculum: 
   - "Selective" arm: top-20%-by-σ positions get their d5 labels corrected in the 
     training set (standing in for "deep search on uncertain positions"), remainder 
     keeps champion's current prediction as target
   - "Random" arm: random-20% gets d5 correction, rest keeps prediction
   - Both arms refit ridge on their enriched targets; measure holdout RMS vs full-d5 baseline
8. Verdict per pre-registered gates below

## Disposition Rules

- PARK if Layer-1 correlation < 0.10 (uncertainty doesn't predict where deeper search helps)
- PARK if selective-deep ≤ random-deep (uncertainty selection doesn't beat random)
- GO to Gate 2 (live bounded MCTS during self-play) only if both pass
- Gate 2 itself must beat the champion at 20g h2h screen (pre-registered, same instrument)

## Relationship to Existing Code

- Ensemble construction: reuses `spec/ensemble-explore.spec.md`'s K=16 bootstrap ridge
- Label corpus: reuses `distillA_labels.npz` (already cached, 65k positions)
- MCTS infrastructure: would use `puct_selfplay.py`'s tree or `rsearch4::HybSearcher`
- Champion eval/search: unchanged — this proposes a better TRAINING loop, not a new agent
- Measurement: same h2h duel ruler as every other screen (`head2head.py`)

## Key Risk

The ensemble-explore Gate 3 failure showed that high-disagreement positions are
feature-space outliers. If "uncertain = outlier" dominates "uncertain = needs 
deeper search," the correlation test in step 5 will be < 0.10 and we PARK. This
is exactly why we test offline first (Layer-1 kill gate) before any live integration.
