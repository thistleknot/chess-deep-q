# NNUE + Uncertainty-Directed Training

## CURRENT DISPOSITION: SPEC — awaiting implementation

## Bitter-Lesson Audit (the why)

The champion pipeline is capped by **Method 5** (linear over fixed features) and
**Method 2** (engineered signal on the evaluation/output side):

| Principle | Status | Detail |
|---|---|---|
| 5. Learn the cumulants | **NO — THE CAP** | `V(s) = tanh(w·φ + b)`, 897 weights, linear. Cannot learn conjunctions. |
| 2. Architecture as representation prior | **NO** | amap-897 features sit ON the eval output, not under a learned encoder. |

More data does not fix a NO on 5 or 2. Confirmed empirically: 4 independent
data-augmentation approaches (MC labels, d5 novel, uncertainty-steered, random-eps)
all landed at ~1800 ± 50. The linear basis has converged.

## The Fix

Move the engineered features **under a nonlinear substrate** that can learn
conjunctions and outgrow the hand-designed signal:

```
amap-897 sparse features
    ↓
EmbeddingBag accumulator (learned, 128-dim)  ← learns WHAT combinations matter
    ↓
ClippedReLU
    ↓
Hidden (32) → ReLU → scalar (centipawns)
```

This is the existing `NNUENet` in `chessdq/nnue_model.py`. It satisfies:
- Method 5: nonlinear combiner over sparse features (can learn cumulants)
- Method 2: engineered features become the INPUT to a learned encoder, not the eval itself

## What Failed Before (P2) and Why This Differs

P2 tested halfKP NNUE vs linear champion at matched d4 search: **lost −512 Elo**.
Diagnosis: trained on the SAME 65k corpus that the linear model already fit perfectly.
The NNUE overfitted a small dataset where the linear was already at ceiling —
it couldn't learn what wasn't in the data.

**This time:** train the NNUE on an **uncertainty-enriched** corpus:
1. Standard 65k d5-labeled positions (the existing cache)
2. PLUS 10k+ novel positions from uncertainty-steered generation (confirmed:
   `ens_weights` in rsearch4 produces 10,191 novel positions the linear never saw)
3. PLUS augmentation (mirror + color-flip, ~3x effective data)

The NNUE has capacity to learn from the novel positions; the linear cannot.
The uncertainty-directed data covers the regions where conjunctive features
(piece interactions the linear can't represent) actually matter.

## Pipeline

```
1. GENERATE (existing, confirmed)
   rsearch4.play_games(w, b, ..., eps=0.25, ens_weights=ensemble)
   → 10k+ novel positions from uncertain territory

2. LABEL (existing, confirmed)
   rsearch4.Searcher.search(fen, 5) at d5 native
   → White-tanh values, same quality as existing cache

3. MERGE
   65k existing cache + 10k novel = 75k+ labeled positions

4. CONVERT to NNUE training format
   Each position → sparse feature indices (king-bucketed HalfKP or amap-sparse)
   + centipawn target (from tanh via cp_from_tanh)

5. TRAIN NNUE
   Supervised MSE on the merged corpus
   + augmentation (mirror/color-flip, ~3x)
   Architecture: NNUENet(acc_dim=128, hidden=32) — same as existing

6. DEPLOY
   AlphaBetaEngine(eval_fn=make_incremental_nnue_eval(net), phi_widen=True)
   Same native alpha-beta search, different (nonlinear) leaf eval

7. MEASURE
   Standard anchor ladder: 1320/1500/1700/1900/2100, bell-curve 100g
   Compare vs champion.pt (linear, 1816) on the SAME instrument
```

## Kill Gates

1. **Training convergence:** val loss must decrease for ≥3 epochs (not diverge/overfit)
2. **d4 matched-search duel vs linear champion:** NNUE must score ≥ 0.500
   (P2 scored 0.000 — any improvement is progress)
3. **Standard anchor ladder:** MLE rating must exceed 1816 (champion's measured floor)
   with CI excluding 1816, or at minimum the point estimate must be higher

## What Stays the Same

- Native alpha-beta search (rsearch4) — unchanged
- Deployment depth (d9) — unchanged  
- Measurement instrument (anchor_ladder, standard protocol) — unchanged
- The features themselves (sparse piece-square indices) — unchanged
- The search-distillation labeling (d5 native minimax) — unchanged

## What Changes

- Leaf evaluator: `linear(w·φ+b)` → `NNUENet(EmbeddingBag → ReLU → head)`
- Training data: 65k → 75k+ (uncertainty-enriched)
- Training method: closed-form ridge → SGD on MSE (existing `train_nnue.py`)

## Existing Code (all built, no new modules needed)

- `chessdq/nnue_model.py` — NNUENet, features(), make_nnue_eval, make_incremental_nnue_eval
- `experiments/train_nnue.py` — training loop (MSE, augmentation, checkpointing)
- `experiments/anchor_ladder.py` — standard measurement protocol
- `chessdq/engine.py` — AlphaBetaEngine with pluggable eval_fn
- `rsearch4.play_games` with `ens_weights` — uncertainty-steered generation

## Risk

The P2 failure was on 65k positions without augmentation or enrichment. If the
enriched corpus still isn't enough to prevent NNUE overfitting (the 2560-dim
embedding has ~330k parameters vs ~75k training examples), the model will memorize
rather than generalize. Mitigation: augmentation (3x data) + early stopping on
held-out val loss + weight decay.

## Relationship to Operator's Idea

The operator's original concept (S&B Ch.9 margin notes): "value function model
to explore uncertain areas, followed by heuristic search from those points."
This spec realizes it as:
- **Value function model** = NNUE (nonlinear, capacity to grow)
- **Explore uncertain areas** = uncertainty-steered generation (ens_weights)
- **Heuristic search from those points** = native d5 labeling at the novel positions
- **Everything else stays minimax** = same alpha-beta d9 deployment

The bitter-lesson skill identifies WHY the linear version couldn't absorb the
exploration signal: Method 5 cap. The NNUE removes the cap. The uncertainty
exploration provides the data the NNUE needs but the linear didn't.
