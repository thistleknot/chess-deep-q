import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Uncertainty-directed MCTS — Layer-1 offline kill gate (spec/uncertainty-mcts.spec.md).

Tests whether ensemble disagreement (σ_ens) correlates with model prediction error,
and whether SELECTIVELY correcting high-σ positions improves the fit more than
randomly correcting the same number of positions.

All offline, no new search/generation. Reuses:
  - models/distillA_labels.npz (65k FENs + d5 labels = ground truth)
  - experiments/ensemble_explore.py:bootstrap_ensemble, ensemble_disagreement
  - experiments/distill_linear.py:featurize, fit_ridge
  - models/champion.pt (raw weights = current model's predictions)

Kill gates (pre-registered, spec/uncertainty-mcts.spec.md):
  1. corr(σ_ens, |residual|) >= 0.10  (uncertainty predicts where model is wrong)
  2. selective_rms < random_rms        (σ-directed correction beats random correction)
  Both must pass → GO to Gate 2. Either fails → PARK.

Usage: python -m experiments.uncertainty_mcts_gate1
"""
import os
import time

import numpy as np

from chessdq.corpus_gen import raw_weights
from experiments.distill_linear import CHAMPION, CACHE, featurize, fit_ridge
from experiments.ensemble_explore import bootstrap_ensemble, ensemble_disagreement

# --- config ---
K = 16
RIDGE = 100.0
SEED = 977
HOLDOUT_FRAC = 0.20
CORRECTION_FRAC = 0.20  # fraction of TRAIN set selectively relabeled
SAT = 0.90               # match distill_linear's saturation cut


def load_corpus():
    """Load cached FENs and d5 labels (the ground truth)."""
    if not os.path.exists(CACHE):
        raise SystemExit(f"{CACHE} missing — run experiments/distill_linear.py run first")
    z = np.load(CACHE, allow_pickle=True)
    return list(z["fens"]), z["y"]


def champion_predictions(X, w, b):
    """Champion's own raw predictions in tanh space (what the deployed model believes)."""
    raw = X @ w + b  # atanh-space linear score
    return np.tanh(raw)


def main():
    t0 = time.time()
    print("=" * 72)
    print("UNCERTAINTY-MCTS LAYER-1 KILL GATE")
    print("=" * 72)

    # 1. Load corpus and champion weights
    print("\n[1/7] Loading corpus and champion...", flush=True)
    fens, y_d5 = load_corpus()
    w, b = raw_weights(CHAMPION)
    w = np.asarray(w, dtype=np.float64)
    b = float(b)
    n = len(fens)
    print(f"  corpus: {n} positions, d5 labels range [{y_d5.min():.3f}, {y_d5.max():.3f}]")

    # 2. Featurize
    print("\n[2/7] Featurizing (amap-897)...", flush=True)
    X = featurize(fens)
    print(f"  X: {X.shape}")

    # 3. Champion predictions + residuals
    print("\n[3/7] Computing champion predictions and residuals...", flush=True)
    y_pred = champion_predictions(X, w, b)
    residuals = np.abs(y_pred - y_d5)
    print(f"  mean |residual| = {residuals.mean():.4f}")
    print(f"  median |residual| = {np.median(residuals):.4f}")
    print(f"  max |residual| = {residuals.max():.4f}")
    # Filter to non-saturated (same as fit_ridge does)
    keep = np.abs(y_d5) <= SAT
    print(f"  non-saturated: {keep.sum()}/{n} ({keep.mean():.1%})")

    # 4. Train/holdout split
    print("\n[4/7] Train/holdout split...", flush=True)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_hold = int(n * HOLDOUT_FRAC)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    X_hold, y_hold = X[hold_idx], y_d5[hold_idx]
    X_train, y_train = X[train_idx], y_d5[train_idx]
    residuals_hold = residuals[hold_idx]
    print(f"  train: {len(train_idx)}, holdout: {len(hold_idx)}")

    # 5. Bootstrap ensemble on TRAIN split
    print("\n[5/7] Fitting K=16 bootstrap ensemble on train split...", flush=True)
    W_ens, B_ens = bootstrap_ensemble(X_train, y_train, K, RIDGE, SEED)
    # Compute disagreement on HOLDOUT
    sigma_hold = ensemble_disagreement(W_ens, B_ens, X_hold)
    print(f"  σ_ens holdout: mean={sigma_hold.mean():.4f}, "
          f"std={sigma_hold.std():.4f}, max={sigma_hold.max():.4f}")

    # 6. KILL GATE 1: corr(σ_ens, |residual|)
    print("\n[6/7] KILL GATE 1: corr(σ_ens, |residual|)...", flush=True)
    corr_sigma_resid = float(np.corrcoef(sigma_hold, residuals_hold)[0, 1])
    print(f"  corr(σ_ens, |residual|) = {corr_sigma_resid:.4f}")
    gate1_pass = corr_sigma_resid >= 0.10
    print(f"  threshold: >= 0.10")
    print(f"  GATE 1: {'PASS' if gate1_pass else 'FAIL'}")

    # Also report rank correlation (Spearman) as a robustness check
    from scipy.stats import spearmanr
    rho, p_val = spearmanr(sigma_hold, residuals_hold)
    print(f"  Spearman ρ = {rho:.4f} (p = {p_val:.2e})")

    # Breakdown by quantile
    q_edges = np.quantile(sigma_hold, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"  |residual| by σ_ens quartile:")
    for qi in range(4):
        mask = (sigma_hold >= q_edges[qi]) & (sigma_hold < q_edges[qi + 1] + 1e-9)
        if mask.sum() > 0:
            print(f"    Q{qi+1} (σ in [{q_edges[qi]:.4f}, {q_edges[qi+1]:.4f}]): "
                  f"mean|resid|={residuals_hold[mask].mean():.4f}, n={mask.sum()}")

    # 7. KILL GATE 2: selective correction vs random correction
    print("\n[7/7] KILL GATE 2: selective vs random correction simulation...", flush=True)
    # On the TRAIN set: simulate "correcting" a fraction of labels from the model's
    # prediction to the true d5 value. The rest keeps the model's prediction as target.
    # This simulates "selectively labeling uncertain positions at d5" vs baseline.

    y_pred_train = champion_predictions(X_train, w, b)
    sigma_train = ensemble_disagreement(W_ens, B_ens, X_train)

    n_correct = int(len(train_idx) * CORRECTION_FRAC)
    print(f"  correcting {n_correct}/{len(train_idx)} positions ({CORRECTION_FRAC:.0%})")

    # Selective arm: top-σ positions get d5 truth
    top_sigma_idx = np.argsort(-sigma_train)[:n_correct]
    y_selective = y_pred_train.copy()
    y_selective[top_sigma_idx] = y_train[top_sigma_idx]

    # Random arm: random positions get d5 truth
    rnd_correct_idx = rng.choice(len(train_idx), size=n_correct, replace=False)
    y_random = y_pred_train.copy()
    y_random[rnd_correct_idx] = y_train[rnd_correct_idx]

    # Full-d5 baseline (all positions get d5 truth — the best possible)
    y_full = y_train.copy()

    # Fit ridge on each target set, measure holdout RMS against d5 truth
    # (holdout labels are always the d5 truth — the unbiased ground truth)
    def fit_and_measure(X_tr, y_tr, label):
        w_fit, b_fit, rms_fit = fit_ridge(X_tr, y_tr, RIDGE)
        # Predict on holdout in atanh space, then tanh back
        pred_hold = np.tanh(X_hold.astype(np.float64) @ w_fit + b_fit)
        # RMS against d5 truth (tanh space, the deployment scale)
        hold_rms = float(np.sqrt(np.mean((pred_hold - y_hold) ** 2)))
        # Also correlation
        hold_corr = float(np.corrcoef(pred_hold, y_hold)[0, 1])
        print(f"    {label:12s}: fit_rms(atanh)={rms_fit:.4f}, "
              f"hold_rms(tanh)={hold_rms:.4f}, hold_corr={hold_corr:.4f}")
        return hold_rms, hold_corr

    print(f"  fitting ridge for each arm...")
    rms_sel, corr_sel = fit_and_measure(X_train, y_selective, "selective")
    rms_rnd, corr_rnd = fit_and_measure(X_train, y_random, "random")
    rms_full, corr_full = fit_and_measure(X_train, y_full, "full-d5")
    # Also measure: what if we just use model predictions as targets (no correction)?
    rms_none, corr_none = fit_and_measure(X_train, y_pred_train, "no-correction")

    gate2_pass = rms_sel < rms_rnd
    delta_rms = rms_rnd - rms_sel
    rel_gain = delta_rms / rms_rnd if rms_rnd > 0 else 0
    print(f"\n  RESULTS:")
    print(f"    no-correction hold RMS: {rms_none:.4f}")
    print(f"    random-20%    hold RMS: {rms_rnd:.4f}")
    print(f"    selective-20%  hold RMS: {rms_sel:.4f}")
    print(f"    full-d5       hold RMS: {rms_full:.4f}")
    print(f"    delta (random - selective): {delta_rms:+.5f} ({rel_gain:+.2%} relative)")
    print(f"  GATE 2: {'PASS' if gate2_pass else 'FAIL'} (selective < random: {rms_sel:.5f} vs {rms_rnd:.5f})")

    # --- FINAL VERDICT ---
    print("\n" + "=" * 72)
    elapsed = time.time() - t0
    if gate1_pass and gate2_pass:
        print(f"VERDICT: GO — both gates pass. corr={corr_sigma_resid:.4f} >= 0.10, "
              f"selective RMS {rms_sel:.4f} < random RMS {rms_rnd:.4f} ({rel_gain:+.2%}).")
        print(f"Proceed to Gate 2 (live bounded MCTS during self-play, spec/uncertainty-mcts.spec.md).")
    elif not gate1_pass:
        print(f"VERDICT: PARK — Gate 1 FAILED. corr(σ_ens, |residual|) = {corr_sigma_resid:.4f} < 0.10.")
        print(f"Ensemble disagreement does NOT predict where the model is wrong.")
        print(f"The hybrid uncertainty-MCTS proposal does not have a viable signal source.")
    else:
        print(f"VERDICT: PARK — Gate 2 FAILED. selective RMS {rms_sel:.5f} >= random RMS {rms_rnd:.5f}.")
        print(f"Even though σ_ens correlates with residuals (corr={corr_sigma_resid:.4f}), "
              f"selective correction does NOT produce a better fit than random correction.")
        print(f"The same outlier-bias problem from Gate 3 may be at play.")
    print(f"Wall-clock: {elapsed:.1f}s")
    print("=" * 72)

    return gate1_pass and gate2_pass


if __name__ == "__main__":
    main()
