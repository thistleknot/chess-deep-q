import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Optuna (TPE) tuning of the two-dial hybrid (λ, τ) — literature-grounded space, noise-adequate n.

Method (memory: hyperparam-tuning-method): ground the search space in the literature, use TPE (not
grid), and set n high enough that the per-trial effect exceeds the objective noise. The λ range is
the TD(λ) sweet-spot band — TD-Gammon ≈0.7 (low), GAE ≈0.95 (high), commonly ≈0.83–0.85; λ=1 (MC) is
known-subpar so the range is capped below 1. Objective = the kill-check score vs pst @d2 from
`train_linear_rl` run as a SUBPROCESS (fresh env per trial → multiprocessing-spawn-safe). SQLite-
persisted (resumable).

Usage: python tune_lambda.py [n_trials] [iters] [games] [kill_games]
"""
import os
import re
import subprocess
import sys

import optuna

ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
GAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 60
KILL_GAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 60      # n high enough to beat the ~1-SD noise
_SCORE = re.compile(r"KILL-CHECK.*?score\s+([0-9.]+)")


def _run(lam, tau):
    """One hybrid training run (subprocess, fresh env) → kill-check score vs pst @d2."""
    env = dict(os.environ, LINEAR_TARGET="lambda", LINEAR_LAMBDA=f"{lam:.4f}",
               LINEAR_TAU=f"{tau:.4f}", KILL_GAMES=str(KILL_GAMES))
    env.pop("LINEAR_FEATURES", None)                           # material+PST φ
    env.pop("LINEAR_REDUCE", None)
    out = subprocess.run([sys.executable, "experiments/train_linear_rl.py", str(ITERS), str(GAMES)],
                         env=env, capture_output=True, text=True, timeout=2400).stdout
    m = _SCORE.search(out)
    return float(m.group(1)) if m else 0.0


def objective(trial):
    lam = trial.suggest_float("lambda", 0.70, 0.97)            # TD(λ) sweet-spot band (literature)
    tau = trial.suggest_float("tau", 0.2, 0.8)                 # co-tuned exploration temperature
    score = _run(lam, tau)
    print(f"  trial {trial.number}: lam={lam:.3f} tau={tau:.3f} -> kill-check {score:.3f}", flush=True)
    return score


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs("models", exist_ok=True)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        storage="sqlite:///models/optuna_lambda.db", study_name="lambda_tau_hybrid",
        load_if_exists=True)
    print(f"Optuna TPE: {n_trials} trials | lam in [0.70,0.97] x tau in [0.2,0.8] | objective = "
          f"kill-check score vs pst@d2 (n={KILL_GAMES}) | {ITERS}x{GAMES} train\n", flush=True)
    study.optimize(objective, n_trials=n_trials)
    print("\n=== best ===", flush=True)
    print(f"  best kill-check {study.best_value:.3f} at lam={study.best_params['lambda']:.3f} "
          f"tau={study.best_params['tau']:.3f}", flush=True)
    print("  top 5 trials:", flush=True)
    for t in sorted((t for t in study.trials if t.value is not None), key=lambda t: -t.value)[:5]:
        print(f"    {t.value:.3f}  lam={t.params['lambda']:.3f} tau={t.params['tau']:.3f}", flush=True)


if __name__ == "__main__":
    main()
