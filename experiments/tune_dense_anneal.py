import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Optuna (TPE) tuning of the dense-reward-anneal coefficient/schedule, RESUMED from the real
champion (models/champion.pt, amap-897, floor 1724 / measured 1878) every trial — not a fresh net.

Mirrors tune_qlearn.py's TDLEAF resume-seed branch (reseed a scratch ckpt from the champion at the
start of every trial -> QLEARN_RESUME=1 -> hermetic, never touches models/champion.pt). Objective =
KILL-CHECK elo after a short, bounded epoch budget (default 3, matching the console's historical
3-epoch/3-trial tuning precedent) with QLEARN_EPOCH_ELO_GAMES=24 (matched to the champion's own
measurement basis, per the smoke/control diagnostic run) and QLEARN_PATIENCE >= MAX_EPOCHS so a
trial's own bounded budget completes rather than early-stopping against the historical 24.48 bar
(any epoch with nonzero dense coefficient is expected to score under that bar by construction).

Tuned:
  dense_rw_start [0.0, 0.5]   material-shaping coefficient at epoch 1 (0 = shaping off, replicates
                               today's sparse-only regime as a control point in the search space)
  dense_rw_warmup [0.1, 0.8]  epoch-1-scoped anneal fraction (fraction of epoch 1 before the
                               coefficient starts decaying toward the floor)

Fixed (not tuned): dense_rw floor = 0 (must land back on sparse-only), dense_rw_scale = 39.0
(material_points normalization, unrelated to the anneal shape).

Usage: python tune_dense_anneal.py [n_trials] [sample_games] [max_epochs] [elo_games]
"""
import os
import re
import sys
import shutil
import hashlib
import subprocess

import optuna

N_TRIALS    = int(sys.argv[1]) if len(sys.argv) > 1 else 12
EPOCH_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 300
MAX_EPOCHS  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ELO_GAMES   = int(sys.argv[4]) if len(sys.argv) > 4 else 20
EPOCH_ELO_GAMES = os.environ.get("QLEARN_EPOCH_ELO_GAMES", "24")   # matched to champion's bar 24.48

CHAMPION_SEED = "models/champion.pt"
TRIAL_CKPT    = "models/qlearn_dense_optuna.pt"

_ELO = re.compile(r"KILL-CHECK\s+elo\s+(-?[0-9.]+)")

PRIOR = {"dense_rw_start": 0.02, "dense_rw_warmup": 0.3}   # the value already smoke/control-tested

SPACE  = "dense_rw_start[0.0,0.5]|dense_rw_warmup[0.1,0.8]"
REGIME = "resume-champ|dense-reward-anneal|patience-ge-epochs|v3-discriminating"
PROTO  = f"s{EPOCH_GAMES}-e{MAX_EPOCHS}-elo{ELO_GAMES}-amap897"
STUDY  = "qlearn_elo_" + hashlib.sha1(f"{SPACE}|{REGIME}|{PROTO}".encode()).hexdigest()[:8]


def run_trial(p, trial_no):
    """Reseed the scratch ckpt from the real champion, resume-train MAX_EPOCHS, return KILL-CHECK elo."""
    shutil.copyfile(CHAMPION_SEED, TRIAL_CKPT)
    shutil.copyfile(CHAMPION_SEED, TRIAL_CKPT.replace(".pt", "_best.pt"))
    env = dict(os.environ,
               QLEARN_CKPT=TRIAL_CKPT, QLEARN_RESUME="1", QLEARN_ENC="amap",
               QLEARN_EPOCH_GAMES=str(EPOCH_GAMES), QLEARN_MAX_EPOCHS=str(MAX_EPOCHS),
               QLEARN_ELO_GAMES=str(ELO_GAMES), QLEARN_EPOCH_ELO_GAMES=EPOCH_ELO_GAMES,
               QLEARN_PATIENCE=str(MAX_EPOCHS + 5),   # trial's own budget must complete, not early-stop
               QLEARN_SEED=str(trial_no), QLEARN_TAG=f"trial{trial_no}",
               QLEARN_DENSE_RW_START=f"{p['dense_rw_start']:.4f}",
               QLEARN_DENSE_RW=str(0.0), QLEARN_DENSE_RW_WARMUP=f"{p['dense_rw_warmup']:.4f}")
    proc = subprocess.run([sys.executable, "-m", "chessdq.qlearn"],
                          env=env, capture_output=True, text=True, timeout=3600)
    m = _ELO.search(proc.stdout)
    if not m:
        print(f"trial {trial_no}: NO KILL-CHECK LINE (returncode={proc.returncode})", flush=True)
        print("--- stdout tail ---", flush=True)
        print("\n".join(proc.stdout.splitlines()[-40:]), flush=True)
        print("--- stderr tail ---", flush=True)
        print("\n".join(proc.stderr.splitlines()[-40:]), flush=True)
        return -999.0
    return float(m.group(1))


def objective(trial):
    p = {
        "dense_rw_start":  trial.suggest_float("dense_rw_start", 0.0, 0.5),
        "dense_rw_warmup": trial.suggest_float("dense_rw_warmup", 0.1, 0.8),
    }
    elo = run_trial(p, trial.number)
    print(f"trial {trial.number}: elo {elo:.1f}  {p}", flush=True)
    return elo


def main():
    os.makedirs("models", exist_ok=True)
    storage = "sqlite:///models/qlearn_optuna.db"
    prior_n = 0
    try:
        prior_n = next((s.n_trials for s in optuna.get_all_study_summaries(storage)
                        if s.study_name == STUDY), 0)
    except Exception:
        pass
    study = optuna.create_study(
        study_name=STUDY, direction="maximize",
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=prior_n),
    )
    done = [t for t in study.trials if t.value is not None]
    if done:
        print(f"RESUMING {STUDY} ({PROTO}): {len(done)} prior trial(s), best {study.best_value:.1f}",
              flush=True)
    else:
        print(f"new study {STUDY} ({PROTO})", flush=True)
        study.enqueue_trial(PRIOR)
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"\nbest elo {study.best_value:.1f} @ {study.best_params}")


if __name__ == "__main__":
    main()
