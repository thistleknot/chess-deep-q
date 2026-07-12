"""Feature-set bake-off (operator: "optuna to identify the feature set — 3 trials per").
Sequential 3-trial studies, ORGAN-FREE baseline config (isolation law), one per feature set:
  kc-raw   : 809 donor features, no whitening      (seed: clean raw)
  kc-zca   : 809 + self-ZCA                        (seed: self-whitened)
  pca-320  : whitened-PCA top-320                  (seed: pca)
  nk-512   : 5k -> Kanerva-512 -> ZCA              (seed: nk)
Each study <= 30-min cap (s100 e1). Results: data/bake_<set>.log; bests feed compare.html.
"""
import os
import subprocess
import sys

SETS = [
    ("kc-raw", dict(QLEARN_ENC="kc", QLEARN_ZCA="", QLEARN_TUNE_SEED="models/qlearn_clean_seed.pt")),
    ("kc-zca", dict(QLEARN_ENC="kc", QLEARN_ZCA="models/zca_self.npz", QLEARN_TUNE_SEED="models/qlearn_self_seed.pt")),
    ("pca-320", dict(QLEARN_ENC="kc", QLEARN_ZCA="models/pca_self.npz", QLEARN_TUNE_SEED="models/qlearn_pca_seed.pt")),
    ("nk-512", dict(QLEARN_ENC="nk", QLEARN_ZCA="models/kanerva_zca.npz", QLEARN_TUNE_SEED="models/qlearn_nk_seed.pt")),
]
BASE = dict(QLEARN_SURPRISE="0", QLEARN_GRPO="0", QLEARN_DDQN="0", QLEARN_DECK="0",
            QLEARN_OPP="graded", QLEARN_TDLEAF="1", QLEARN_TRIV_TUNE="1",
            QLEARN_KC_FAITHFUL="1", QLEARN_RAMP="1", QLEARN_RSEARCH_DEPTH="0",
            QLEARN_PARGEN="0", QLEARN_CONFIRM="1", QLEARN_PROXY_GAMES="4", QLEARN_DEV="cpu")
for name, extra in SETS:
    env = dict(os.environ, **BASE, **extra)
    print(f"=== bake-off study: {name} ===", flush=True)
    with open(f"data/bake_{name}.log", "w") as lg:
        subprocess.run([sys.executable, "tune_qlearn.py", "3", "100", "1", "12", "20", "linear", "64", "q"],
                       env=env, stdout=lg, stderr=subprocess.STDOUT)
print("bake-off complete", flush=True)
