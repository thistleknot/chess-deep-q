"""Overnight chain (operator: "kick off training from best parms after optuna finishes").

Waits for the tuner log's `best elo ... @ {params}` line, composes the final-run env from
those params on the Kanerva full-stack regime, seeds the `nk` lineage from the clean seed,
and launches qlearn DETACHED with the console's pid-file + default metrics so the UI's
status/stop/live-plots/ladder all track it. One shot; exits after launching.

Usage: python night_chain.py <tuner_log> [epoch_games=200] [max_epochs=30]
"""
import ast
import os
import re
import subprocess
import sys
import time

import torch

LOG = sys.argv[1]
EPOCH_GAMES = sys.argv[2] if len(sys.argv) > 2 else "200"
MAX_EPOCHS = sys.argv[3] if len(sys.argv) > 3 else "30"
BEST = re.compile(r"best elo ([0-9.]+) @ (\{.*\})")

while True:
    try:
        m = BEST.search(open(LOG, encoding="utf-8", errors="replace").read())
        if m:
            break
    except OSError:
        pass
    time.sleep(30)

elo, p = float(m.group(1)), ast.literal_eval(m.group(2))
print(f"study best {elo:.0f}; composing final run", flush=True)

ck = torch.load("models/qlearn_nk_seed.pt", map_location="cpu")
torch.save(ck, "models/qlearn_nk.pt")
torch.save(ck, "models/qlearn_nk_best.pt")

a_s = 1.0 - p["b_srch"] - p["c_start"]
a_e = 1.0 - p["b_srch"] - p["c_end"]
env = dict(os.environ,
           QLEARN_ALPHA=f"{p['alpha']:.6f}", QLEARN_GAMMA="1.0", QLEARN_LAMBDA="0.7",
           QLEARN_WARMUP=f"{p['warmup']:.4f}", QLEARN_LAMBDA_WARMUP=f"{p['lambda_warmup']:.4f}",
           QLEARN_TAU_FLOOR=f"{p['tau_floor']:.4f}", QLEARN_PATIENCE="8",
           QLEARN_ELO_GAMES="20", QLEARN_EPOCH_ELO_GAMES="24", QLEARN_LOG_EVERY="100",
           QLEARN_BATCH_GAMES="20", QLEARN_FREEZE_EPOCH="1", QLEARN_RESUME="1",
           QLEARN_ANCHOR="1", QLEARN_TDLEAF="1", QLEARN_OPP="graded", QLEARN_ENC="nk",
           QLEARN_CONFIRM="1", QLEARN_RAMP="1", QLEARN_KC_FAITHFUL="1",
           QLEARN_RSEARCH_DEPTH="0", QLEARN_PARGEN="0",
           QLEARN_ZCA="models/kanerva_zca.npz",
           QLEARN_TRIVIUM=f"{a_s:.3f},{p['b_srch']:.3f},{p['c_start']:.3f}",
           QLEARN_TRIVIUM_END=f"{a_e:.3f},{p['b_srch']:.3f},{p['c_end']:.3f}",
           QLEARN_TRIVIUM_WARMUP=f"{p['triv_warmup']:.4f}",
           QLEARN_SURPRISE="1", QLEARN_GRPO="1", QLEARN_DDQN="1",
           QLEARN_REPLAY_T=f"{p.get('replay_t', 1.0):.3f}",
           QLEARN_SURPRISE_K=f"{p.get('surprise_k', 32.0):.1f}",
           QLEARN_PROXY_GAMES="4", QLEARN_DEV="cpu", QLEARN_TAG="final",
           QLEARN_CKPT="models/qlearn_nk.pt")
flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
log = open("data/train.log", "w")
proc = subprocess.Popen([sys.executable, "qlearn.py", EPOCH_GAMES, MAX_EPOCHS],
                        env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
with open("data/train.pid", "w") as fh:
    fh.write(str(proc.pid))
print(f"FINAL RUN LAUNCHED pid {proc.pid} (console-tracked via data/train.pid)", flush=True)
