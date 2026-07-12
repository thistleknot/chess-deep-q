"""Merge 17 chain: wait for nk completion -> mlp study (3 trials) -> final 30-epoch mlp run.
Operator-authorized end-to-end ("fine do it"); termination clause applies to the RESULT."""
import ast, os, re, subprocess, sys, time
def wait_for(path, pat):
    rx = re.compile(pat)
    while True:
        try:
            m = rx.search(open(path, encoding="utf-8", errors="replace").read())
            if m:
                return m
        except OSError:
            pass
        time.sleep(60)
wait_for("data/train.log", r"KILL-CHECK")
print("nk completion done; launching mlp study", flush=True)
env = dict(os.environ, QLEARN_ENC="kc", QLEARN_SURPRISE="1", QLEARN_GRPO="1", QLEARN_DDQN="1",
           QLEARN_REPLAY_TUNE="1", QLEARN_OPP="graded", QLEARN_TDLEAF="1", QLEARN_TRIV_TUNE="1",
           QLEARN_KC_FAITHFUL="1", QLEARN_RAMP="1", QLEARN_RSEARCH_DEPTH="0", QLEARN_PARGEN="0",
           QLEARN_CONFIRM="1", QLEARN_PROXY_GAMES="4", QLEARN_DEV="cpu")
with open("data/tune_m17.log", "w") as lg:
    subprocess.run([sys.executable, "tune_qlearn.py", "3", "100", "1", "12", "20", "mlp", "64", "q"],
                   env=env, stdout=lg, stderr=subprocess.STDOUT)
m = wait_for("data/tune_m17.log", r"best elo ([0-9.]+) @ (\{.*\})")
p = ast.literal_eval(m.group(2))
print(f"study best {m.group(1)}; launching final mlp run", flush=True)
a_s, a_e = 1.0 - p["b_srch"] - p["c_start"], 1.0 - p["b_srch"] - p["c_end"]
env2 = dict(os.environ,
            QLEARN_ARCH="mlp", QLEARN_HIDDEN="64",
            QLEARN_ALPHA=f"{p['alpha']:.6f}", QLEARN_GAMMA="1.0", QLEARN_LAMBDA="0.7",
            QLEARN_WARMUP=f"{p['warmup']:.4f}", QLEARN_LAMBDA_WARMUP=f"{p['lambda_warmup']:.4f}",
            QLEARN_TAU_FLOOR=f"{p['tau_floor']:.4f}", QLEARN_PATIENCE="99",
            QLEARN_ELO_GAMES="20", QLEARN_EPOCH_ELO_GAMES="24", QLEARN_LOG_EVERY="100",
            QLEARN_BATCH_GAMES="20", QLEARN_FREEZE_EPOCH="1", QLEARN_RESUME="0",
            QLEARN_ANCHOR="1", QLEARN_TDLEAF="1", QLEARN_OPP="graded", QLEARN_ENC="kc",
            QLEARN_CONFIRM="1", QLEARN_RAMP="1", QLEARN_KC_FAITHFUL="1",
            QLEARN_RSEARCH_DEPTH="0", QLEARN_PARGEN="0",
            QLEARN_TRIVIUM=f"{a_s:.3f},{p['b_srch']:.3f},{p['c_start']:.3f}",
            QLEARN_TRIVIUM_END=f"{a_e:.3f},{p['b_srch']:.3f},{p['c_end']:.3f}",
            QLEARN_TRIVIUM_WARMUP=f"{p['triv_warmup']:.4f}",
            QLEARN_SURPRISE="1", QLEARN_GRPO="1", QLEARN_DDQN="1",
            QLEARN_REPLAY_T=f"{p.get('replay_t', 1.0):.3f}",
            QLEARN_SURPRISE_K=f"{p.get('surprise_k', 32.0):.1f}",
            QLEARN_PROXY_GAMES="4", QLEARN_DEV="cpu", QLEARN_TAG="final",
            QLEARN_CKPT="models/qlearn_mlp.pt")
flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
log = open("data/train.log", "w")
proc = subprocess.Popen([sys.executable, "qlearn.py", "200", "30"],
                        env=env2, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
open("data/train.pid", "w").write(str(proc.pid))
print(f"FINAL MLP RUN LAUNCHED pid {proc.pid} — full 30 epochs, patience deferred", flush=True)
