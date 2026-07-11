"""Interactive training console for the RL ladder (spec/observability.spec.md + q-learning.spec.md).

FastAPI server. Boot with `python app.py` (opens the browser). Beyond the baseline/measure buttons it now
drives Q-learning training: a settings form PRELOADED from the Optuna best, editable, with a start/stop and
LIVE plots (loss, points/material, turns/length, strength) polled from the training run's metrics.

  GET  /                  -> the console
  GET  /api/trend         -> Elo-ladder rows (data/rl_trend.jsonl)
  POST /api/baseline      -> chess_rl.py N random games
  POST /api/measure       -> measure_elo.py N games (append the ladder)
  GET  /api/optuna/best   -> best tuned {decay,alpha,lambda,warmup} from the study
  POST /api/train/start   -> launch qlearn.py DETACHED with the posted settings
  POST /api/train/stop    -> terminate the active training run
  GET  /api/train/status  -> is a run active?
  GET  /api/train/metrics -> per-epoch training curve (data/qlearn_metrics.jsonl) + final result
"""
import os
import sys
import json
import subprocess

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()
ROOT = os.path.dirname(os.path.abspath(__file__))
TREND = os.path.join(ROOT, "data", "rl_trend.jsonl")
METRICS = os.path.join(ROOT, "data", "qlearn_metrics.jsonl")
RESULTS = os.path.join(ROOT, "data", "qlearn_results.jsonl")
PY = sys.executable
TRAIN_PROC = None   # in-memory handle when THIS server instance spawned the run
TRAIN_PID_FILE = os.path.join(ROOT, "data", "train.pid")   # survives server restarts -> re-attach


def _pid_alive(pid):
    """Is the process alive? Windows: OpenProcess+GetExitCodeProcess (os.kill(pid,0) is NOT a probe on
    win32 — it TerminateProcess-es). POSIX: signal 0."""
    if pid is None:
        return False
    pid = int(pid)
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259            # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _active_pid():
    """PID of the live training run, whether spawned by this server instance or a previous one."""
    if TRAIN_PROC is not None and TRAIN_PROC.poll() is None:
        return TRAIN_PROC.pid
    try:
        with open(TRAIN_PID_FILE) as fh:
            pid = int(fh.read().strip())
        return pid if _pid_alive(pid) else None
    except (OSError, ValueError):
        return None


class RunReq(BaseModel):
    n: int = 20


class TrainReq(BaseModel):
    alpha: float = 3e-3
    gamma: float = 0.99
    lam: float = 0.8
    warmup: float = 0.4
    lambda_warmup: float = 0.4    # λ's e-folding fraction (shared asymptotic anneal with τ)
    epoch_games: int = 200
    max_epochs: int = 5
    patience: int = 2
    elo_games: int = 20
    adaptive_lambda: bool = True
    epoch_elo_games: int = 12     # cheap live Elo vs SF each epoch (0 = off)
    samples_per_epoch: int = 2    # plotted/measured metric points per epoch -> QLEARN_LOG_EVERY
    buffer_epochs: float = 0      # replay window in EPOCHS of positions (policy-admixture horizon); 0 = 100k cap
    batch_games: int = 20         # BATCH SIZE (games) per gradient cycle; epoch_games/batch_games = num batches
    freeze_epoch: bool = True     # generation+bootstrap from a frozen copy, synced at epoch boundaries
    resume: bool = False          # continue from models/qlearn.pt (weights+optimizer+schedule progress)
    anchor: bool = True           # AlphaZero-style gate: revert non-improving epochs to the best checkpoint
    curriculum: float = 0.0       # exploring-starts fraction (ac): episodes from mate-dense reduced positions
    pg_discount: bool = True      # (ac) textbook gamma^t policy-gradient weighting; off = practical A2C
    adv_norm: bool = False        # (ac) normalize actor advantage by running sigma(delta) — scale-free PG
    actor_arch: str = "linear"    # (ac) actor head: linear | mlp (the measured sharpness wall)
    behavior: str = "softmax"     # (q) softmax = 1-ply | search = depth-2 negamax over V (Merge 4)
    search_width: int = 8         # (q/search) top-K root moves expanded
    search_depth: int = 2         # (q/search/tdleaf) negamax depth for behavior + measurement (2|3)
    tdleaf: bool = False          # (q) Merge 5 TDLeaf(λ) (spec/tdleaf.spec.md): generation plays the
                                  # search policy; V trains on PV-LEAF states toward minimax targets
    opp: str = "self"             # (q) generation opponents: self-play | graded ladder with
                                  # matchmaking (spec :Graded-opponents:, KnightCap's headline)
    enc: str = "pst"              # (q) input encoding: pst = raw 769 planes | kc = Merge 6
                                  # KnightCap-donor features, 809-dim (spec/eval-features.spec.md)
    confirm: bool = True          # (q) spec :Confirmed-crown:: candidate bests must survive an
                                  # independent re-measure before the ratchet crowns them
    opp_reach: float = 0.0        # (q) spec :Opponent-diet:: probability a graded generation game
                                  # plays the rung ABOVE the current matchmaking rung (reach games)
    ramp: bool = False            # (q) spec :Ramp-filter:: zero favorable TDs on unpredicted moves
    kc_faithful: bool = False     # (q) Merge 7 :Faithful-mode:: the full KnightCap recipe (implies
                                  # tdleaf+ramp, online per-game SGD, fixed lam=.7 gamma=1, greedy)
    rsearch_depth: int = 0        # (q) Merge 8: >0 = native full-width alpha-beta+quiescence at
                                  # this depth for generation moves, TDLeaf targets, predictions
    trivium: str = ""             # (q) compound target "a,b,c" = λ-return : search value : outcome
                                  # (S&B §12 averaged backups / KataGo mixed targets); "" = off
    proxy_games: int = 20         # greedy eval games per sample (search runs: lower = cheaper samples)
    device: str = ""              # "" = trainer default; "cpu" recommended for search (small batches)
    lineage: str = ""             # checkpoint lineage name -> models/<algo>_<lineage>.pt; isolates
                                  # concurrent experiment lines (a fresh run only deletes ITS lineage's best)
    # --- advanced knobs (API-only; the form doesn't render them, pydantic defaults apply) ---
    algo: str = "q"               # q = Merge 2 Q-learning | ac = Merge 3 online actor-critic
    alpha_theta: float = 1e-4     # (ac) actor step size — slower than critic (two-timescale)
    lambda_theta: float = 0.8     # (ac) actor trace decay
    entropy_beta: float = 0.01    # (ac) entropy bonus start (anneals to /10)
    arch: str = "linear"          # ValueNet arch: linear | mlp (same TD(λ), different V capacity)
    hidden: int = 64              # mlp hidden width
    tau_start: float = 0.7
    tau_floor: float = 0.05
    lambda_start: float = 0.95
    k_adapt: float = 0.5
    train_steps: int = 200


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # tolerate a half-written last line during a live run
    return rows


def load_rows():
    return _read_jsonl(TREND)


def run_script(args, timeout):
    p = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return {"stdout": (p.stdout or "")[-6000:], "stderr": (p.stderr or "")[-1500:], "code": p.returncode}


@app.get("/api/trend")
def api_trend():
    return {"rows": load_rows()}


@app.post("/api/baseline")
def api_baseline(req: RunReq):
    out = run_script(["chess_rl.py", str(req.n), "200"], timeout=1800)
    out["rows"] = load_rows()
    return out


@app.post("/api/measure")
def api_measure(req: RunReq):
    out = run_script(["measure_elo.py", str(req.n)], timeout=1800)
    out["rows"] = load_rows()
    return out


@app.get("/api/optuna/best")
def api_optuna_best():
    try:
        import optuna
        storage = "sqlite:///models/qlearn_optuna.db"
        # study names are FINGERPRINTED by (search space, regime, protocol) so runs resume; serve the
        # best of the MOST RECENTLY ACTIVE study = the current protocol's study
        best_study, best_ts = None, None
        for s in optuna.get_all_study_summaries(storage):
            st = optuna.load_study(study_name=s.study_name, storage=storage)
            ts = max((t.datetime_start for t in st.trials if t.datetime_start), default=None)
            if ts is not None and (best_ts is None or ts > best_ts):
                best_study, best_ts = st, ts
        if best_study is None:
            return {"ok": False, "msg": "no studies yet"}
        done = [t for t in best_study.trials if t.value is not None]
        if not done:
            return {"ok": False, "msg": "no completed trials yet"}
        return {"ok": True, "best": best_study.best_params, "elo": round(best_study.best_value),
                "n": len(done), "study": best_study.study_name}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


@app.post("/api/train/start")
def api_train_start(cfg: TrainReq):
    global TRAIN_PROC
    if _active_pid() is not None:
        return {"ok": False, "msg": "a training run is already active"}
    if cfg.batch_games > cfg.epoch_games:
        return {"ok": False, "msg": "batch size must be <= sample size (games/epoch)"}
    env = dict(os.environ,
               QLEARN_ALPHA=f"{cfg.alpha:.6f}", QLEARN_GAMMA=f"{cfg.gamma:.4f}",
               QLEARN_LAMBDA=f"{cfg.lam:.4f}", QLEARN_WARMUP=f"{cfg.warmup:.4f}",
               QLEARN_LAMBDA_WARMUP=f"{cfg.lambda_warmup:.4f}",
               QLEARN_PATIENCE=str(cfg.patience), QLEARN_ELO_GAMES=str(cfg.elo_games),
               QLEARN_ADAPTIVE_LAMBDA="1" if cfg.adaptive_lambda else "0",
               QLEARN_EPOCH_ELO_GAMES=str(cfg.epoch_elo_games),
               QLEARN_LOG_EVERY=str(max(1, cfg.epoch_games // max(1, cfg.samples_per_epoch))),
               QLEARN_BUFFER_EPOCHS=f"{cfg.buffer_epochs:g}",
               QLEARN_BATCH_GAMES=str(cfg.batch_games),
               QLEARN_FREEZE_EPOCH="1" if cfg.freeze_epoch else "0",
               QLEARN_RESUME="1" if cfg.resume else "0",
               QLEARN_ANCHOR="1" if cfg.anchor else "0",
               QLEARN_CURRICULUM=f"{cfg.curriculum:.3f}",
               QLEARN_PG_DISCOUNT="1" if cfg.pg_discount else "0",
               QLEARN_ADV_NORM="1" if cfg.adv_norm else "0",
               QLEARN_ACTOR_ARCH=cfg.actor_arch,
               QLEARN_BEHAVIOR=cfg.behavior, QLEARN_SEARCH_WIDTH=str(cfg.search_width),
               QLEARN_SEARCH_DEPTH=str(cfg.search_depth),
               QLEARN_TDLEAF="1" if cfg.tdleaf else "0",
               QLEARN_OPP=cfg.opp, QLEARN_ENC=cfg.enc,
               QLEARN_CONFIRM="1" if cfg.confirm else "0",
               QLEARN_OPP_REACH=f"{cfg.opp_reach:.3f}",
               QLEARN_RAMP="1" if cfg.ramp else "0",
               QLEARN_KC_FAITHFUL="1" if cfg.kc_faithful else "0",
               QLEARN_RSEARCH_DEPTH=str(cfg.rsearch_depth),
               QLEARN_TRIVIUM=cfg.trivium,
               QLEARN_PROXY_GAMES=str(cfg.proxy_games),
               **({"QLEARN_DEV": cfg.device} if cfg.device else {}),
               QLEARN_CKPT=(f"models/{'ac' if cfg.algo == 'ac' else 'qlearn'}_"
                            f"{cfg.lineage}.pt" if cfg.lineage else
                            ("models/ac_learn.pt" if cfg.algo == "ac" else "models/qlearn.pt")),
               QLEARN_ARCH=cfg.arch, QLEARN_HIDDEN=str(cfg.hidden), QLEARN_TAU_START=f"{cfg.tau_start:.4f}",
               QLEARN_TAU_FLOOR=f"{cfg.tau_floor:.4f}", QLEARN_LAMBDA_START=f"{cfg.lambda_start:.4f}",
               QLEARN_K_ADAPT=f"{cfg.k_adapt:.4f}", QLEARN_TRAIN_STEPS=str(cfg.train_steps),
               # (ac) mapping: alpha -> critic step, lam -> critic trace; actor knobs below
               QLEARN_ALPHA_W=f"{cfg.alpha:.6f}", QLEARN_LAMBDA_W=f"{cfg.lam:.4f}",
               QLEARN_ALPHA_TH=f"{cfg.alpha_theta:.6f}", QLEARN_LAMBDA_TH=f"{cfg.lambda_theta:.4f}",
               QLEARN_BETA=f"{cfg.entropy_beta:.5f}",
               QLEARN_TAG="final")
    script = "ac_learn.py" if cfg.algo == "ac" else "qlearn.py"
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    log = open(os.path.join(ROOT, "data", "train.log"), "w")
    # ACTUALLY detach on Windows: without flags the trainer is a plain child of the server and dies with
    # it (a server restart killed a live run). CREATE_NO_WINDOW gives the trainer its OWN windowless
    # console — NOT DETACHED_PROCESS, which made the child interpreter allocate a VISIBLE console window
    # that a user could close, aborting the run (forrtl error 200: window-CLOSE event).
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
    TRAIN_PROC = subprocess.Popen([PY, script, str(cfg.epoch_games), str(cfg.max_epochs)],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=flags)
    with open(TRAIN_PID_FILE, "w") as fh:
        fh.write(str(TRAIN_PROC.pid))
    return {"ok": True, "pid": TRAIN_PROC.pid}


@app.post("/api/train/stop")
def api_train_stop():
    global TRAIN_PROC
    pid = TRAIN_PROC.pid if (TRAIN_PROC is not None and TRAIN_PROC.poll() is None) else _active_pid()
    if pid is None:
        return {"ok": False, "msg": "no active run"}
    # Kill the WHOLE TREE: the venv launcher (py310\Scripts\python.exe) runs the real interpreter as a
    # CHILD; terminating only the launcher orphans that child, which keeps training (the zombie-trainer
    # bug that double-wrote qlearn_metrics.jsonl).
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        os.kill(pid, signal.SIGTERM)
    return {"ok": True, "msg": f"terminated tree (pid {pid})"}


@app.get("/api/train/status")
def api_train_status():
    pid = _active_pid()
    return {"running": pid is not None, "pid": pid}


@app.get("/api/train/metrics")
def api_train_metrics():
    metrics = _read_jsonl(METRICS)
    results = _read_jsonl(RESULTS)
    result = results[-1] if results else None
    try:                                   # run start = pid-file mtime -> lets the UI hide STALE results
        run_start = int(os.path.getmtime(TRAIN_PID_FILE))
    except OSError:
        run_start = None
    return {"metrics": metrics, "result": result, "running": _active_pid() is not None,
            "run_start": run_start}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>chess RL — training console</title>
<style>
  :root{color-scheme:dark;}
  body{background:#0d1117;color:#e6edf3;font:14px/1.5 system-ui,sans-serif;margin:0;padding:28px;max-width:1100px;}
  h1{font-size:20px;margin:0 0 4px;} .sub{color:#7d8590;margin:0 0 20px;}
  .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:20px;}
  .card h2{font-size:13px;margin:0 0 14px;color:#7d8590;text-transform:uppercase;letter-spacing:.04em;}
  .form{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:14px;}
  label{display:block;color:#7d8590;font-size:12px;margin-bottom:3px;}
  input[type=number]{width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:7px;}
  button{background:#238636;color:#fff;border:0;border-radius:7px;padding:9px 15px;font-size:14px;font-weight:600;cursor:pointer;}
  button:hover{background:#2ea043;} button.alt{background:#1f6feb;} button.alt:hover{background:#388bfd;}
  button.grey{background:#30363d;} button.stop{background:#da3633;} button:disabled{opacity:.5;cursor:not-allowed;}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
  #status{color:#7d8590;margin-left:6px;}
  .plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;}
  .plot h3{font-size:12px;margin:0 0 6px;color:#7d8590;} .plot .now{float:right;color:#e6edf3;font-weight:600;}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin-bottom:18px;}
  .tile{background:#0d1117;border:1px solid #30363d;border-radius:9px;padding:13px 15px;}
  .tile .k{color:#7d8590;font-size:11px;text-transform:uppercase;} .tile .v{font-size:24px;font-weight:600;margin-top:4px;}
  svg{width:100%;height:90px;} code{color:#58a6ff;}
  table.lad{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px;}
  table.lad th{color:#7d8590;text-align:left;font-weight:600;padding:4px 8px;border-bottom:1px solid #30363d;}
  table.lad td{padding:4px 8px;border-bottom:1px solid #21262d;color:#e6edf3;}
  table.lad tr.best td{color:#3fb950;font-weight:600;}
</style></head><body>
  <h1>chess RL — training console</h1>
  <p class="sub">Merge 2 Q-learning · TD(λ)+variance-adaptive λ · softmax-τ self-play · anchor SF@1320<br>
    <b>all live scores are the GREEDY (optimal) policy</b> · cheap metrics land every BATCH (faint = raw, bold = MA-10) · evals per samples/epoch · resets each run<br>
    τ &amp; λ share one asymptotic anneal (fast-then-slow, never reaches the endpoint — exploration room never closes)</p>

  <div class="card"><h2>Settings (load the tuned config, edit, train)</h2>
    <div class="form">
      <div><label>α (step)</label><input id="alpha" type="number" step="0.0001" value="0.003"></div>
      <div><label>γ (decay)</label><input id="gamma" type="number" step="0.001" value="0.99"></div>
      <div><label>λ (trace)</label><input id="lam" type="number" step="0.01" value="0.8"></div>
      <div><label>warmup ratio (τ)</label><input id="warmup" type="number" step="0.01" value="0.4"></div>
      <div><label>warmup ratio (λ)</label><input id="lam_warmup" type="number" step="0.01" value="0.4"></div>
      <div><label>sample size (games/epoch)</label><input id="epoch_games" type="number" value="200"></div>
      <div><label>max epochs (runs)</label><input id="max_epochs" type="number" value="10"></div>
      <div><label>patience</label><input id="patience" type="number" value="2"></div>
      <div><label>Elo games (final)</label><input id="elo_games" type="number" value="20"></div>
      <div><label>Elo games/epoch (live)</label><input id="epoch_elo" type="number" value="12"></div>
      <div><label>samples / epoch (plot)</label><input id="samples" type="number" value="2" min="1"></div>
      <div><label>buffer (epochs, 0=100k cap)</label><input id="buf_epochs" type="number" step="0.5" value="0" min="0"></div>
      <div><label>batch size (games)</label><input id="batch_games" type="number" value="20" min="1"></div>
      <div><label>freeze gen / epoch</label><input id="freeze" type="checkbox" checked style="width:auto"></div>
      <div><label>resume last model</label><input id="resume" type="checkbox" style="width:auto"></div>
      <div><label>anchor to best (gate)</label><input id="anchor" type="checkbox" checked style="width:auto"></div>
      <div><label>TDLeaf(λ) search-gen (M5)</label><input id="tdleaf" type="checkbox" style="width:auto"></div>
      <div><label>opponents (M5)</label><select id="opp" style="width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:7px">
        <option value="self">self-play</option><option value="graded">graded ladder</option></select></div>
      <div><label>encoding (M6)</label><select id="enc" style="width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:7px">
        <option value="pst">pst 769</option><option value="kc">kc features 809</option></select></div>
      <div><label>algo</label><select id="algo" style="width:100%;background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:7px">
        <option value="q">q-learn (M2)</option><option value="ac">actor-critic (M3)</option></select></div>
      <div><label>adaptive λ</label><input id="adaptive" type="checkbox" checked style="width:auto"></div>
    </div>
    <div class="row">
      <button class="alt" onclick="loadBest()">⬇ Load Optuna best</button>
      <button onclick="startTrain()" id="startBtn">▶ Start training</button>
      <button class="stop" onclick="stopTrain()">■ Stop</button>
      <span id="status"></span>
    </div>
  </div>

  <div class="card"><h2>Live training</h2>
    <div class="tiles">
      <div class="tile"><div class="k">Final Elo</div><div class="v" id="t-elo">—</div></div>
      <div class="tile"><div class="k">Epoch</div><div class="v" id="t-epoch">—</div></div>
      <div class="tile"><div class="k">λ (adapted)</div><div class="v" id="t-lam">—</div></div>
      <div class="tile"><div class="k">Buffer</div><div class="v" id="t-buf">—</div></div>
      <div class="tile"><div class="k">Opponent rung (graded)</div><div class="v" id="t-rung">—</div></div>
    </div>
    <div class="plots">
      <div class="plot"><h3>score vs SF@1320 (nominal pts) <span class="now" id="n-sfpts"></span></h3><svg id="p-sfpts"></svg></div>
      <div class="plot"><h3>Elo (live, vs SF@1320) <span class="now" id="n-elo"></span></h3><svg id="p-elo"></svg></div>
      <div class="plot"><h3>loss <span class="now" id="n-loss"></span></h3><svg id="p-loss"></svg></div>
      <div class="plot"><h3>trace σ (td_sigma) <span class="now" id="n-sig"></span></h3><svg id="p-sig"></svg></div>
      <div class="plot"><h3>points (margin vs heuristic, pawns) <span class="now" id="n-pts"></span></h3><svg id="p-pts"></svg></div>
      <div class="plot"><h3>turns (game length) <span class="now" id="n-turns"></span></h3><svg id="p-turns"></svg></div>
      <div class="plot"><h3>strength vs heuristic <span class="now" id="n-str"></span></h3><svg id="p-str"></svg></div>
      <div class="plot"><h3>learned piece worth (pawn=1, line=Q) <span class="now" id="n-pw"></span></h3><svg id="p-pw"></svg></div>
      <div class="plot"><h3>checkmate rate (self-play) <span class="now" id="n-mate"></span></h3><svg id="p-mate"></svg></div>
      <div class="plot"><h3>avg reward (self-play z, mean/sample) <span class="now" id="n-rew"></span></h3><svg id="p-rew"></svg></div>
      <div class="plot"><h3>EPOCH STRENGTH (pooled SF + proxy — the goal curve) <span class="now" id="n-eps"></span></h3><svg id="p-eps"></svg></div>
      <div class="plot"><h3>policy entropy (self-play, AC) <span class="now" id="n-ent"></span></h3><svg id="p-ent"></svg></div>
      <div class="plot"><h3>strength vs random <span class="now" id="n-rnd"></span></h3><svg id="p-rnd"></svg></div>
    </div>
  </div>

  <div class="card"><h2>Search ladder — measured Elo (data/rl_trend.jsonl)</h2>
    <p class="sub" style="margin:0 0 12px">offline measurements vs SF@1320 —
      <span style="color:#3fb950">&#9679; net + search wrapper</span> ·
      <span style="color:#e3b341">&#9679; raw policy</span> ·
      whiskers = 95% CI · <b>this is where search-side progress lands</b> — the live panels above only ever show the raw net</p>
    <div class="tiles">
      <div class="tile"><div class="k">Latest rung</div><div class="v" id="lad-latest">—</div></div>
      <div class="tile"><div class="k">Best rung</div><div class="v" id="lad-best">—</div></div>
      <div class="tile"><div class="k">Rungs measured</div><div class="v" id="lad-n">—</div></div>
    </div>
    <svg id="p-ladder" style="height:180px"></svg>
    <table class="lad" id="lad-table"></table>
  </div>

<script>
function val(id){return document.getElementById(id).value;}
async function loadBest(){
  const r=await fetch('/api/optuna/best'); const j=await r.json();
  if(!j.ok){ document.getElementById('status').textContent='optuna: '+j.msg; return; }
  document.getElementById('alpha').value=j.best.alpha.toFixed(5);
  document.getElementById('gamma').value=j.best.decay.toFixed(4);
  document.getElementById('lam').value=j.best.lambda.toFixed(3);
  document.getElementById('warmup').value=j.best.warmup.toFixed(3);
  document.getElementById('lam_warmup').value=(j.best.lambda_warmup!=null?j.best.lambda_warmup:j.best.warmup).toFixed(3);
  if(j.best.batch_games!=null) document.getElementById('batch_games').value=j.best.batch_games;
  document.getElementById('status').textContent=`loaded best of ${j.n} trials (Elo ${j.elo})`;
}
async function startTrain(){
  const cfg={alpha:+val('alpha'),gamma:+val('gamma'),lam:+val('lam'),warmup:+val('warmup'),lambda_warmup:+val('lam_warmup'),
    epoch_games:+val('epoch_games'),max_epochs:+val('max_epochs'),patience:+val('patience'),
    elo_games:+val('elo_games'),epoch_elo_games:+val('epoch_elo'),samples_per_epoch:+val('samples'),buffer_epochs:+val('buf_epochs'),
    batch_games:+val('batch_games'),freeze_epoch:document.getElementById('freeze').checked,
    resume:document.getElementById('resume').checked,algo:val('algo'),anchor:document.getElementById('anchor').checked,
    adaptive_lambda:document.getElementById('adaptive').checked,tdleaf:document.getElementById('tdleaf').checked,
    opp:val('opp'),enc:val('enc')};
  const r=await fetch('/api/train/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
  const j=await r.json();
  document.getElementById('status').textContent=j.ok?`training started (pid ${j.pid})`:('busy: '+j.msg);
}
async function stopTrain(){
  const r=await fetch('/api/train/stop',{method:'POST'}); const j=await r.json();
  document.getElementById('status').textContent=j.msg;
}
function movavg(vals,w){const out=[];let s=0;for(let i=0;i<vals.length;i++){s+=vals[i];if(i>=w)s-=vals[i-w];out.push(s/Math.min(i+1,w));}return out;}
function spark(id,vals,color){
  // raw per-batch points (faint) + MA(10) trend (bold): batch-scale noise AND epoch-scale drift together
  const svg=document.getElementById(id); if(!vals.length){svg.innerHTML='';return;}
  const w=300,h=90,pad=6, lo=Math.min(...vals), hi=Math.max(...vals), rng=(hi-lo)||1;
  const X=i=>pad+i*((w-2*pad)/Math.max(1,vals.length-1)), Y=v=>h-pad-((v-lo)/rng)*(h-2*pad);
  const path=a=>a.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  const lab=v=>Math.abs(v)>=100?v.toFixed(0):(+v.toFixed(2)).toString();   // actual hi/lo values, not just shape
  svg.innerHTML=`<path d="${path(vals)}" fill="none" stroke="${color}" stroke-width="1" opacity="0.35"/>`
    +vals.map((v,i)=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="1.6" fill="${color}" opacity="0.45"/>`).join('')
    +(vals.length>2?`<path d="${path(movavg(vals,10))}" fill="none" stroke="${color}" stroke-width="2.2"/>`:'')
    +`<text x="3" y="11" fill="#7d8590" font-size="10">${lab(hi)}</text>`
    +`<text x="3" y="${h-2}" fill="#7d8590" font-size="10">${lab(lo)}</text>`;
}
async function poll(){
  try{
    const r=await fetch('/api/train/metrics'); const j=await r.json();
    const m=j.metrics||[];
    const PLOTS=['p-sfpts','p-elo','p-loss','p-sig','p-pts','p-turns','p-str','p-pw','p-rew','p-mate','p-eps','p-ent','p-rnd'];
    const NOWS=['n-sfpts','n-elo','n-loss','n-sig','n-pts','n-turns','n-str','n-pw','n-rew','n-mate','n-eps','n-ent','n-rnd'];
    if(!m.length){
      // FRESH RUN, no samples yet: clear everything so the previous run's curves don't ghost
      for(const id of PLOTS) spark(id,[]);
      for(const id of NOWS.concat(['t-epoch','t-lam','t-buf','t-rung'])) document.getElementById(id).textContent='—';
    }else{
      const last=m[m.length-1];
      document.getElementById('t-epoch').textContent=last.epoch;
      document.getElementById('t-lam').textContent=(+last.lam_eff).toFixed(3);
      document.getElementById('t-buf').textContent=last.buf;
      document.getElementById('t-rung').textContent=last.opp_rung||'—';
      // NOMINAL raw points vs SF (W + D/2 out of n). Legacy rows lack sf_pts -> invert the Elo anchor.
      const sfN=x=>x.sf_n!=null?x.sf_n:12;
      const sfp=m.map(x=>x.sf_pts!=null?x.sf_pts:(x.epoch_elo!=null?+((sfN(x))/(1+Math.pow(10,(1320-x.epoch_elo)/400))).toFixed(2):null)).filter(v=>v!=null);
      spark('p-sfpts',sfp,'#3fb950');
      // ACTUAL Elo on the rating scale: half-point clamp (shutout @12 -> 775), derived from the nominal pts
      const eloOf=(pts,n)=>{const s=Math.min(Math.max(pts/n,1/(2*n)),1-1/(2*n));return Math.round(1320+400*Math.log10(s/(1-s)));};
      document.getElementById('n-sfpts').textContent=sfp.length
        ?`${sfp[sfp.length-1]}/${sfN(last)} = ${eloOf(sfp[sfp.length-1],sfN(last))} Elo`:'—';
      const elos=m.map(x=>x.sf_pts!=null?eloOf(x.sf_pts,sfN(x)):x.epoch_elo).filter(v=>v!=null);
      spark('p-elo',elos,'#e3b341');
      document.getElementById('n-elo').textContent=elos.length?elos[elos.length-1]:'—';
      // eval-cadence series carry null on batch-only rows -> filter; "now" labels = last non-null
      const num=a=>a.filter(v=>v!=null), tail=a=>a.length?a[a.length-1]:'—';
      spark('p-loss',num(m.map(x=>x.loss)),'#f0883e');   document.getElementById('n-loss').textContent=(+tail(num(m.map(x=>x.loss)))).toFixed(4);
      spark('p-sig',num(m.map(x=>x.td_sigma)),'#db61a2');document.getElementById('n-sig').textContent=(+tail(num(m.map(x=>x.td_sigma)))).toFixed(3);
      const pts=num(m.map(x=>x.avg_points)), tns=num(m.map(x=>x.avg_turns)), str=num(m.map(x=>x.wr_vs_heuristic));
      spark('p-pts',pts,'#58a6ff');                      document.getElementById('n-pts').textContent=tail(pts);
      spark('p-turns',tns,'#a371f7');                    document.getElementById('n-turns').textContent=tail(tns);
      spark('p-str',str,'#2ea043');                      document.getElementById('n-str').textContent=str.length?(+tail(str)).toFixed(2):'—';
      // learned piece worth (pawn=1): sparkline tracks the QUEEN's learned value; readout shows all four.
      // Ratios only mean anything once the pawn weight is meaningfully POSITIVE — early near-zero /
      // negative pawn weights make the normalization explode (e.g. N=-99), so those samples are skipped.
      const pvOK=x=>x.piece_vals&&x.piece_vals.Q!=null&&x.piece_vals.pawn_raw>1e-4;
      spark('p-pw',m.filter(pvOK).map(x=>x.piece_vals.Q),'#f778ba');
      const pv=last.piece_vals;
      document.getElementById('n-pw').textContent=pvOK(last)?`N ${pv.N} · B ${pv.B} · R ${pv.R} · Q ${pv.Q}`:'— (pawn weight still ~0)';
      // avg reward: each point is the mean terminal z over that sample's self-play games (a moving average)
      spark('p-rew',m.map(x=>x.avg_reward).filter(v=>v!=null),'#79c0ff');
      document.getElementById('n-rew').textContent=last.avg_reward==null?'—':(+last.avg_reward).toFixed(3);
      // checkmate rate: fraction of that sample's self-play games ending in mate — the OBJECTIVE signal
      // density. Rising = the agent is learning to actually finish games (converging on checkmate).
      spark('p-mate',m.map(x=>x.decisive).filter(v=>v!=null),'#ff7b72');
      document.getElementById('n-mate').textContent=last.decisive==null?'—':(+last.decisive).toFixed(2);
      // THE GOAL CURVE: faint = per-epoch PROPOSALS (may dip — failed ones are discarded work, not
      // lost ground); bold = the ANCHOR (running max = the kept policy) — monotone by construction
      const eps=num(m.map(x=>x.epoch_strength));
      const ratchet=[]; let mx=-1e9; for(const v of eps){mx=Math.max(mx,v); ratchet.push(mx);}
      (function(){const svg=document.getElementById('p-eps'); if(!eps.length){svg.innerHTML='';return;}
        const w=300,h=90,pad=6, lo=Math.min(...eps), hi=Math.max(...eps), rng=(hi-lo)||1;
        const X=i=>pad+i*((w-2*pad)/Math.max(1,eps.length-1)), Y=v=>h-pad-((v-lo)/rng)*(h-2*pad);
        const path=a=>a.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
        svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
        svg.innerHTML=`<path d="${path(eps)}" fill="none" stroke="#e3b341" stroke-width="1" opacity="0.35"/>`
          +eps.map((v,i)=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="2" fill="#e3b341" opacity="0.5"/>`).join('')
          +`<path d="${path(ratchet)}" fill="none" stroke="#3fb950" stroke-width="2.5"/>`
          +`<text x="3" y="11" fill="#7d8590" font-size="10">${(+hi.toFixed(2))}</text>`
          +`<text x="3" y="${h-2}" fill="#7d8590" font-size="10">${(+lo.toFixed(2))}</text>`;})();
      document.getElementById('n-eps').textContent=eps.length?`${tail(eps)} (anchor ${tail(ratchet)})`:'—';
      // AC policy entropy: falling = policy committing; ~0 = premature determinism; flat-max = not learning
      const ent=num(m.map(x=>x.entropy));
      spark('p-ent',ent,'#d2a8ff'); document.getElementById('n-ent').textContent=ent.length?(+tail(ent)).toFixed(3):'—';
      const rnd=num(m.map(x=>x.wr_vs_random));
      spark('p-rnd',rnd,'#56d364'); document.getElementById('n-rnd').textContent=rnd.length?(+tail(rnd)).toFixed(2):'—';
    }
    // Final Elo tile: only THIS run's final — hide results left over from previous runs
    const fresh=j.result && j.result.elo!=null && (!j.run_start || j.result.ts>=j.run_start);
    document.getElementById('t-elo').textContent=fresh
      ?`${j.result.elo}`+(j.result.elo_lo!=null?` (${j.result.elo_lo}..${j.result.elo_hi})`:''):'—';
    document.getElementById('startBtn').disabled=j.running;
    if(j.running) document.getElementById('status').textContent='training…';
  }catch(e){}
  setTimeout(poll,3000);
}
poll();

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
async function pollLadder(){
  try{
    const r=await fetch('/api/trend'); const j=await r.json();
    const rows=(j.rows||[]).filter(x=>x.elo!=null);
    const svg=document.getElementById('p-ladder');
    if(!rows.length){svg.innerHTML=''; document.getElementById('lad-table').innerHTML='';}
    else{
      const isSearch=x=>/search|rsearch/i.test(x.agent||'')&&!/no search|1-?ply|purist/i.test(x.agent||'');
      // ---- chart: one point per measurement, whiskers = CI, color by agent kind ----
      const w=1000,h=180,padL=44,padR=10,padT=12,padB=16;
      const los=rows.map(x=>x.elo_lo!=null?x.elo_lo:x.elo), his=rows.map(x=>x.elo_hi!=null?x.elo_hi:x.elo);
      const lo=Math.min(...los), hi=Math.max(...his), rng=(hi-lo)||1;
      const X=i=>padL+i*((w-padL-padR)/Math.max(1,rows.length-1));
      const Y=v=>h-padB-((v-lo)/rng)*(h-padT-padB);
      svg.setAttribute('viewBox',`0 0 ${w} ${h}`); svg.setAttribute('preserveAspectRatio','none');
      let out='';
      // gridlines at round Elo steps
      const step=rng>600?200:100;
      for(let g=Math.ceil(lo/step)*step; g<=hi; g+=step)
        out+=`<line x1="${padL}" y1="${Y(g)}" x2="${w-padR}" y2="${Y(g)}" stroke="#21262d"/>`
            +`<text x="4" y="${Y(g)+4}" fill="#7d8590" font-size="11">${g}</text>`;
      // connect the search rungs so the progression reads as a line
      const sPts=rows.map((x,i)=>({x,i})).filter(p=>isSearch(p.x));
      out+=sPts.length>1?`<path d="${sPts.map((p,k)=>`${k?'L':'M'}${X(p.i).toFixed(1)} ${Y(p.x.elo).toFixed(1)}`).join(' ')}" fill="none" stroke="#3fb950" stroke-width="1.6" opacity="0.5"/>`:'';
      rows.forEach((x,i)=>{
        const c=isSearch(x)?'#3fb950':'#e3b341';
        const l=x.elo_lo!=null?x.elo_lo:x.elo, u=x.elo_hi!=null?x.elo_hi:x.elo;
        out+=`<line x1="${X(i)}" y1="${Y(l)}" x2="${X(i)}" y2="${Y(u)}" stroke="${c}" stroke-width="1" opacity="0.55"/>`
            +`<circle cx="${X(i)}" cy="${Y(x.elo)}" r="3.4" fill="${c}"><title>${esc(x.agent)} — ${x.elo} (${l}..${u}), ${x.games} games</title></circle>`;
      });
      const last=rows[rows.length-1];
      out+=`<text x="${Math.min(X(rows.length-1)+6,w-64)}" y="${Y(last.elo)-6}" fill="#e6edf3" font-size="11">${last.elo}</text>`;
      svg.innerHTML=out;
      // ---- tiles ----
      // best rung: only rows with a REAL interval (elo_hi>elo_lo) qualify — degenerate/CI-less
      // legacy rows (e.g. "1129 (1129..1129)") are noise, not a high-water mark
      const cand=rows.filter(x=>x.elo_lo!=null&&x.elo_hi!=null&&x.elo_hi>x.elo_lo);
      const pool=cand.length?cand:rows;
      let best=pool[0]; for(const x of pool) if(x.elo>best.elo) best=x;
      document.getElementById('lad-latest').textContent=`${last.elo} (${last.elo_lo}..${last.elo_hi})`;
      document.getElementById('lad-best').textContent=`${best.elo} (${best.elo_lo}..${best.elo_hi})`;
      document.getElementById('lad-n').textContent=rows.length;
      // ---- table: most recent first ----
      const rec=rows.slice(-10).reverse();
      document.getElementById('lad-table').innerHTML=
        '<tr><th>agent</th><th>games</th><th>vs SF@1320 (W-D-L)</th><th>score</th><th>Elo (95% CI)</th></tr>'
        +rec.map(x=>`<tr${x===best?' class="best"':''}><td>${esc(x.agent)}</td><td>${x.games}</td>`
          +`<td>${x.vs_sf_W!=null?`${x.vs_sf_W}-${x.vs_sf_D}-${x.vs_sf_L}`:'—'}</td>`
          +`<td>${x.vs_sf_score!=null?(+x.vs_sf_score).toFixed(3):'—'}</td>`
          +`<td>${x.elo} (${x.elo_lo}..${x.elo_hi})</td></tr>`).join('');
    }
  }catch(e){}
  setTimeout(pollLadder,10000);
}
pollLadder();
</script>
</body></html>"""
