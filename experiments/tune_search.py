"""Optuna screen for the search-arm campaign (spec/search-arms.spec.md :Screen-instrument:).

One study per arm, all over the FROZEN champion evaluator (amap-897): the objective is
the H2H Elo diff (A = arm mover, B = the standard d2-width-8 beam duel lens over the
same champion weights) at the explore tier, parsed from head2head.py's verdict line.
Optuna is the cheap surrogate (rank-ordering configs); the real numbers come from the
50g round-robin and, on a passed gate, the pooled ladder.

Usage:  python experiments/tune_search.py <arm> [timeout_s] [games] [n_trials]
  arm       hyb | puc | ucv | baseline
  timeout_s wall budget for study.optimize (default 1800 — the 30-min study cap)
  games     duel games per trial (default 16; explore-tier law caps at 20)
  n_trials  generous ceiling; the timeout is the binding budget (default 24)

sims is an infrastructure CONTROL, not a search dimension (:Budget-match:): calibrated
once per (arm, leaf) so the arm's move pace matches the reference mover's within ~25%,
cached in data/search_sims_budget.json.

Storage: sqlite:///models/search_optuna.db, study sa_<arm>_<sha8(space|games|ref)>.
Trial 0 = literature prior (c_puct 1.5 Grill et al. 2020; UCB-V c 1.0 Audibert et al.
2009; lb 0.5 midpoint; baseline = the incumbent's own defaults).
Failure modes: duel subprocess timeout/parse-miss -> trial pruned, not a study abort.
"""
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT = "models/champion.pt"
REF_SPEC = "enc:amap:models/champion.pt"      # the standard d2w8 τ0.02 duel lens
DB = "sqlite:///models/search_optuna.db"
BUDGET_CACHE = "data/search_sims_budget.json"
CAL_SIMS = 64                                  # probe sims for pace measurement
CAL_FENS = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "2rq1rk1/pp1bppbp/3p1np1/8/3NP3/1BN1B3/PPPQ1PPP/2KR3R w - - 0 1",
    "r4rk1/pp1n1ppp/2pbpn2/3p4/2PP4/1PN1PN2/PB3PPP/R2Q1RK1 b - - 0 11",
    "r3kb1r/ppp1qppp/2n1bn2/4p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 7",
    "8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 1",
]

PRIOR = {
    "hyb": dict(lb=0.5, c=1.5, leaf="ab2", tp=0.2),
    "puc": dict(c=1.5, tp=0.2),
    "ucv": dict(ucbc=1.0, tp=0.2),
    "baseline": dict(tau=0.02, width=8, depth=2),
}


def _median_move_time(spec, fens):
    """Median seconds/move of a mover spec over probe FENs (in-process, 1 thread)."""
    import random
    import chess
    import torch
    torch.set_num_threads(1)
    from chessdq.head2head import mover_from_spec
    mv = mover_from_spec(spec, random.Random(7))
    ts = []
    for fen in fens:
        b = chess.Board(fen)
        t0 = time.perf_counter()
        mv(b)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _arm_spec(arm, p, sims):
    if arm == "baseline":
        return f"beam:{p['tau']:.4f}:{p['width']}:{p['depth']}:{REF_SPEC}"
    if arm == "hyb":
        return (f"sa:sel=puct,lb={p['lb']:.4f},leaf={p['leaf']},sims={sims},"
                f"c={p['c']:.4f},tp={p['tp']:.4f}:{CKPT}")
    if arm == "puc":
        return f"sa:sel=puct,lb=0.0,leaf=vf,sims={sims},c={p['c']:.4f},tp={p['tp']:.4f}:{CKPT}"
    if arm == "ucv":
        return f"sa:sel=ucbv,leaf=vf,sims={sims},ucbc={p['ucbc']:.4f},tp={p['tp']:.4f}:{CKPT}"
    raise ValueError(arm)


def calibrated_sims(arm, leaf="vf"):
    """sims for this (arm, leaf) matching the reference mover's pace (:Budget-match:)."""
    if arm == "baseline":
        return 0
    key = f"{arm}:{leaf}"
    cache = {}
    if os.path.exists(BUDGET_CACHE):
        with open(BUDGET_CACHE, encoding="utf-8") as fh:
            cache = json.load(fh)
    if key in cache:
        return cache[key]
    t_ref = _median_move_time(REF_SPEC, CAL_FENS)
    probe = _arm_spec(arm, {**PRIOR[arm], "leaf": leaf}, CAL_SIMS)
    t_arm = _median_move_time(probe, CAL_FENS)
    sims = max(16, min(512, int(CAL_SIMS * t_ref / max(t_arm, 1e-6))))
    cache[key] = sims
    os.makedirs("data", exist_ok=True)
    with open(BUDGET_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1)
    print(f"[calibrate] {key}: ref {t_ref:.3f}s/mv, probe@{CAL_SIMS} {t_arm:.3f}s/mv "
          f"-> sims={sims}", flush=True)
    return sims


def run_duel(spec_a, games, tag):
    """One H2H duel vs the reference; returns final Elo diff or None on failure."""
    env = dict(os.environ, H2H_CAP=str(games),
               H2H_SHARDS=os.environ.get("H2H_SHARDS", "2"))
    cmd = [sys.executable, "head2head.py", spec_a, REF_SPEC, str(games), tag]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             timeout=90 * games + 300).stdout
    except subprocess.TimeoutExpired:
        print(f"[trial {tag}] duel TIMEOUT", flush=True)
        return None
    m = re.findall(r"-> Elo ([+-]?\d+)", out or "")
    if not m:
        print(f"[trial {tag}] no verdict line; tail: {(out or '')[-300:]}", flush=True)
        return None
    return float(m[-1])


def make_objective(arm, games):
    def objective(trial):
        if arm == "hyb":
            p = dict(lb=trial.suggest_float("lb", 0.0, 1.0),
                     c=trial.suggest_float("c", 0.8, 3.0),
                     leaf=trial.suggest_categorical("leaf", ["ab1", "ab2", "ab3"]),
                     tp=trial.suggest_float("tp", 0.1, 0.5))
        elif arm == "puc":
            p = dict(c=trial.suggest_float("c", 0.8, 3.0),
                     tp=trial.suggest_float("tp", 0.1, 0.5))
        elif arm == "ucv":
            p = dict(ucbc=trial.suggest_float("ucbc", 0.5, 2.0),
                     tp=trial.suggest_float("tp", 0.1, 0.5))
        else:
            p = dict(tau=trial.suggest_float("tau", 0.02, 0.25),
                     width=trial.suggest_categorical("width", [4, 8, 12]),
                     depth=trial.suggest_categorical("depth", [2, 3]))
        sims = calibrated_sims(arm, p.get("leaf", "vf"))
        spec = _arm_spec(arm, p, sims)
        elo = run_duel(spec, games, f"sa_{arm}_t{trial.number}")
        if elo is None:
            import optuna
            raise optuna.TrialPruned()
        trial.set_user_attr("spec", spec)
        print(f"[trial sa_{arm}_t{trial.number}] {spec} -> {elo:+.0f}", flush=True)
        return elo
    return objective


def main():
    import optuna
    arm = sys.argv[1]
    assert arm in PRIOR, f"arm must be one of {sorted(PRIOR)}"
    timeout_s = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    space_id = hashlib.sha1(f"{arm}|{games}|{REF_SPEC}|v1".encode()).hexdigest()[:8]
    name = f"sa_{arm}_{space_id}"
    study = optuna.create_study(direction="maximize", storage=DB, study_name=name,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))
    if not study.trials:
        study.enqueue_trial(PRIOR[arm])
    print(f"[study {name}] arm={arm} games/trial={games} timeout={timeout_s}s "
          f"prior={PRIOR[arm]}", flush=True)
    study.optimize(make_objective(arm, games), n_trials=n_trials, timeout=timeout_s)
    done = [t for t in study.trials if t.value is not None]
    best = study.best_trial
    print(f"[study {name}] DONE {len(done)} scored trials | best {best.value:+.0f} "
          f"params {best.params} spec {best.user_attrs.get('spec')}", flush=True)


if __name__ == "__main__":
    main()
