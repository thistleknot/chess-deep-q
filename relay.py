"""Relay driver (spec/relay.spec.md) — ordered waves over one qlearn.py lane.

Each LEG resumes from the BATON (furthest confirmed position) with at most ONE diagnosed
change; every leg boundary runs a PAIR (tweak T vs plain-resume control C, parallel) so
the tweak has a counterfactual. Baton bar is monotone: losing pairs never demote it.

Usage:
  python relay.py                 # live: run pairs until RELAY_BUDGET epochs consumed
  RELAY_DRY=1 python relay.py     # dry-run: diagnose the current p7 history, print the
                                  # order that WOULD be issued, launch nothing

Failure modes: a leg subprocess that dies (hard-exit path in qlearn.py) scores as bar 0
for that arm — the pair still resolves; the driver never blocks on a corpse.
"""
import json
import os
import subprocess
import sys
import shutil
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

BATON_CKPT = os.environ.get("RELAY_BATON", "models/qlearn_p7_best.pt")
LEG_EPOCHS = int(os.environ.get("RELAY_LEG_EPOCHS", "3"))   # sized to <15 min/leg (policy)
EPOCH_GAMES = int(os.environ.get("RELAY_EPOCH_GAMES", "200"))
BUDGET = int(os.environ.get("RELAY_BUDGET", "30"))          # total epochs across all legs
BAND = 3.0            # strength-scale noise band (spec :Relay: — declared, ~binomial sd)
LOG = os.path.join(ROOT, "data", "relay.md")
STOP = os.path.join(ROOT, "data", "relay.stop")

# The winning-protocol env (console parity — round-2 winner: pst-769, organ-free, proven
# parms). ONE dict, per the spec Require clause. τ_start/α are the only knobs the rule
# table may scale.
BASE = dict(
    QLEARN_ENC="pst", QLEARN_OPP="graded", QLEARN_TDLEAF="1", QLEARN_KC_FAITHFUL="1",
    QLEARN_RAMP="1", QLEARN_RSEARCH_DEPTH="0", QLEARN_PARGEN="0", QLEARN_CONFIRM="1",
    QLEARN_ARCH="linear", QLEARN_ALPHA="0.0003", QLEARN_GAMMA="0.99", QLEARN_LAMBDA="0.8",
    QLEARN_WARMUP="0.4", QLEARN_LAMBDA_WARMUP="0.8", QLEARN_TAU_FLOOR="0.05",
    QLEARN_TAU_START="0.7", QLEARN_TRIVIUM="0.285,0.341,0.374",
    QLEARN_TRIVIUM_END="0.516,0.341,0.143", QLEARN_TRIVIUM_WARMUP="0.481",
    QLEARN_BATCH_GAMES="20", QLEARN_EPOCH_ELO_GAMES="24", QLEARN_PROXY_GAMES="4",
    QLEARN_ELO_GAMES="0",          # legs never run the long final measure; bars only
    QLEARN_PATIENCE="99",          # the DRIVER owns stopping (leg boundary = patience)
    QLEARN_RESUME="1", QLEARN_FREEZE_EPOCH="1", QLEARN_ADAPTIVE_LAMBDA="1",
)


def bar_of(ckpt):
    """Confirmed bar stored in a checkpoint; 0.0 for missing/corrupt (dead-arm score)."""
    try:
        return float(torch.load(ckpt, map_location="cpu").get("strength") or 0.0)
    except Exception:
        return 0.0


def log_event(lines):
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def epoch_series(metrics_path):
    """Non-null epoch_strength values from a leg's metrics stream."""
    out = []
    if os.path.exists(metrics_path):
        for ln in open(metrics_path, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("epoch_strength") is not None:
                out.append(float(r["epoch_strength"]))
    return out


def diagnose(series, leg_bar, baton_bar):
    """Rule-table signature (spec :Relay: — pre-registered; default order = no change)."""
    if leg_bar > baton_bar:
        return "ADVANCE"
    if series and max(series) < baton_bar - BAND:
        return "REGRESS"
    diffs = [s - baton_bar for s in series]
    flips = sum(1 for a, b in zip(diffs, diffs[1:]) if a * b < 0)
    if flips >= 2 and series and (max(series) - min(series)) > BAND:
        return "OSCILL"
    return "STALL"


def order_for(sig, knobs):
    """Apply ONE knob change per the rule table; returns (new_knobs, description)."""
    tau0, a0 = 0.7, 0.0003                                  # base values (caps/floors)
    k = dict(knobs)
    if sig == "ADVANCE":
        k["tau"], k["alpha"] = tau0, a0
        return k, "keep orders (knobs reset to base)"
    if sig == "REGRESS":
        k["tau"] = max(tau0 * 0.5, k["tau"] * 0.75)
        return k, f"discard leg; cool tau -> {k['tau']:.3f}"
    if sig == "OSCILL":
        k["alpha"] = max(a0 * 0.25, k["alpha"] * 0.5)
        return k, f"alpha -> {k['alpha']:.6f}"
    k["tau"] = min(tau0 * 4.0, k["tau"] * 1.5)
    return k, f"reheat tau -> {k['tau']:.3f}"


def launch_leg(arm, knobs):
    """Copy baton -> arm ckpt, wipe the arm's metrics stream, start the leg subprocess."""
    ck = f"models/relay_{arm}.pt"
    shutil.copyfile(BATON_CKPT, ck)
    shutil.copyfile(BATON_CKPT, ck.replace(".pt", "_best.pt"))
    metrics = f"data/relay_{arm}.jsonl"
    if os.path.exists(metrics):
        os.remove(metrics)
    env = {**os.environ, **BASE,                     # overrides win over BASE (knob scaling)
           "QLEARN_TAU_START": f"{knobs['tau']:.4f}", "QLEARN_ALPHA": f"{knobs['alpha']:.6f}",
           "QLEARN_CKPT": ck, "QLEARN_METRICS": metrics, "QLEARN_TAG": f"relay_{arm}",
           "QLEARN_SEED": str(int(time.time()) % 9973)}
    out = open(f"data/relay_{arm}.log", "w")
    return subprocess.Popen([PY, "qlearn.py", str(EPOCH_GAMES), str(LEG_EPOCHS)],
                            env=env, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)


def main():
    baton_bar = bar_of(BATON_CKPT)
    if os.environ.get("RELAY_DRY") == "1":
        # Diagnose the p7 monolithic run's tail (last 8 posted epochs of the console
        # stream — the run that motivated the relay) and print the order. Launch nothing.
        series = epoch_series("data/qlearn_metrics.jsonl")[-8:]
        sig = diagnose(series, leg_bar=baton_bar, baton_bar=baton_bar + 1e-9)
        knobs, desc = order_for(sig, {"tau": 0.7, "alpha": 0.0003})
        print(f"DRY | baton {BATON_CKPT} bar {baton_bar:.2f}")
        print(f"DRY | series {[round(s, 2) for s in series]}")
        print(f"DRY | signature {sig} -> order: {desc}")
        return

    # Leg 0 = the monolithic run's tail: diagnose it so leg 1 already carries an order
    # (otherwise leg 1 T would be a pure seed-replicate of C).
    tail = epoch_series("data/qlearn_metrics.jsonl")[-8:]
    sig0 = diagnose(tail, leg_bar=baton_bar, baton_bar=baton_bar + 1e-9)
    knobs, desc0 = order_for(sig0, {"tau": 0.7, "alpha": 0.0003})
    spent, leg_no = 0, 0
    log_event([f"## relay start {time.strftime('%Y-%m-%d %H:%M')}",
               f"1. baton {BATON_CKPT} bar {baton_bar:.2f} | leg={LEG_EPOCHS}ep budget={BUDGET}ep",
               f"2. leg-0 (monolith tail) signature {sig0} -> leg-1 order: {desc0}"])
    while spent < BUDGET and not os.path.exists(STOP):
        leg_no += 1
        pT, pC = launch_leg("T", knobs), launch_leg("C", {"tau": 0.7, "alpha": 0.0003})
        pT.wait(timeout=3600); pC.wait(timeout=3600)
        spent += 2 * LEG_EPOCHS
        tb, cb = bar_of("models/relay_T_best.pt"), bar_of("models/relay_C_best.pt")
        winner, wbar = ("T", tb) if tb > cb else ("C", cb)
        refuted = cb >= tb and knobs != {"tau": 0.7, "alpha": 0.0003}
        advanced = wbar > baton_bar
        if advanced:                       # Guarantee: baton monotone — winners only
            shutil.copyfile(f"models/relay_{winner}_best.pt", BATON_CKPT)
            baton_bar = wbar
        series = epoch_series(f"data/relay_{winner}.jsonl")
        sig = diagnose(series, wbar, baton_bar if not advanced else baton_bar - 1e-9)
        knobs, desc = order_for(sig, knobs)
        log_event([f"### leg {leg_no}",
                   f"1. pair done: T bar {tb:.2f} vs C bar {cb:.2f} -> winner {winner}",
                   f"2. tweak {'REFUTED (control >= tweak)' if refuted else 'held'}",
                   f"3. signature {sig} -> next order: {desc}",
                   f"scoreboard: baton {baton_bar:.2f} | epochs {spent}/{BUDGET} | leg {leg_no}"])
    log_event([f"## relay end: baton bar {baton_bar:.2f}, {spent} epochs consumed"])


if __name__ == "__main__":
    main()
