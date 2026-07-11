"""Merge 11 :Population: — pathfinding over weight space (spec/pathfind-population.spec.md).

RRT*/PBT hybrid on top of the UNCHANGED canon trivium recipe: N lanes train in parallel from
the optimistic zero seed; every generation the population is ranked by confirmed-crown
strength, the bottom lanes are pruned and re-seeded from the top lane's weights with fresh
exploration jitter (:Slope-guard: protects the best-improving lane from greedy pruning —
the FIRE-PBT caveat). Every node/prune/fork is logged to data/pathfind_tree.md; lanes feed
the console ladder via qlearn's :Crown-rung:.

Preconditions: models/qlearn_zero_seed.pt (build_zero_seed.py), models/zca.npz, rsearch3,
Stockfish available (confirmed crowns). One population at a time; don't run alongside a
console training run (thread + SF contention).

Usage: python pathfind.py [gens=6] [epoch_games=1000] [epochs_per_gen=2] [lanes=4]
Smoke:  python pathfind.py 1 200 1 2
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

GENS = int(sys.argv[1]) if len(sys.argv) > 1 else 6          # safety CAP, not the target
EPOCH_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
EPOCHS_PER_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NLANES = int(sys.argv[4]) if len(sys.argv) > 4 else 4
RESUME_POP = int(sys.argv[5]) if len(sys.argv) > 5 else 0    # 1 = continue existing lanes
# (weights carry over; configs re-evolve from canon/explorer/jitter — not persisted)
GOAL_BAR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0  # stop when pop best crown bar
# reaches this (bar 26.4 ~ 1100 on the d2-greedy Elo scale); 0 = run to the GENS cap
LIN = sys.argv[7] if len(sys.argv) > 7 else "zero"           # lineage prefix: zero | mat —
# separate checkpoint/metrics/tag namespaces so campaigns never overwrite each other

SEED_FILE = f"models/qlearn_{LIN}_seed.pt"
TREE = "data/pathfind_tree.md"
LANE_IDS = "abcdef"[:NLANES]
THREADS = max(1, 12 // NLANES)

# canonical proven config (spec/trivium.spec.md) — lane 'a' runs exactly this, others jitter.
# Exploration = :Softmax-tau: (optimism-directed; pairs with the optimistic zero seed); ε=0.
CANON = {"alpha": 3e-4, "tau_start": 0.7, "tau_floor": 0.05, "c_start": 0.374, "c_end": 0.143,
         "b_srch": 0.341, "warmup": 0.481}


def jitter(gen, lane, reheat=1.0):
    """Fresh exploration knobs for a jittered lane (deterministic per (gen, lane)).
    reheat > 1 scales the τ anneal UP (operator's stall rule: local optimum -> raise
    exploration temperature; cools back to 1 on population progress)."""
    r = np.random.default_rng(1000 * gen + ord(lane))
    c_start = float(np.clip(CANON["c_start"] + r.uniform(-0.1, 0.1), 0.05, 0.6))
    return {"alpha": CANON["alpha"] * 2 ** r.uniform(-1, 1),
            "tau_start": float(min(CANON["tau_start"] * 2 ** r.uniform(-1, 1) * reheat, 3.0)),
            "tau_floor": float(min(CANON["tau_floor"] * 2 ** r.uniform(-1, 1) * reheat, 0.5)),
            "c_start": c_start, "c_end": CANON["c_end"], "b_srch": CANON["b_srch"],
            "warmup": float(np.clip(CANON["warmup"] + r.uniform(-0.15, 0.15), 0.1, 0.8))}


def explorer(reheat=1.0):
    """Dedicated high-τ scout (operator: 'leave one for exploration'): canon knobs with τ
    pinned hot — never anneals into an exploiter; when forked from the top node it keeps
    exploring from the best-known weights (RRT expansion from the best node)."""
    return dict(CANON, tau_start=min(1.4 * reheat, 3.0), tau_floor=min(0.3 * reheat, 0.5))


def triv(cfg):
    a_s = 1.0 - cfg["b_srch"] - cfg["c_start"]
    a_e = 1.0 - cfg["b_srch"] - cfg["c_end"]
    return (f"{a_s:.3f},{cfg['b_srch']:.3f},{cfg['c_start']:.3f}",
            f"{a_e:.3f},{cfg['b_srch']:.3f},{cfg['c_end']:.3f}")


def lane_env(lane, cfg):
    t_start, t_end = triv(cfg)
    return dict(os.environ,
                QLEARN_ALPHA=f"{cfg['alpha']:.6f}", QLEARN_GAMMA="1.0", QLEARN_LAMBDA="0.7",
                QLEARN_WARMUP="0.5", QLEARN_PATIENCE="99",     # population prunes, not patience
                QLEARN_ELO_GAMES="0", QLEARN_EPOCH_ELO_GAMES="24", QLEARN_LOG_EVERY="500",
                QLEARN_BATCH_GAMES="200", QLEARN_FREEZE_EPOCH="1", QLEARN_RESUME="1",
                QLEARN_ANCHOR="1", QLEARN_TDLEAF="1", QLEARN_OPP="self", QLEARN_ENC="kc",
                QLEARN_CONFIRM="1", QLEARN_RAMP="1", QLEARN_KC_FAITHFUL="1",
                QLEARN_RSEARCH_DEPTH="2", QLEARN_RSEARCH_MOD="rsearch4",
                QLEARN_ZCA="models/zca.npz",
                QLEARN_TRIVIUM=t_start, QLEARN_TRIVIUM_END=t_end,
                QLEARN_TRIVIUM_WARMUP=f"{cfg['warmup']:.4f}",
                QLEARN_PARGEN="1", QLEARN_PARGEN_EPS="0.0",
                QLEARN_PARGEN_SOFTMAX="1", QLEARN_STALE_REHEAT="0.5",
                QLEARN_TAU_START=f"{cfg['tau_start']:.4f}",
                QLEARN_TAU_FLOOR=f"{cfg['tau_floor']:.4f}",
                QLEARN_PARGEN_THREADS=str(THREADS), QLEARN_PROXY_GAMES="4",
                QLEARN_DEV="cpu", QLEARN_TAG=f"{LIN}{lane.upper()}",
                QLEARN_CKPT=f"models/qlearn_{LIN}_{lane}.pt",
                # control lane feeds the console's LIVE panel directly; others get own files
                QLEARN_METRICS=("data/qlearn_metrics.jsonl" if lane == "a"
                                else f"data/qlearn_{LIN}_{lane}.jsonl"))


def bar(lane):
    """Confirmed-crown strength of a lane's best checkpoint (None before first crown)."""
    try:
        return torch.load(f"models/qlearn_{LIN}_{lane}_best.pt", map_location="cpu").get("strength")
    except (FileNotFoundError, RuntimeError):
        return None


def fork(dst, src):
    """Re-seed lane dst from lane src's best weights: new lineage — bar + optimizer stripped."""
    ck = torch.load(f"models/qlearn_{LIN}_{src}_best.pt", map_location="cpu")
    ck.pop("strength", None)
    ck.pop("opt_state", None)
    torch.save(ck, f"models/qlearn_{LIN}_{dst}.pt")
    torch.save(ck, f"models/qlearn_{LIN}_{dst}_best.pt")


def log_tree(line):
    with open(TREE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


# --- :Model-the-model: (operator, 2026-07-11) — the search gets its own value function ----
def knob_vec(cfg):
    """Config -> feature vector for the jitter surrogate (log-alpha, temps, mix, warmup, 1)."""
    return np.array([math.log10(cfg["alpha"]), cfg["tau_start"], cfg["tau_floor"],
                     cfg["c_start"], cfg["warmup"], 1.0])


def fit_surrogate(mX, my):
    """Ridge fit of delta-bar over knobs; None until >=6 (config, outcome) observations."""
    if len(my) < 6:
        return None
    X, yv = np.array(mX), np.array(my)
    return np.linalg.solve(X.T @ X + 0.1 * np.eye(X.shape[1]), X.T @ yv)


def guided_jitter(gen, lane, reheat, theta):
    """Sample 12 candidate jitters; with the surrogate fitted, take the predicted-best half
    the time (the other half stays random — the surrogate must never own exploration)."""
    cands = [jitter(gen * 97 + i * 7919, lane, reheat) for i in range(12)]
    if theta is None or np.random.default_rng(1009 * gen + ord(lane)).random() < 0.5:
        return cands[0]
    scores = [float(knob_vec(c) @ theta) for c in cands]
    return cands[int(np.argmax(scores))]


def main():
    os.makedirs("data", exist_ok=True)
    if not RESUME_POP:
        seed = torch.load(SEED_FILE, map_location="cpu")
        for lane in LANE_IDS:                          # every lane starts at the SAME zero node
            torch.save(seed, f"models/qlearn_{LIN}_{lane}.pt")
            torch.save(seed, f"models/qlearn_{LIN}_{lane}_best.pt")
    log_tree(f"\n## Population run {time.strftime('%Y-%m-%d %H:%M')} — "
             f"{NLANES} lanes x <= {GENS} gens x {EPOCHS_PER_GEN}x{EPOCH_GAMES} games "
             f"(threads/lane {THREADS}, resume={RESUME_POP}, goal bar {GOAL_BAR or '—'})")
    log_tree("| gen | lane | config | bar | action |")
    log_tree("|---|---|---|---|---|")

    prev = {lane: None for lane in LANE_IDS}
    configs = {lane: (dict(CANON) if lane == "a" else
                      explorer() if lane == LANE_IDS[-1] else
                      jitter(0, lane)) for lane in LANE_IDS}
    reheat, prev_pop_best = 1.0, None                  # :Reheat:: stall -> raise τ, progress -> cool
    hist = {lane: [] for lane in LANE_IDS}             # :Model-the-model: bar trajectories
    mX, my = [], []                                    # (knobs, delta-bar) surrogate data
    for gen in range(1, GENS + 1):
        procs = {}
        for lane in LANE_IDS:
            cfg = configs[lane]
            log = open(f"data/pathfind_{lane}.log", "a")
            procs[lane] = subprocess.Popen(
                [sys.executable, "qlearn.py", str(EPOCH_GAMES), str(EPOCHS_PER_GEN)],
                env=lane_env(lane, cfg), stdout=log, stderr=subprocess.STDOUT)
        for lane, p in procs.items():
            rc = p.wait()
            if rc != 0:
                log_tree(f"| {gen} | {lane} | — | — | LANE FAILED rc={rc} (see data/pathfind_{lane}.log) |")

        bars = {lane: bar(lane) for lane in LANE_IDS}
        slopes = {lane: (bars[lane] - prev[lane]) if (bars[lane] is not None and prev[lane] is not None)
                  else None for lane in LANE_IDS}
        for lane in LANE_IDS:                          # :Model-the-model: observe outcomes
            if bars[lane] is not None:
                hist[lane].append(bars[lane])
            if slopes[lane] is not None:
                mX.append(knob_vec(configs[lane]))
                my.append(slopes[lane])
        theta = fit_surrogate(mX, my)

        def predicted(lane):
            """FIRE-style rank: current bar + 3-gen trajectory extrapolation (falls back to
            the bar itself while a lane's post-fork history is too short to model)."""
            if bars[lane] is None:
                return -1e9
            h = hist[lane]
            if len(h) < 2:
                return bars[lane]
            return bars[lane] + 3.0 * float(np.mean(np.diff(h[-4:])))

        ranked = sorted(LANE_IDS, key=predicted, reverse=True)
        best_slope = max((l for l in LANE_IDS if slopes[l] is not None),
                         key=lambda l: slopes[l], default=None)
        top = ranked[0]
        prune = [l for l in ranked[-(NLANES // 2):] if l != best_slope and l != top]

        pop_best = max((b for b in bars.values() if b is not None), default=None)
        if GOAL_BAR > 0 and pop_best is not None and pop_best >= GOAL_BAR:
            log_tree(f"| {gen} | — | GOAL REACHED: pop best {pop_best:.2f} >= {GOAL_BAR} "
                     f"(~1100 d2-greedy) | — | — |")
            for lane in LANE_IDS:
                b = "—" if bars[lane] is None else f"{bars[lane]:.2f}"
                log_tree(f"| {gen} | {lane} | final | {b} | — |")
            break
        if pop_best is not None:
            if prev_pop_best is not None and pop_best <= prev_pop_best + 1e-3:
                reheat = min(reheat * 1.5, 4.0)        # stall: turn the temperature up
                log_tree(f"| {gen} | — | REHEAT x{reheat:.2f} (pop best flat at {pop_best:.2f}) | — | — |")
            else:
                reheat = 1.0                           # progress: cool back to canon scale
            prev_pop_best = pop_best

        for lane in LANE_IDS:
            cfg = configs[lane]
            desc = (f"a{cfg['alpha']:.4f} t{cfg['tau_start']:.2f}/{cfg['tau_floor']:.2f} "
                    f"c{cfg['c_start']:.2f} w{cfg['warmup']:.2f}")
            action = ("TOP" if lane == top else
                      "slope-guard" if lane == best_slope and lane in ranked[-(NLANES // 2):] else
                      f"PRUNED -> fork({top})" if lane in prune else "keep")
            b = "—" if bars[lane] is None else f"{bars[lane]:.2f}"
            p = predicted(lane)
            pd = "" if (bars[lane] is None or abs(p - bars[lane]) < 1e-6) else f" pred{p:.1f}"
            log_tree(f"| {gen} | {lane} | {desc} | {b}{pd} | {action} |")

        if gen < GENS:
            for lane in prune:
                fork(lane, top)
                hist[lane] = []                        # forked lane = new trajectory
                configs[lane] = (explorer(reheat) if lane == LANE_IDS[-1]
                                 else guided_jitter(gen, lane, reheat, theta))
            prev = dict(bars)

    log_tree(f"| — | — | final ranking: {' > '.join(ranked)} | {bars[ranked[0]]} | "
             f"winner models/qlearn_{LIN}_{ranked[0]}_best.pt |")
    print(f"\nWINNER lane {ranked[0]}: models/qlearn_{LIN}_{ranked[0]}_best.pt "
          f"(bar {bars[ranked[0]]}) — next: purist + d7 rungs")


if __name__ == "__main__":
    main()
