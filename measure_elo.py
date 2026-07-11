"""Elo harness for the RL ladder (spec/observability.spec.md). Anchors the current agent to Stockfish.

Plays the current merge's agent vs two rungs — random (ordinal baseline) and SF@1320 (anchored) — computes
Elo = 1320 + elo_diff(score) for the anchored rung, and appends one row to data/rl_trend.jsonl that the
dashboard reads. At Merge 0 the agent is uniform-random, so the anchored score is ~0 and Elo floors below
the anchor: the honest bottom every learning rung climbs from.

The agent is passed as a move function so each future merge swaps in its learner without touching this file.

Usage: python measure_elo.py [games]
"""
import sys
import os
import json
import glob
import math
import time
import random

import chess
import chess.engine

from chess_rl import random_policy

TREND = "data/rl_trend.jsonl"
PLY_CAP = 120


def elo_diff(s, n=None):
    """Elo gap implied by score s (0..1). With n (the game count), clamps at the HALF-POINT bound s=1/(2n):
    a 0/n shutout only proves "at or below" a finite-sample floor (~-545 @ n=12, ~-758 @ n=40), keeping the
    result on the actual rating scale. Without n, falls back to the legacy 1e-4 clamp (floor -1600)."""
    eps = 1.0 / (2 * n) if n else 1e-4
    s = min(max(s, eps), 1 - eps)
    return 400 * math.log10(s / (1 - s))


def play(agent_move, opp_move, games, cap=PLY_CAP):
    """Agent vs opponent, alternating colors. Return (W, D, L, avg_plies) from the AGENT's perspective."""
    W = D = L = 0
    plies_total = 0
    for g in range(games):
        agent_white = (g % 2 == 0)
        b = chess.Board()
        plies = 0
        while not b.is_game_over() and plies < cap:
            mv = agent_move(b) if (b.turn == chess.WHITE) == agent_white else opp_move(b)
            b.push(mv)
            plies += 1
        plies_total += plies
        if b.is_checkmate():
            agent_won = (b.turn == chess.BLACK) == agent_white   # side to move is mated
            W += int(agent_won)
            L += int(not agent_won)
        else:
            D += 1    # stalemate / insufficient / repetition / ply cap
    return W, D, L, plies_total / games


def measure(agent_move, agent_name, games, merge=0):
    """Anchor ANY agent: play vs random + SF@1320, append a trend row, return (elo, vs_sf_score).

    Reusable by every rung — Merge 0 passes random_policy; the Q-learner passes its greedy move fn. This is
    the single :Elo-objective: entry point (spec/observability.spec.md): the returned elo is what Optuna
    maximizes and what the KILL-CHECK line reports.
    """
    rng = random.Random(0)
    print(f"measuring {agent_name}: {games} games/rung", flush=True)
    t = time.time()

    rW, rD, rL, r_len = play(agent_move, lambda b: random_policy(b, rng), games)
    s_rand = (rW + 0.5 * rD) / games
    print(f"  vs random          {rW}W-{rD}D-{rL}L  score {s_rand:.2f}  "
          f"({elo_diff(s_rand, games):+.0f} Elo vs opp, ordinal)")

    elo = None
    s_sf = None
    sW = sD = sL = 0
    sf_len = 0.0
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if sfp:
        sf = chess.engine.SimpleEngine.popen_uci(sfp[0])
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
        lim = chess.engine.Limit(time=0.05)
        sf_move = lambda b: sf.play(b, lim).move
        sW, sD, sL, sf_len = play(agent_move, sf_move, games)
        sf.quit()
        s_sf = (sW + 0.5 * sD) / games
        elo = 1320 + elo_diff(s_sf, games)
        # the score is a MEAN over games -> report its standard error as an Elo CI (variance, not just mean)
        se = math.sqrt(max(s_sf * (1 - s_sf), 1e-9) / games)
        elo_lo, elo_hi = 1320 + elo_diff(s_sf - 1.96 * se, games), 1320 + elo_diff(s_sf + 1.96 * se, games)
        note = "  <- floored (score ~0: far below the anchor)" if s_sf < 0.05 else ""
        print(f"  vs SF@1320         {sW}W-{sD}D-{sL}L  score {s_sf:.2f}  -> ~{elo:.0f} Elo "
              f"(95% CI {elo_lo:.0f}..{elo_hi:.0f}){note}")
    else:
        print("  (no stockfish found -> no anchored Elo)")

    try:                                   # peg every measurement to the code state that produced it
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    row = {
        "merge": merge,
        "agent": agent_name,
        "commit": commit,
        "ts": int(time.time()),
        "games": games,
        "vs_random_score": round(s_rand, 4),
        "vs_sf_W": sW, "vs_sf_D": sD, "vs_sf_L": sL,
        "vs_sf_score": None if s_sf is None else round(s_sf, 4),
        "elo": None if elo is None else round(elo),
        "elo_lo": None if s_sf is None else round(elo_lo),
        "elo_hi": None if s_sf is None else round(elo_hi),
        "avg_len": round((r_len + sf_len) / (2 if sfp else 1)),
    }
    os.makedirs(os.path.dirname(TREND) or ".", exist_ok=True)
    with open(TREND, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    try:                                   # MLflow mirror (local ./mlruns): every rung = one run,
        import mlflow                      # tagged with the commit — NEVER fatal to a measurement
        mlflow.set_tracking_uri("file:mlruns")
        mlflow.set_experiment("chess-rl-elo")
        with mlflow.start_run(run_name=agent_name):
            mlflow.set_tags({"git_commit": commit or "dirty/unknown", "merge": merge})
            mlflow.log_params({"agent": agent_name, "games": games})
            mlflow.log_metrics({k: float(v) for k, v in row.items()
                                if isinstance(v, (int, float)) and v is not None
                                and k not in ("ts", "merge")})
    except Exception as e:
        print(f"(mlflow logging skipped: {e})", flush=True)
    print(f"[wall {time.time()-t:.0f}s] appended to {TREND}")
    return row


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = random.Random(0)
    measure(lambda b: random_policy(b, rng), "random (Merge 0)", games, merge=0)
    print("-> run  python dashboard.py")


if __name__ == "__main__":
    main()
