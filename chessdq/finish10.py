"""One-shot finisher: play the last 10 games of the amap1600 d9 rung on the pooled
ladder and merge with the preserved serial 49W-1D-0L/50 (operator call 2026-07-15:
'preserve the runs we have and just issue the remaining 10'). Windows spawn needs a
real file for the pool initializer — hence this script rather than a heredoc."""
import json
import math
import os
import time

os.environ.setdefault("CHESS_THERMAL_OFF", "1")   # workers pin their own single cores

from chessdq import claims_rung as cr
from chessdq.measure_elo import elo_diff, TREND


def main():
    t0 = time.time()
    W, D, L, avg = cr._pooled_rung("models/qlearn_amap1600_best.pt", 9, 10, "sf",
                                   6, [0, 1, 2, 3, 4, 5])
    W += 49
    D += 1
    games = 60
    s = (W + 0.5 * D) / games
    elo = 1320 + elo_diff(s, games)
    se = math.sqrt(max(s * (1 - s), 1e-9) / games)
    lo = 1320 + elo_diff(s - 1.96 * se, games)
    hi = 1320 + elo_diff(s + 1.96 * se, games)
    print(f"FINAL amap1600 d9: {W}W-{D}D-{L}L over {games}g score {s:.3f} "
          f"-> {elo:.0f} (95% {lo:.0f}..{hi:.0f})", flush=True)
    row = {"merge": 20, "agent": "amap1600 d9 rung (50 serial + 10 pooled)",
           "commit": None, "ts": int(time.time()), "games": games,
           "vs_random_score": None, "vs_sf_W": W, "vs_sf_D": D, "vs_sf_L": L,
           "vs_sf_score": round(s, 4), "elo": round(elo), "elo_lo": round(lo),
           "elo_hi": round(hi), "avg_len": round(avg)}
    with open(TREND, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[wall {time.time() - t0:.0f}s] appended", flush=True)


if __name__ == "__main__":
    main()
