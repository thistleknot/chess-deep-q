"""Ladder feed for runs started before :Crown-rung: landed in qlearn.py.

Tails data/train.log; every KEPT crown becomes a data/rl_trend.jsonl row on the d2-greedy
scale, reconstructed from the crown's pooled SF games (epoch window + 24 confirmation games).
Purpose: the console's SEARCH LADDER tracks the live run. Precondition: the run prints the
standard "epoch N strength" / "crown check" lines. Failure mode: if the log is replaced by a
NEW run, reconstruction keys (epoch, confirmed) reset with it — rows are deduped against the
trend file by agent label, so restarts never double-append. Rows are tagged "(live, est)":
score is recovered as (confirmed - epoch proxy)/100, which trusts the confirmation proxy to
match the epoch proxy (error ~ +/-10 Elo); exact rows come from the trainer itself on runs
started after commit 673a076.
"""
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chessdq.measure_elo import elo_diff

LOG, TREND = "data/train.log", "data/rl_trend.jsonl"
CONF_GAMES = 24                                    # EPOCH_ELO_GAMES of the watched run
EP = re.compile(r"epoch (\d+) strength: sf ([\d.]+)/(\d+) \+ proxy ([\d.]+)")
CR = re.compile(r"crown check: epoch [\d.]+ -> confirmed ([\d.]+) \(kept\)")


def existing_labels():
    labels = set()
    if os.path.exists(TREND):
        with open(TREND) as fh:
            for ln in fh:
                try:
                    labels.add(json.loads(ln).get("agent", ""))
                except json.JSONDecodeError:
                    pass
    return labels


def crown_rows(text, labels):
    last = None
    for ln in text.splitlines():
        m = EP.search(ln)
        if m:
            last = m
            continue
        c = CR.search(ln)
        if not (c and last):
            continue
        conf = float(c.group(1))
        proxy, n = float(last.group(4)), int(last.group(3)) + CONF_GAMES
        agent = f"triv crown {conf:.2f} d2-greedy (live, est)"
        if agent in labels:
            continue
        labels.add(agent)
        s = max(0.0, min(1.0, (conf - proxy) / 100.0))
        se = math.sqrt(max(s * (1 - s), 1e-9) / n)
        yield {"merge": 9, "agent": agent, "ts": int(time.time()), "games": n,
               "vs_sf_W": None, "vs_sf_D": None, "vs_sf_L": None,
               "vs_sf_score": round(s, 4),
               "elo": round(1320 + elo_diff(s, n)),
               "elo_lo": round(1320 + elo_diff(s - 1.96 * se, n)),
               "elo_hi": round(1320 + elo_diff(s + 1.96 * se, n))}


def main():
    labels = existing_labels()
    while True:
        try:
            with open(LOG, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            text = ""
        new = list(crown_rows(text, labels))
        if new:
            with open(TREND, "a") as fh:
                for row in new:
                    fh.write(json.dumps(row) + "\n")
            print(f"appended {len(new)} crown row(s)", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
