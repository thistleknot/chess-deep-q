import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Console bridge for the RUNNING population (Merge 11 stopgap).

Purpose: the live pathfind lanes write per-lane metrics files the console never reads; this
mirrors the CONTROL lane's rows (data/qlearn_zero_a.jsonl) into the console's default file
(data/qlearn_metrics.jsonl) so the LIVE TRAINING panel tracks the population's canonical
lane in real time. Handles per-generation truncation (offset resets when the source shrinks).
Future runs don't need this — pathfind.py points lane 'a' at the default file directly.

Usage: python bridge_lane_metrics.py   (detached; stop via PID in data/bridge.pid)
"""
import os
import time

SRC = "data/qlearn_zero_a.jsonl"
DST = "data/qlearn_metrics.jsonl"

offset = 0
while True:
    try:
        size = os.path.getsize(SRC)
        if size < offset:
            offset = 0                                  # new generation truncated the source
        if size > offset:
            with open(SRC, "rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
            offset = size
            with open(DST, "ab") as out:
                out.write(chunk)
    except OSError:
        pass                                            # source not yet created this generation
    time.sleep(5)
