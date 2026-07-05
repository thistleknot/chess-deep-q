"""Autonomous NNUE distillation climb (spec teacher-distillation :Run-contract:, NNUE track).

Each cycle: grow the corpus (gen_labels, coverage-seeded) -> retrain (train_nnue) -> kill-check vs
pst@d2 (measure_nnue) -> append a trajectory point to models/nnue_trend.jsonl -> regenerate the chart.
Runs `cycles` cycles then stops. Observability is the growing trend log + chart, checkable any time.

Usage: python climb_nnue.py [cycles] [gen_seconds] [epochs] [kc_games]
"""
import json
import os
import re
import subprocess
import sys

TREND = "models/nnue_trend.jsonl"
CORPUS = "data/distill_v2.jsonl"
PST_D2 = 1367
PY = sys.executable
_ED = re.compile(r"elo_diff\s+([+-]?\d+)")
_SC = re.compile(r"score\s+([0-9.]+)")


def records():
    return sum(1 for _ in open(CORPUS)) if os.path.exists(CORPUS) else 0


def run(cmd, logpath):
    """Run a subprocess, tee stdout+stderr to logpath, return the captured text."""
    with open(logpath, "w") as fh:
        subprocess.run([PY] + cmd, stdout=fh, stderr=subprocess.STDOUT)
    with open(logpath) as fh:
        return fh.read()


def main():
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    gen_s = sys.argv[2] if len(sys.argv) > 2 else "1500"
    epochs = sys.argv[3] if len(sys.argv) > 3 else "45"
    kc = sys.argv[4] if len(sys.argv) > 4 else "30"
    for c in range(1, cycles + 1):
        print(f"[climb {c}/{cycles}] grow corpus ...", flush=True)
        run(["gen_labels.py", gen_s, "8", "10", CORPUS], f"climb_gen_{c}.log")
        recs = records()
        print(f"[climb {c}] train on {recs} records ...", flush=True)
        run(["train_nnue.py", epochs, "1024"], f"climb_train_{c}.log")
        print(f"[climb {c}] kill-check vs pst@d2 ...", flush=True)
        out = run(["measure_nnue.py", kc, "2"], f"climb_kc_{c}.log")
        m, s = _ED.search(out), _SC.search(out)
        if not m:
            print(f"[climb {c}] no kill-check result parsed; stopping", flush=True)
            break
        ed = int(m.group(1))
        sc = float(s.group(1)) if s else None
        pt = {"records": recs, "elo_diff": ed, "elo": PST_D2 + ed, "score": sc,
              "label": f"climb {c} (+coverage)", "cycle": c}
        with open(TREND, "a") as fh:
            fh.write(json.dumps(pt) + "\n")
        subprocess.run([PY, "plot_nnue_trend.py"])
        print(f"[climb {c}] {recs} records -> Elo {PST_D2 + ed} (diff {ed:+d}, score {sc})", flush=True)
    print("[climb] done", flush=True)


if __name__ == "__main__":
    main()
