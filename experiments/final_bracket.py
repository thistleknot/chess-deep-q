import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Final selection bracket for :Search-teacher: — pick the champion candidate.
SWA (weight-avg of A2/A3/A4) vs champion, and SWA vs A3 (best single iterate). Native d9, the
deployment-truth ruler (instrument validated: champ-vs-champ d9 = 0.490)."""
import os
os.environ.setdefault("DISTILL_DUEL_DEPTH", "9")
from experiments.distill_iterate import duel

PAIRS = [
    ("models/champion_distillSWA.pt", "models/champion.pt",          "SWA-vs-champ"),
    ("models/champion_distillSWA.pt", "models/champion_distillA3.pt", "SWA-vs-A3"),
]


def main():
    for a, b, tag in PAIRS:
        _, line = duel(a, b, 48, tag)
        print(line, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
