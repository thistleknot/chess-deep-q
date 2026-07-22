import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Robustness confirmation for :Search-teacher: — do ALL distilled iterates beat the champion at
native d9 (method robust), or only the selected peak A3 (lucky)? Direct head-to-head duels, the
deployment-truth ruler (instrument validated: champ-vs-champ d9 = 0.490, band includes 0.5)."""
import os
os.environ.setdefault("DISTILL_DUEL_DEPTH", "9")
from experiments.distill_iterate import duel

CH = "models/champion.pt"


def main():
    for cand in ("models/champion_distillA2.pt", "models/champion_distillA4.pt"):
        _, line = duel(cand, CH, 48, f"{os.path.basename(cand)[:-3]}-vs-champ")
        print(line, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
