import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""One-off deployment-faithful native duel (real file so Windows spawn-multiprocessing works).

Usage: python duel_native.py <a_ckpt> <b_ckpt> [games=30] [depth=6] [tag=duel]
Env DISTILL_DUEL_DEPTH overrides depth. Prints the native-d<depth> verdict line.
"""
import os
import sys

os.environ.setdefault("DISTILL_DUEL_DEPTH", sys.argv[4] if len(sys.argv) > 4 else "6")
from experiments.distill_iterate import duel


def main():
    a = sys.argv[1]; b = sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    tag = sys.argv[5] if len(sys.argv) > 5 else "duel"
    threads = int(os.environ.get("DUEL_THREADS", "2"))     # respect the operator's ≤2-core cap
    _, line = duel(a, b, games, tag, threads)
    print(line, flush=True)


if __name__ == "__main__":
    main()
