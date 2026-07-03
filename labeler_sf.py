"""Stockfish labelling worker: one OS process, one engine, appends labels to its own shard.

Spec: teacher-distillation :Process-separated-labeling: — labellers never share the trainer's
Python interpreter. Each label row is [fen, value, best_uci]: value = tanh(white_cp/400)
(White-absolute, spec :Value-target-convention:), best_uci = the PV best move (free from the same
analyse call), the one-hot policy target.

Usage: python labeler_sf.py <shard_path> <seconds> <depth> [seed]
"""

import sys
import json
import math
import time
import glob
import random

import chess
import chess.engine

TARGET_CP_SCALE = 400.0


def gen_position(rng, max_plies=44, capture_bias=0.5):
    b = chess.Board()
    for _ in range(rng.randint(4, max_plies)):
        if b.is_game_over():
            break
        legal = list(b.legal_moves)
        caps = [m for m in legal if b.is_capture(m)]
        b.push(rng.choice(caps) if (caps and rng.random() < capture_bias) else rng.choice(legal))
    return b


def main():
    shard, seconds, depth = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rng = random.Random(seed or None)

    sf_path = glob.glob("engines/**/stockfish*.exe", recursive=True)[0]
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 1, "Hash": 16})
    limit = chess.engine.Limit(depth=depth)

    n = 0
    start = time.time()
    with open(shard, "a") as fh:
        while time.time() - start < seconds:
            b = gen_position(rng)
            if b.is_game_over():
                continue
            try:
                info = eng.analyse(b, limit)
            except Exception:
                continue
            cp = info["score"].white().score(mate_score=100000)
            if cp is None:
                continue
            value = math.tanh(cp / TARGET_CP_SCALE)
            pv = info.get("pv")
            best = pv[0].uci() if pv else None
            fh.write(json.dumps([b.fen(), value, best]) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
    eng.quit()
    print(f"{shard}: {n} labels in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
