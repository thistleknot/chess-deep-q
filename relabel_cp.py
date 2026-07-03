"""Re-label the existing dataset's FENs with RAW Stockfish centipawns (not the squashed tanh).

The tanh(cp/400) target compresses material (a queen read +377..+640cp instead of ~+900), which
alpha-beta exploited. Raw cp lets the GBDT value material linearly. Multiprocess: one SF per worker.

Writes data/distill_cp.jsonl as [fen, cp] (White-absolute centipawns, mate as +/-100000).
"""
import json, glob, time
import multiprocessing as mp
import chess, chess.engine

SRC = "data/distill_sf.jsonl"
OUT = "data/distill_cp.jsonl"
DEPTH = 8

_eng = None


def _init():
    global _eng
    _eng = chess.engine.SimpleEngine.popen_uci(glob.glob('engines/**/stockfish*.exe', recursive=True)[0])
    _eng.configure({"Threads": 1, "Hash": 16})


def _label(fen):
    try:
        info = _eng.analyse(chess.Board(fen), chess.engine.Limit(depth=DEPTH))
        cp = info["score"].white().score(mate_score=100000)
        return (fen, cp) if cp is not None else None
    except Exception:
        return None


def main():
    fens = []
    with open(SRC) as fh:
        for line in fh:
            try:
                fens.append(json.loads(line)[0])
            except Exception:
                continue
    print(f"re-labelling {len(fens)} FENs with raw cp @ SF depth-{DEPTH}...")
    t = time.time()
    workers = max(1, mp.cpu_count() - 2)
    with mp.Pool(workers, initializer=_init) as pool:
        results = [r for r in pool.map(_label, fens, chunksize=200) if r]
    with open(OUT, "w") as f:
        for fen, cp in results:
            f.write(json.dumps([fen, cp]) + "\n")
    print(f"wrote {len(results)} rows to {OUT} in {time.time()-t:.0f}s ({workers} workers)")


if __name__ == "__main__":
    main()
