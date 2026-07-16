import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Generate :Search-visited-positions: (the H3 fix) and append raw-cp labels to the dataset.

The GBDT was trained on RANDOM playouts but alpha-beta plays REAL games and evaluates NON-QUIET
(post-capture) leaves — a distribution mismatch the strong search exploits. This generates positions
from real engine games plus 1-ply capture perturbations (a cheap proxy for search-visited leaves),
labels them with Stockfish depth-8 raw centipawns, and appends to data/distill_cp.jsonl.

Usage: python gen_searchvisited.py [games] [engine_depth]
"""
import sys, json, glob, time, random
import multiprocessing as mp
import chess, chess.engine

from chessdq.engine import AlphaBetaEngine

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


def gen_positions(games, depth):
    """Real engine-vs-engine games (random opening + 20% random move for diversity), collecting each
    position AND a 1-ply capture-perturbed variant (the non-quiet, search-visited slice)."""
    eng = AlphaBetaEngine(time_limit=1e9, max_depth=depth)
    seen = set()
    rng = random.Random(0)
    for g in range(games):
        b = chess.Board()
        for _ in range(rng.randint(2, 10)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        plies = 0
        while not b.is_game_over() and plies < 80:
            seen.add(b.fen())
            caps = [m for m in b.legal_moves if b.is_capture(m)]
            if caps:
                bp = b.copy(); bp.push(rng.choice(caps))
                if not bp.is_game_over():
                    seen.add(bp.fen())
            mv = rng.choice(list(b.legal_moves)) if rng.random() < 0.2 else eng.search(b)[0]
            if mv is None:
                break
            b.push(mv); plies += 1
        if (g + 1) % 25 == 0:
            print(f"  {g+1}/{games} games, {len(seen)} positions", flush=True)
    return list(seen)


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Dedup against existing dataset.
    existing = set()
    try:
        with open(OUT) as fh:
            for line in fh:
                try:
                    existing.add(json.loads(line)[0])
                except Exception:
                    continue
    except FileNotFoundError:
        pass

    t = time.time()
    print(f"generating search-visited positions from {games} depth-{depth} games...")
    fens = [f for f in gen_positions(games, depth) if f not in existing]
    print(f"{len(fens)} new positions in {time.time()-t:.0f}s; labelling with SF depth-{DEPTH}...")

    t = time.time()
    workers = max(1, mp.cpu_count() - 2)
    with mp.Pool(workers, initializer=_init) as pool:
        rows = [r for r in pool.map(_label, fens, chunksize=200) if r]
    with open(OUT, "a") as f:
        for fen, cp in rows:
            f.write(json.dumps([fen, cp]) + "\n")
    print(f"appended {len(rows)} rows to {OUT} in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
