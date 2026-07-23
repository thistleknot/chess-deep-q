"""Claims-ladder rung (spec/bullet-route.spec.md step 4): rsearch-dN over a kc/amap
linear ckpt, measured by the SAME protocol that produced the 1441/1484/1540/1670 claims
(SF@1320 anchor, 0.05s/move, alternating colors, ply cap 120, CI from score SE).

:Pooled-ladder: (operator 2026-07-15 — "a queue of 6 games at a time"): games fan out
over LADDER_WORKERS processes (default 6), each worker PINNED TO ITS OWN CORE (clock
fairness: SF's 0.05s/move must never be starved by a sibling game) with its own SF
engine + Searcher built once; games dispatch as-completed with a PER-GAME progress
line (no more silent 10-game windows). LADDER_WORKERS=1 = the legacy serial
measure_elo.measure path, byte-identical protocol.

Usage: python claims_rung.py <ckpt.pt> <depth> <games> <agent_name>
Env: LADDER_WORKERS (default 6), LADDER_CORES ("0,1,2,3,4,5" — pool pinning; defaults
to the first LADDER_WORKERS cores of CHESS_THERMAL_CORES or the machine).
Failure modes: ZCA identity gate failure -> SystemExit (corpus_gen.raw_weights);
no stockfish -> ordinal only (not claims-grade); a dead worker -> its game re-raises
in the parent (fail fast — a partial rung is not a rung).
"""
import glob
import json
import math
import os
import sys
import time

import chess

from chessdq.measure_elo import elo_diff, TREND, PLY_CAP

_CTX = {}

# head2head mover-spec prefixes — a rung target with one of these is a Python mover,
# not a raw ckpt (plain paths, incl. Windows drive paths, never start with them)
_SPEC_PREFIXES = ("sa:", "beam:", "puct:", "rs:", "enc:", "nnue:", "kcz:")


def _is_mover_spec(target):
    return target.startswith(_SPEC_PREFIXES)


def _init_worker(ckpt, depth, cores, counter):
    """Per-process setup: pin ONE core, build the Searcher and a private SF engine."""
    global _CTX
    import atexit
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    try:
        import psutil
        if cores:
            psutil.Process().cpu_affinity([cores[idx % len(cores)]])
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 5)
    except Exception:
        pass
    if _is_mover_spec(ckpt):
        # spec-mover rung (spec/search-arms.spec.md): any head2head mover spec
        # (sa:/beam:/puct:/rs:/enc:/...) measured by the SAME anchored protocol.
        import random as _random
        from chessdq.head2head import mover_from_spec
        mover = mover_from_spec(ckpt, _random.Random(9000 + idx))
    else:
        import importlib
        from chessdq.corpus_gen import raw_weights
        w, b = raw_weights(ckpt)
        srch = importlib.import_module("rsearch4").Searcher(w, b)
        mover = lambda bd, _s=srch, _d=depth: chess.Move.from_uci(_s.search(bd.fen(), _d)[0])
    sf = None
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if sfp:
        import chess.engine
        sf = chess.engine.SimpleEngine.popen_uci(sfp[0])
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
        atexit.register(sf.quit)
    _CTX = {"mover": mover, "sf": sf}


def _one_game(args):
    """Play ONE anchored game. args = (game_index, opp). Returns (g, result, plies)
    with result from the AGENT's perspective: 1 win / 0.5 draw-or-cap / 0 loss."""
    g, opp = args
    import random as _random
    import chess.engine
    mover, sf = _CTX["mover"], _CTX["sf"]
    rng = _random.Random(1000 + g)
    lim = chess.engine.Limit(time=0.05)
    agent_white = (g % 2 == 0)
    b = chess.Board()
    plies = 0
    while not b.is_game_over() and plies < PLY_CAP:
        if (b.turn == chess.WHITE) == agent_white:
            mv = mover(b)
        elif opp == "sf":
            mv = sf.play(b, lim).move
        else:
            mv = rng.choice(list(b.legal_moves))
        b.push(mv)
        plies += 1
    if b.is_checkmate():
        res = 1.0 if (b.turn == chess.BLACK) == agent_white else 0.0
    else:
        res = 0.5
    return g, res, plies


def _pooled_rung(ckpt, depth, games, opp, workers, cores):
    """(W, D, L, avg_plies) — as-completed pool with per-game streaming."""
    import multiprocessing as mp
    counter = mp.Value("i", 0)
    W = D = L = 0
    plies_total = 0
    done = 0
    t0 = time.time()
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(ckpt, depth, cores, counter)) as pool:
        for g, res, plies in pool.imap_unordered(_one_game, [(g, opp) for g in range(games)]):
            done += 1
            W += res == 1.0
            D += res == 0.5
            L += res == 0.0
            plies_total += plies
            print(f"    game {done}/{games} (#{g})  {W}W-{D}D-{L}L  {plies} plies  "
                  f"{(time.time() - t0) / done:.1f}s/game-effective", flush=True)
    return int(W), int(D), int(L), plies_total / max(1, games)


def main():
    from chessdq.thermal import engage
    engage()                    # :Thermal-guard: — cap + below-normal; workers re-pin singly
    ckpt, depth, games, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    workers = max(1, int(os.environ.get("LADDER_WORKERS", "6")))
    if workers == 1:
        from chessdq.measure_elo import measure
        if _is_mover_spec(ckpt):
            import random as _random
            from chessdq.head2head import mover_from_spec
            mover = mover_from_spec(ckpt, _random.Random(9000))
        else:
            import importlib
            from chessdq.corpus_gen import raw_weights
            w, b = raw_weights(ckpt)
            srch = importlib.import_module("rsearch4").Searcher(w, b)
            mover = lambda bd: chess.Move.from_uci(srch.search(bd.fen(), depth)[0])
        measure(mover, name, games, merge=20)
        return
    cores_env = os.environ.get("LADDER_CORES") or os.environ.get("CHESS_THERMAL_CORES", "")
    cores = [int(c) for c in cores_env.split(",") if c.strip() != ""][:workers] if cores_env else []
    print(f"measuring {name}: {games} games/rung, pool of {workers}"
          + (f" on cores {cores}" if cores else ""), flush=True)
    t = time.time()
    rW, rD, rL, r_len = _pooled_rung(ckpt, depth, games, "random", workers, cores)
    s_rand = (rW + 0.5 * rD) / games
    print(f"  vs random          {rW}W-{rD}D-{rL}L  score {s_rand:.2f}", flush=True)
    sW, sD, sL, sf_len = _pooled_rung(ckpt, depth, games, "sf", workers, cores)
    s_sf = (sW + 0.5 * sD) / games
    elo = 1320 + elo_diff(s_sf, games)
    se = math.sqrt(max(s_sf * (1 - s_sf), 1e-9) / games)
    elo_lo = 1320 + elo_diff(s_sf - 1.96 * se, games)
    elo_hi = 1320 + elo_diff(s_sf + 1.96 * se, games)
    print(f"  vs SF@1320         {sW}W-{sD}D-{sL}L  score {s_sf:.2f}  -> ~{elo:.0f} Elo "
          f"(95% CI {elo_lo:.0f}..{elo_hi:.0f})", flush=True)
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    row = {"merge": 20, "agent": name, "commit": commit, "ts": int(time.time()),
           "games": games, "vs_random_score": round(s_rand, 4),
           "vs_sf_W": sW, "vs_sf_D": sD, "vs_sf_L": sL, "vs_sf_score": round(s_sf, 4),
           "elo": round(elo), "elo_lo": round(elo_lo), "elo_hi": round(elo_hi),
           "avg_len": round((r_len + sf_len) / 2), "pooled_workers": workers}
    os.makedirs(os.path.dirname(TREND) or ".", exist_ok=True)
    with open(TREND, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[wall {time.time() - t:.0f}s] appended to {TREND}", flush=True)


if __name__ == "__main__":
    main()
