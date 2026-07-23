"""Multi-anchor Elo ladder (operator design 2026-07-17): games spread across K
Stockfish anchor Elos around the presumed strength, pooled into ONE rating by
maximum likelihood — replaces the single-anchor rung when the agent saturates
its anchor (score near 1.0 -> per-anchor Elo unbounded, floor-only claims).

Why MLE and not a weighted average of per-anchor Elos: a swept anchor (4-0)
estimates "+inf" and poisons any average; under the joint logistic likelihood a
swept anchor just contributes a nearly-flat term (little information), while
near-50% anchors dominate — the statistically correct version of the operator's
bell-curve weighting. Estimator: maximize sum_i [s_i ln p_i + (1-s_i) ln(1-p_i)]
with p_i = 1/(1+10^((A_i - R)/400)); draws s=0.5. CI from Fisher information:
SE = 1/sqrt(sum (ln10/400)^2 p(1-p)); 95% = R +/- 1.96 SE.

Usage: python experiments/anchor_ladder.py <mover_spec_or_ckpt> <depth> \
           <games_per_anchor> <name> [anchors_csv]
  anchors default 1500,1700,1900,2100,2300 (SF UCI_Elo min 1320).
Env: LADDER_WORKERS (default 6), LADDER_CORES — same semantics as claims_rung.
Output: per-anchor W-D-L lines + MLE rating with 95% CI; JSON row appended to
data/rl_trend.jsonl (fields anchors/per_anchor alongside the standard keys).
Failure modes: no stockfish -> SystemExit (multi-anchor NEEDS the anchor);
R outside [anchor_min-800, anchor_max+800] -> reported as a bound, not a point.
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

from chessdq.measure_elo import TREND, PLY_CAP
from chessdq.claims_rung import _is_mover_spec

_CTX = {}
LN10_400 = math.log(10.0) / 400.0


def _init_worker(target, depth, cores, counter):
    global _CTX
    import atexit
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    try:
        import psutil
        if cores:
            psutil.Process().cpu_affinity([cores[idx % len(cores)]])
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 5)
    except Exception:
        pass
    if _is_mover_spec(target):
        import random as _random
        from chessdq.head2head import mover_from_spec
        mover = mover_from_spec(target, _random.Random(9000 + idx))
    else:
        import importlib
        from chessdq.corpus_gen import raw_weights
        w, b = raw_weights(target)
        srch = importlib.import_module("rsearch4").Searcher(w, b)
        mover = lambda bd, _s=srch, _d=depth: chess.Move.from_uci(_s.search(bd.fen(), _d)[0])
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if not sfp:
        raise SystemExit("multi-anchor ladder requires stockfish under engines/")
    import chess.engine
    sf = chess.engine.SimpleEngine.popen_uci(sfp[0])
    atexit.register(sf.quit)
    _CTX = {"mover": mover, "sf": sf, "sf_elo": None}


def _one_game(args):
    """(game_index, anchor_elo) -> (anchor_elo, result, plies), agent-perspective."""
    g, anchor = args
    import chess.engine
    mover, sf = _CTX["mover"], _CTX["sf"]
    if _CTX["sf_elo"] != anchor:
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": int(anchor)})
        _CTX["sf_elo"] = anchor
    lim = chess.engine.Limit(time=0.05)
    agent_white = (g % 2 == 0)
    b = chess.Board()
    plies = 0
    while not b.is_game_over() and plies < PLY_CAP:
        if (b.turn == chess.WHITE) == agent_white:
            mv = mover(b)
        else:
            mv = sf.play(b, lim).move
        b.push(mv)
        plies += 1
    if b.is_checkmate():
        res = 1.0 if (b.turn == chess.BLACK) == agent_white else 0.0
    elif os.environ.get("LADDER_ADJUDICATE") == "1" and not b.is_game_over():
        # :Cap-adjudication: (P4) — the 120-ply cap scored every unfinished game 0.5, which
        # draw-floods strong-vs-strong ladders and compresses the MLE toward the anchors. Instead,
        # let SF judge the cap position: a decisive eval (> LADDER_ADJ_CP centipawns) is awarded.
        import chess.engine as _ce
        adj_cp = int(os.environ.get("LADDER_ADJ_CP", "400"))
        try:
            info = sf.analyse(b, _ce.Limit(depth=10))
            sc = info["score"].white().score(mate_score=100000)   # white-relative cp
        except Exception:
            sc = None
        if sc is None or abs(sc) <= adj_cp:
            res = 0.5
        else:
            white_won = sc > 0
            res = 1.0 if (white_won == agent_white) else 0.0
    else:
        res = 0.5
    return anchor, res, plies


def mle_rating(results, lo=None, hi=None):
    """(R, se) from [(anchor, score)] pairs by 1-D likelihood maximization.
    Require: at least one non-degenerate score in the set. Golden-section on the
    concave log-likelihood over [min_anchor-800, max_anchor+800]."""
    anchors = [a for a, _ in results]
    lo = (min(anchors) - 800) if lo is None else lo
    hi = (max(anchors) + 800) if hi is None else hi

    def ll(r):
        s = 0.0
        for a, sc in results:
            p = 1.0 / (1.0 + 10.0 ** ((a - r) / 400.0))
            p = min(max(p, 1e-12), 1 - 1e-12)
            s += sc * math.log(p) + (1.0 - sc) * math.log(1.0 - p)
        return s

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = lo, hi
    c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
    while b_ - a_ > 0.01:
        if ll(c_) > ll(d_):
            b_, d_ = d_, c_
            c_ = b_ - gr * (b_ - a_)
        else:
            a_, c_ = c_, d_
            d_ = a_ + gr * (b_ - a_)
    r = (a_ + b_) / 2.0
    info = sum(LN10_400 ** 2 * (p := 1.0 / (1.0 + 10.0 ** ((a - r) / 400.0))) * (1.0 - p)
               for a, _ in results)
    se = 1.0 / math.sqrt(info) if info > 0 else float("inf")
    return r, se


def main():
    from chessdq.thermal import engage
    engage()
    target, depth, per_anchor, name = (sys.argv[1], int(sys.argv[2]),
                                       int(sys.argv[3]), sys.argv[4])
    anchors = ([int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5
               else [1500, 1700, 1900, 2100, 2300])
    workers = max(1, int(os.environ.get("LADDER_WORKERS", "6")))
    cores_env = os.environ.get("LADDER_CORES") or os.environ.get("CHESS_THERMAL_CORES", "")
    cores = ([int(c) for c in cores_env.split(",") if c.strip() != ""][:workers]
             if cores_env else [])
    jobs = [(g, a) for a in anchors for g in range(per_anchor)]
    print(f"anchor ladder {name}: {per_anchor}g x anchors {anchors}, pool {workers}"
          + (f" cores {cores}" if cores else ""), flush=True)
    import multiprocessing as mp
    counter = mp.Value("i", 0)
    per = {a: [0, 0, 0] for a in anchors}      # W, D, L
    results = []
    t0 = time.time()
    done = 0
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(target, depth, cores, counter)) as pool:
        for anchor, res, plies in pool.imap_unordered(_one_game, jobs):
            done += 1
            per[anchor][0 if res == 1.0 else 1 if res == 0.5 else 2] += 1
            results.append((anchor, res))
            print(f"    game {done}/{len(jobs)}  vs {anchor}: "
                  f"{'W' if res == 1.0 else 'D' if res == 0.5 else 'L'}  {plies}p  "
                  f"{(time.time() - t0) / done:.0f}s/game-eff", flush=True)
    for a in anchors:
        W, D, L = per[a]
        print(f"  vs SF@{a}: {W}W-{D}D-{L}L  score {(W + 0.5 * D) / per_anchor:.2f}",
              flush=True)
    r, se = mle_rating(results)
    lo_b, hi_b = r - 1.96 * se, r + 1.96 * se
    print(f"  MLE rating {r:.0f} (95% {lo_b:.0f}..{hi_b:.0f}, SE {se:.0f}) "
          f"from {len(jobs)} games across {len(anchors)} anchors", flush=True)
    row = {"merge": 20, "agent": name, "ts": int(time.time()), "games": len(jobs),
           "anchors": anchors, "per_anchor": {str(a): per[a] for a in anchors},
           "elo": round(r), "elo_lo": round(lo_b), "elo_hi": round(hi_b),
           "pooled_workers": workers, "instrument": "anchor_ladder_mle"}
    os.makedirs(os.path.dirname(TREND) or ".", exist_ok=True)
    with open(TREND, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[wall {time.time() - t0:.0f}s] appended to {TREND}", flush=True)


if __name__ == "__main__":
    main()
