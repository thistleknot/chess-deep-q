import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Score-vs-compute frontier tuner for the net-guided beam (spec :Compute-frontier:).

Sweeps the two knobs (depth, total_ops) -> a schedule -> the NetBeam, measures Elo D vs SF@1320 on
a FIXED, color-balanced opening suite shared with a zero-search baseline (rfr), and reports the
paired ADVANTAGE (excess over rfr, common-random-numbers variance reduction), the Sharpe (advantage
per sec/move -- the real, time-denominated efficiency), Pareto-dominance, a tangency/frontier summary
line, and the best config that fits under a call budget. Plots D vs total_calls -> training_plots.

Usage:
  python frontier.py [--depths 3,4,5,6] [--openings 6] [--rfr policy|random] [--sf-ms 50]
                     [--budget-calls 400] [--anchor 1320]
"""
import argparse
import glob
import math
import random
import time

import chess
import chess.engine
import torch

from chessdq.beam import NetBeam, schedule_for, total_calls
from chessdq.resnet_model import ChessResNet
from chessdq.net_search import policy_priors

# Fixed non-book midgame openings; each is played by the config as BOTH colors (paired CRN).
OPENINGS = [
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
    "r1bq1rk1/pp2bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
    "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 10",
    "r1bqr1k1/ppp2ppp/2np1n2/2b5/2B1P3/2NP1N2/PPPQ1PPP/R3K2R w KQ - 0 9",
    "r3k2r/pppq1ppp/2npbn2/2b1p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 9",
]


def find_sf():
    hits = glob.glob("engines/**/stockfish*.exe", recursive=True)
    return hits[0] if hits else None


def load_net(device):
    net = ChessResNet().to(device)
    sd = torch.load("models/tower.pth", map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    net.eval()
    return net


def elo_diff(s):
    s = min(max(s, 1e-4), 1 - 1e-4)
    return 400 * math.log10(s / (1 - s))


def play_paired(mover, sf, anchor, sf_ms, cap=100):
    """Play `mover` from each OPENING as BOTH colors vs SF@anchor. Returns (per_opening_scores,
    sec_per_move): per_opening_scores[i] in [0,1] averaged over the two colors (common random
    numbers -> pairing with the baseline cancels shared opening/color/SF noise)."""
    lim = chess.engine.Limit(time=sf_ms / 1000.0)
    scores, t_mover, n_moves = [], 0.0, 0
    for fen in OPENINGS:
        pair = 0.0
        for mover_white in (True, False):
            board = chess.Board(fen)
            plies = 0
            while not board.is_game_over() and plies < cap:
                if (board.turn == chess.WHITE) == mover_white:
                    t0 = time.time(); mv = mover(board); t_mover += time.time() - t0; n_moves += 1
                else:
                    mv = sf.play(board, lim).move
                if mv is None or mv not in board.legal_moves:
                    break
                board.push(mv); plies += 1
            if board.is_checkmate():
                won = (board.turn == chess.BLACK) == mover_white
                pts = 1.0 if won else 0.0
            elif board.is_game_over():
                pts = 0.5
            else:
                cp = sf.analyse(board, chess.engine.Limit(time=0.05))["score"].white().score(mate_score=100000)
                pts = 0.5 if (cp is None or abs(cp) <= 150) else (1.0 if (cp > 150) == mover_white else 0.0)
            pair += pts
        scores.append(pair / 2.0)
    return scores, (t_mover / max(1, n_moves))


def summarize(scores, anchor):
    s = sum(scores) / len(scores)
    return anchor + elo_diff(s), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="3,4,5,6")
    ap.add_argument("--rfr", choices=["policy", "random"], default="policy")
    ap.add_argument("--sf-ms", type=int, default=50)
    ap.add_argument("--anchor", type=int, default=1320)
    ap.add_argument("--pair-k", type=int, default=2)
    ap.add_argument("--budget-calls", type=int, default=0, help="0 = no budget filter")
    args = ap.parse_args()

    sf_path = find_sf()
    if not sf_path:
        print("Stockfish not found under engines/."); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(device)

    # rfr: zero-search baseline (V in A = Q - V). policy = argmax policy head; random = uniform.
    if args.rfr == "policy":
        def rfr_mover(b):
            pr = policy_priors(b, net, device)
            return max(pr, key=pr.get) if pr else random.choice(list(b.legal_moves))
    else:
        def rfr_mover(b):
            return random.choice(list(b.legal_moves))

    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.anchor})
    n_op = len(OPENINGS)
    print(f"Frontier tuner: net-guided beam vs SF@{args.anchor}, {n_op} openings x2 colors (paired), "
          f"rfr={args.rfr}\n")

    rfr_scores, rfr_spm = play_paired(rfr_mover, sf, args.anchor, args.sf_ms)
    rfr_D, rfr_s = summarize(rfr_scores, args.anchor)
    print(f"rfr({args.rfr}): D~{rfr_D:.0f} (s={rfr_s:.2f})  sec/move={rfr_spm*1000:.1f}ms\n")

    rows = []
    for depth in [int(x) for x in args.depths.split(",")]:
        # canonical Fibonacci schedule at this depth (the natural (depth -> ops) point).
        widths = schedule_for(depth, 0, args.pair_k)  # scale=1 => canonical fib ladder
        calls = total_calls(widths, args.pair_k)
        beam = NetBeam(net, device, widths, pair_k=args.pair_k)
        cfg_scores, spm = play_paired(lambda b: beam.select(b), sf, args.anchor, args.sf_ms)
        D, s = summarize(cfg_scores, args.anchor)
        # paired advantage (common random numbers): a_i = score_cfg,i - score_rfr,i
        adv = [c - r for c, r in zip(cfg_scores, rfr_scores)]
        a_mean = sum(adv) / len(adv)
        a_sd = (sum((x - a_mean) ** 2 for x in adv) / max(1, len(adv) - 1)) ** 0.5
        a_se = a_sd / (len(adv) ** 0.5)
        sharpe_t = (D - rfr_D) / (spm * 1000) if spm > 0 else 0.0   # excess Elo per ms/move (real)
        sharpe_c = (D - rfr_D) / calls if calls else 0.0            # excess Elo per call (proxy)
        rows.append(dict(depth=depth, widths=widths, calls=calls, spm=spm, D=D, s=s,
                         adv=a_mean, adv_se=a_se, sharpe_t=sharpe_t, sharpe_c=sharpe_c))
        print(f"depth={depth} sched={widths} calls={calls} sec/move={spm*1000:5.0f}ms "
              f"D~{D:.0f} adv={a_mean:+.3f}+-{1.96*a_se:.3f} "
              f"Sharpe(t)={sharpe_t:+.2f}/ms Sharpe(c)={sharpe_c:+.2f}/call")
    sf.quit()

    # Pareto (max D at <= calls) and tangency (max time-Sharpe).
    for r in rows:
        r["pareto"] = not any(o["calls"] <= r["calls"] and o["D"] > r["D"] for o in rows if o is not r)
    tang = max(rows, key=lambda r: r["sharpe_t"]) if rows else None
    fit = [r for r in rows if not args.budget_calls or r["calls"] <= args.budget_calls]
    best_fit = max(fit, key=lambda r: r["D"]) if fit else None

    print("\n--- frontier ---")
    print(f"{'depth':>5} {'calls':>6} {'ms/mv':>6} {'D':>6} {'adv':>7} {'Sharpe/ms':>10} {'Pareto':>7}")
    for r in sorted(rows, key=lambda r: r["calls"]):
        print(f"{r['depth']:>5} {r['calls']:>6} {r['spm']*1000:>6.0f} {r['D']:>6.0f} "
              f"{r['adv']:>+7.3f} {r['sharpe_t']:>+10.2f} {'  *' if r['pareto'] else '':>7}")
    if tang:
        print(f"\nTANGENCY (max time-Sharpe): depth={tang['depth']} calls={tang['calls']} "
              f"D~{tang['D']:.0f} @ {tang['spm']*1000:.0f}ms/move")
    if best_fit:
        b = f"<= {args.budget_calls} calls" if args.budget_calls else "any budget"
        print(f"BUDGET PICK ({b}): depth={best_fit['depth']} D~{best_fit['D']:.0f} calls={best_fit['calls']}")

    _plot(rows, rfr_D, tang)


def _plot(rows, rfr_D, tang):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    import os
    rows = sorted(rows, key=lambda r: r["calls"])
    xs = [r["calls"] for r in rows]; ys = [r["D"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "o-", color="#4C78A8", label="beam configs")
    pe = [r for r in rows if r["pareto"]]
    ax.plot([r["calls"] for r in pe], [r["D"] for r in pe], "s", color="#E45756", label="Pareto frontier")
    ax.axhline(rfr_D, ls="--", color="gray", label=f"rfr ~{rfr_D:.0f}")
    if tang:
        ax.plot([0, tang["calls"]], [rfr_D, tang["D"]], ":", color="#54A24B", label="CML (tangency)")
        ax.annotate("tangency", (tang["calls"], tang["D"]))
    ax.set_xlabel("total_calls (compute)"); ax.set_ylabel("Elo D vs SF anchor")
    ax.set_title("Score-vs-compute frontier (net-guided beam)"); ax.legend(); ax.grid(alpha=0.3)
    os.makedirs("training_plots", exist_ok=True)
    path = "training_plots/frontier.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"\nplot -> {path}")


if __name__ == "__main__":
    main()
