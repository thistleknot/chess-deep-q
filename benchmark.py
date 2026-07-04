"""Unified cross-track benchmark (plan Phase 2): place all three tracks on ONE ladder so the
"which track" decision is evidence-based, not assumed.

- A  search strength:        AlphaBetaEngine(pst) @ 0.3s/move  (reference point)
- C  net-eval-in-search:     net-minimax with the SF-distilled tower (= B2 iter-0)
- B  self-play net curves:   B1 heuristic-bootstrap, B2 SF-distilled-bootstrap, over N iterations

Emits a table + training_plots/track_benchmark.png (per-iteration ladder curve vs the A/C lines).
Rungs: random / heuristic-1ply (ordinal). Run measure_ladder / measure_sf for the SF-anchored read.

Usage: python benchmark.py [iters] [games_per_iter] [depth] [ladder_games]
"""
import os, sys, time
import torch

from resnet_model import ChessResNet
from engine import AlphaBetaEngine, pst_eval
from selfplay import expert_iteration, heuristic_eval
from measure_ladder import random_mover, heuristic_mover, adj_pst, play, elo_diff


def load_tower(dev):
    net = ChessResNet().to(dev)
    sd = torch.load("models/tower.pth", map_location=dev)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd); net.eval()
    return net


def ladder_point(mover, games):
    r = {}
    for name, opp in (("random", random_mover), ("heuristic", heuristic_mover)):
        W, D, L = play(mover, opp, games, adj_pst)
        r[name] = (W + 0.5 * D) / (W + D + L)
    return r


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    gpi = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    lg = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cross-track benchmark: {iters} iters x {gpi} games, depth {depth}, {lg} ladder games\n")

    # A — search strength reference (alpha-beta pst @ 0.3s), on the same ordinal rungs.
    t = time.time()
    eng = AlphaBetaEngine(eval_fn=pst_eval, time_limit=0.3)
    A = ladder_point(lambda b: eng.search(b)[0], lg)
    print(f"[A] alpha-beta pst @0.3s: vs_random={A['random']:.2f} vs_heuristic={A['heuristic']:.2f} "
          f"(and ~1428 vs SF@1320, measured) [{time.time()-t:.0f}s]\n")

    # B2 — SF-distilled bootstrap (start from tower.pth, net eval throughout). iter-0 ≈ track C.
    print("[B2] SF-distilled bootstrap (from tower.pth):")
    net_b2 = load_tower(dev)
    curve_b2 = expert_iteration(net_b2, dev, iters, gpi, depth=depth, ladder_games=lg,
                                eval_for_iter=lambda i: None, label="B2")

    # B1 — heuristic bootstrap (fresh net; heuristic leaf first half, net takes over second half).
    print("\n[B1] heuristic bootstrap (fresh net, eps 1->0):")
    net_b1 = ChessResNet().to(dev)
    half = max(1, iters // 2)
    curve_b1 = expert_iteration(net_b1, dev, iters, gpi, depth=depth, ladder_games=lg,
                                eval_for_iter=lambda i: heuristic_eval if i < half else None, label="B1")

    _table(A, curve_b1, curve_b2)
    _plot(A, curve_b1, curve_b2)


def _table(A, b1, b2):
    print("\n--- vs heuristic-1ply (ordinal score; 0.5 = even) ---")
    print(f"{'iter':>4} {'B1(heur-boot)':>14} {'B2(distilled)':>14}")
    for i in range(max(len(b1), len(b2))):
        s1 = f"{b1[i]['vs_heuristic']:.2f}" if i < len(b1) else "-"
        s2 = f"{b2[i]['vs_heuristic']:.2f}" if i < len(b2) else "-"
        print(f"{i:>4} {s1:>14} {s2:>14}")
    print(f"ref A (alpha-beta pst @0.3s) vs_heuristic = {A['heuristic']:.2f}")
    print("C (net-minimax tower, distilled) = B2 iter-0 above")


def _plot(A, b1, b2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["iter"] for r in b1], [r["vs_heuristic"] for r in b1], "o-", label="B1 heuristic-boot")
    ax.plot([r["iter"] for r in b2], [r["vs_heuristic"] for r in b2], "s-", label="B2 SF-distilled")
    ax.axhline(A["heuristic"], ls="--", color="gray", label="A alpha-beta pst @0.3s")
    ax.axhline(0.5, ls=":", color="black", alpha=0.4, label="even")
    ax.set_xlabel("self-play iteration"); ax.set_ylabel("score vs heuristic-1ply")
    ax.set_title("Cross-track benchmark: self-play curves vs search reference")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    os.makedirs("training_plots", exist_ok=True)
    fig.tight_layout(); fig.savefig("training_plots/track_benchmark.png", dpi=150); plt.close(fig)
    print("\nplot -> training_plots/track_benchmark.png")


if __name__ == "__main__":
    main()
