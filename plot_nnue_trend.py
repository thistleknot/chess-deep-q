"""Chart the NNUE distillation climb: measured Elo vs corpus size (models/nnue_trend.jsonl).

Shows the trajectory the eval is on as data + coverage grow, against the bars that matter: pst @d2
(~1367, the equal-depth wall) and pst @d3 (~1672, the >1600 convergence target). The 156k point is the
balanced-data dip (wrong data type) and is flagged so the climb reads correctly. Regenerated each climb
cycle by climb_nnue.py. -> training_plots/nnue_trend.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TREND = "models/nnue_trend.jsonl"
OUT = "training_plots/nnue_trend.png"
PST_D2, PST_D3, LINEAR_CEIL = 1367, 1672, 1140


def main():
    pts = [json.loads(l) for l in open(TREND) if l.strip()]
    pts.sort(key=lambda p: p["records"])
    xs = [p["records"] / 1000.0 for p in pts]        # thousands of positions
    ys = [p["elo"] for p in pts]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.2))

    # reference bars
    ax.axhspan(1600, 1750, color="#3f7d5c", alpha=0.08, zorder=0)
    ax.axhline(PST_D3, color="#3f7d5c", ls="--", lw=1.4, zorder=1)
    ax.axhline(PST_D2, color="#a8503f", ls="--", lw=1.4, zorder=1)
    ax.axhline(LINEAR_CEIL, color="#8b8b8b", ls=":", lw=1.2, zorder=1)
    ax.text(xs[-1], PST_D3 + 8, "pst @d3  ~1672  (the >1600 target)", color="#2f6147", fontsize=9, ha="right")
    ax.text(xs[-1], PST_D2 + 8, "pst @d2  ~1367  (equal-depth wall)", color="#8a3f30", fontsize=9, ha="right")
    ax.text(xs[0], LINEAR_CEIL - 22, "linear-eval ceiling ~1140", color="#777", fontsize=8.5, ha="left")

    # the climb line
    ax.plot(xs, ys, "-", color="#9a6c2b", lw=1.6, zorder=2, alpha=0.7)
    for p in pts:
        x, y = p["records"] / 1000.0, p["elo"]
        balanced = "balanced" in p["label"]
        col = "#b0b0b0" if balanced else "#9a6c2b"
        ax.plot(x, y, "o", ms=9, color=col, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        dy = -20 if balanced else 14
        ax.annotate(f"{y}\n{p['records']//1000}k", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8.5,
                    color="#555" if balanced else "#3a2b12", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("corpus size — labeled positions (thousands, log scale)", fontsize=10.5)
    ax.set_ylabel("Elo (vs Stockfish@1320 anchor, pst@d2=1367)", fontsize=10.5)
    ax.set_title("NNUE distillation climb — Elo vs corpus size", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(700, 1760)
    ax.margins(x=0.12)

    os.makedirs("training_plots", exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT} ({len(pts)} points, latest {ys[-1]} Elo at {pts[-1]['records']//1000}k)")


if __name__ == "__main__":
    main()
