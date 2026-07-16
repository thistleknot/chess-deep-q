import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Chart the NNUE distillation climb: measured Elo vs corpus size (models/nnue_trend.jsonl).

Two tracks by search depth — d2 (the primary climb) and d3 (the depth lever) — against the bars that
matter: pst @d2 (~1367) and pst @d3 (~1672, the >1600 convergence target). The 156k point is the
balanced-data dip (flagged). Regenerated each climb cycle by climb_nnue.py. -> training_plots/nnue_trend.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TREND = "models/nnue_trend.jsonl"
OUT = "training_plots/nnue_trend.png"
PST_D2, PST_D3, LINEAR_CEIL = 1367, 1672, 1140
DEPTH_COL = {2: "#9a6c2b", 3: "#3f7d5c"}


def main():
    pts = [json.loads(l) for l in open(TREND) if l.strip()]
    for p in pts:
        p.setdefault("depth", 2)
    pts.sort(key=lambda p: (p["depth"], p["records"]))
    allx = [p["records"] / 1000.0 for p in pts]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.4))

    # reference bars
    ax.axhspan(1600, 1760, color="#3f7d5c", alpha=0.07, zorder=0)
    ax.axhline(PST_D3, color="#3f7d5c", ls="--", lw=1.4, zorder=1)
    ax.axhline(PST_D2, color="#a8503f", ls="--", lw=1.4, zorder=1)
    ax.axhline(LINEAR_CEIL, color="#8b8b8b", ls=":", lw=1.2, zorder=1)
    ax.text(max(allx), PST_D3 + 8, "pst @d3  ~1672  (>1600 target)", color="#2f6147", fontsize=9, ha="right")
    ax.text(max(allx), PST_D2 + 8, "pst @d2  ~1367", color="#8a3f30", fontsize=9, ha="right")
    ax.text(min(allx), LINEAR_CEIL - 24, "linear-eval ceiling ~1140", color="#777", fontsize=8.5, ha="left")

    # one line per depth track
    for depth in sorted({p["depth"] for p in pts}):
        tp = [p for p in pts if p["depth"] == depth]
        col = DEPTH_COL.get(depth, "#555")
        xs = [p["records"] / 1000.0 for p in tp]
        ys = [p["elo"] for p in tp]
        ax.plot(xs, ys, "-o", color=col, lw=1.6, ms=8, zorder=3,
                markeredgecolor="white", markeredgewidth=1.1, label=f"NNUE @d{depth}")
        for p in tp:
            x, y = p["records"] / 1000.0, p["elo"]
            balanced = "balanced" in p["label"]
            mcol = "#b0b0b0" if balanced else col
            if balanced:
                ax.plot(x, y, "o", ms=8, color=mcol, zorder=4, markeredgecolor="white", markeredgewidth=1.1)
            ax.annotate(f"{y}", (x, y), textcoords="offset points",
                        xytext=(0, -18 if balanced else 12), ha="center", fontsize=9,
                        color="#777" if balanced else col, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("corpus size — labeled positions (thousands, log scale)", fontsize=10.5)
    ax.set_ylabel("Elo (vs Stockfish@1320 anchor)", fontsize=10.5)
    ax.set_title("NNUE distillation climb — Elo vs corpus size, by search depth",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(700, 1770)
    ax.margins(x=0.14)
    ax.legend(loc="center right", frameon=True, fontsize=9.5)

    import os
    os.makedirs("training_plots", exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT} ({len(pts)} points)")


if __name__ == "__main__":
    main()
