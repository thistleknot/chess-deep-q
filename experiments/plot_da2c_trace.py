import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Plot the DA2C :Loss-trace: + amplification signals — spec/self-play-leela.spec.md.

Reads every models/da2c_trace_<wid>.jsonl shard and renders a 2x2 figure:
    (0,0) actor loss        (0,1) critic loss
    (1,0) mean game length  (1,1) mean Elo (per-iteration ladder)

The actor and critic losses are ADVERSARIAL and chaotic BY DESIGN (Zai & Brown ch5 fig 5.13) -- they
are traced separately and are diagnostic only (a run to +/-inf or a collapse to 0 flags a bug). The
REAL objective is the rising Elo / game-length curve. Output: training_plots/da2c_loss.png.

Usage: python plot_da2c_trace.py
"""
import os
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows():
    """Pool trace rows across all worker shards, sorted for loss/Elo by their global update index."""
    actor, critic, glen, elo = [], [], [], []
    for path in sorted(glob.glob("models/da2c_trace_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "actor_loss" in r:
                    actor.append((r["update"], r["actor_loss"]))
                    critic.append((r["update"], r["critic_loss"]))
                elif "game_len" in r:
                    glen.append(r["game_len"])
                elif "elo" in r:
                    elo.append((r["update"], r["elo"]))
    actor.sort()
    critic.sort()
    elo.sort()
    return actor, critic, glen, elo


def rolling(vals, w=20):
    """Trailing-window mean (smooths the chaotic per-step signal)."""
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        window = vals[lo:i + 1]
        out.append(sum(window) / len(window))
    return out


def _loss_panel(ax, pts, title, color):
    if pts:
        xs = [u for u, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, lw=0.6, alpha=0.4, color=color, label="per update")
        ax.plot(xs, rolling(ys), lw=1.8, color=color, label="rolling(20)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("global update")
    ax.set_ylabel("loss")


def main():
    actor, critic, glen, elo = load_rows()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    _loss_panel(ax[0, 0], actor, "actor loss (policy-gradient)", "tab:blue")
    _loss_panel(ax[0, 1], critic, "critic loss (value MSE)", "tab:red")

    # mean game length (ch5 fig 5.14/5.15): x = game index, rolling mean over games.
    if glen:
        idx = list(range(len(glen)))
        ax[1, 0].plot(idx, glen, lw=0.6, alpha=0.4, color="tab:green", label="per game")
        ax[1, 0].plot(idx, rolling(glen, 20), lw=1.8, color="tab:green", label="rolling(20)")
        ax[1, 0].legend(fontsize=8)
    else:
        ax[1, 0].text(0.5, 0.5, "no data", ha="center", va="center", transform=ax[1, 0].transAxes)
    ax[1, 0].set_title("mean game length (plies)")
    ax[1, 0].set_xlabel("game index")
    ax[1, 0].set_ylabel("plies")

    # mean Elo (per-iteration ladder vs heuristic-1ply) -- the real objective.
    if elo:
        xs = [u for u, _ in elo]
        ys = [v for _, v in elo]
        ax[1, 1].plot(xs, ys, "-o", lw=1.8, color="tab:purple")
    else:
        ax[1, 1].text(0.5, 0.5, "no ladder rows yet", ha="center", va="center",
                      transform=ax[1, 1].transAxes)
    ax[1, 1].set_title("mean Elo vs heuristic-1ply (ladder)")
    ax[1, 1].set_xlabel("global update")
    ax[1, 1].set_ylabel("Elo diff")
    ax[1, 1].axhline(0.0, color="grey", lw=0.8, ls="--")

    fig.suptitle("DA2C trace -- actor/critic losses are ADVERSARIAL & chaotic BY DESIGN "
                 "(ch5 fig 5.13); the RISING Elo / game-length curve is the real objective",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("training_plots", exist_ok=True)
    out = "training_plots/da2c_loss.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  "
          f"(loss rows: {len(actor)}, games: {len(glen)}, ladder rows: {len(elo)})")


if __name__ == "__main__":
    main()
