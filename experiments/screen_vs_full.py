"""Screen-vs-full surrogate calibration (operator design 2026-07-17, ladder-freeze
revision): NO new ladder runs — y comes from each net's RECORDED full-run receipt
(claims/rung bands already in the ledger), x from the cheap 20g screen duel vs the
FIXED champion d2 lens (the contender-screen instrument). Question answered:
does the cheap screen rank-predict the recorded full-run truth?

Points (y receipts, all pre-existing):
  champion  amap-897 @d9   floor 1724 (50g scout, anchor-saturated; band 1724..2118)
  kc1670    kc @d9         1670 (1605..1762) @200g claims
  triv1540  kc @d7         1540 (1486..1605) @200g claims
  vol1484   kc @d7         1484 (1434..1542) @200g claims
  hyb       sa-mover @d9-budget  1314 (1224..1404) @60g rung — x ALSO pre-existing
            (+597 screen trial 0, data/h2h_sa_hyb_t0.md), so zero games for this point.

Usage: python experiments/screen_vs_full.py   (env H2H_SHARDS passes through)
Outputs: data/screen_vs_full.json + data/screen_vs_full.png; duel rows data/h2h_svf_*.md.
Failure modes: missing hyb receipt files -> that point dropped with a warning;
<3 surviving points -> SystemExit (not plottable).
"""
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF = "enc:amap:models/champion.pt"          # the fixed screen reference lens

# (name, screen mover spec or None-if-prerecorded, recorded full Elo, lo, hi)
SUBJECTS = [
    ("champion", "enc:amap:models/champion.pt",          1724, 1724, 2118),
    ("kc1670",   "kcz:models/champion_backup_kc1670.pt", 1670, 1605, 1762),
    ("triv1540", "kcz:models/qlearn_triv_best.pt",       1540, 1486, 1605),
    ("vol1484",  "kcz:models/qlearn_vol_best.pt",        1484, 1434, 1542),
    ("hyb",      None,                                   1314, 1224, 1404),
]


def run_screen(name, spec):
    """20g H2H vs REF -> (elo, lo, hi) or None."""
    env = dict(os.environ, H2H_CAP="20", H2H_SHARDS=os.environ.get("H2H_SHARDS", "6"),
               H2H_BLOCK="20")
    out = subprocess.run([sys.executable, "head2head.py", spec, REF, "20", f"svf_{name}"],
                         capture_output=True, text=True, env=env, timeout=3600).stdout
    m = re.findall(r"-> Elo ([+-]?\d+) \(95% ([+-]?\d+)\.\.([+-]?\d+)\)", out or "")
    if not m:
        print(f"[svf] screen {name}: NO VERDICT; tail: {(out or '')[-200:]}", flush=True)
        return None
    e, lo, hi = map(float, m[-1])
    print(f"[svf] screen {name}: {e:+.0f} ({lo:+.0f}..{hi:+.0f})", flush=True)
    return e, lo, hi


def hyb_screen_receipt():
    """The hybrid's pre-existing screen verdict (Optuna trial 0 duel vs REF)."""
    path = "data/h2h_sa_hyb_t0.md"
    if not os.path.exists(path):
        return None
    m = re.findall(r"-> Elo ([+-]?\d+) \(95% ([+-]?\d+)\.\.([+-]?\d+)\)",
                   open(path, encoding="utf-8").read())
    return tuple(map(float, m[-1])) if m else None


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for r, i in enumerate(order):
            rk[i] = r + 1.0
        return rk
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    t0 = time.time()
    pts = []
    for name, spec, rec, rec_lo, rec_hi in SUBJECTS:
        x = hyb_screen_receipt() if spec is None else run_screen(name, spec)
        if x is None:
            print(f"[svf] DROPPED {name} (no screen verdict)", flush=True)
            continue
        pts.append(dict(name=name, x=x[0], x_lo=x[1], x_hi=x[2],
                        y=rec, y_lo=rec_lo, y_hi=rec_hi))
    if len(pts) < 3:
        raise SystemExit(f"only {len(pts)} points survived — not plottable")
    rho = spearman([p["x"] for p in pts], [p["y"] for p in pts])
    out = dict(ts=int(time.time()), ref=REF, spearman=round(rho, 3),
               y_source="recorded full-run receipts (ladder freeze — no new runs)",
               points=pts)
    with open("data/screen_vs_full.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    ink, muted, blue = "#1a1f26", "#6b7280", "#4269d0"
    for p in pts:
        ax.errorbar(p["x"], p["y"], xerr=[[p["x"] - p["x_lo"]], [p["x_hi"] - p["x"]]],
                    yerr=[[p["y"] - p["y_lo"]], [p["y_hi"] - p["y"]]],
                    fmt="o", ms=8, color=blue, ecolor=muted, elinewidth=1.2,
                    capsize=2, zorder=3)
        ax.annotate(p["name"], (p["x"], p["y"]), xytext=(7, 6),
                    textcoords="offset points", fontsize=9, color=ink)
    ax.set_xlabel("screen tier: 20g Elo diff vs champion d2 lens", color=ink)
    ax.set_ylabel("full tier: recorded full-run Elo (claims/rung receipts)", color=ink)
    ax.set_title(f"Does the cheap screen predict the full run?  "
                 f"Spearman ρ = {rho:.2f}  (n={len(pts)})", color=ink, fontsize=11)
    ax.grid(True, color="#e5e7eb", linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig("data/screen_vs_full.png")
    print(f"[svf] DONE n={len(pts)} spearman={rho:.2f} wall={time.time() - t0:.0f}s "
          f"-> data/screen_vs_full.png + .json", flush=True)


if __name__ == "__main__":
    main()
