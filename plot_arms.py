"""Elo-per-feature-set comparison page (operator: "we should be plotting elo per feature set").

Two single-axis panels (never dual-axis): (A) pooled final Elo — every from-scratch arm
measured by the IDENTICAL KILL-CHECK protocol, plus bake-off study bests as they land;
(B) best confirmed crown (d2 scale) per arm. Dot plot, CI whiskers where measured, arms
grouped/colored by feature-set FAMILY (fixed slot order, never cycled), direct labels,
table view below (accessibility), dark console theme.

Canonical numbers come from data/experiments.md (hand-pinned here with their entries);
live numbers (bake-off bests, b0/mlpb finals) are re-read on every build.

Usage: python plot_arms.py  -> data/compare.html   (server serves it at /compare)
"""
import glob
import json
import os
import re

# family -> dark-mode categorical slots (validated reference palette, fixed order)
FAM = {"809": "#3987e5", "PCA": "#199e70", "Kanerva": "#c98500", "capacity": "#9085e9"}

CANONICAL = [
    # (arm, family, pooled_final, (lo,hi) or None, best_crown, note)
    ("clean 809 raw",        "809",     756, None, 9.40,  "pure seed, raw space"),
    ("self-ZCA 809",         "809",     788, None, 8.83,  "own-games whitening"),
    ("zero pop (ZCA*)",      "809",     None, None, 11.17, "population peak; ZCA contaminated"),
    ("material pop (ZCA*)",  "809",     None, None, 13.94, "declared constants"),
    ("Kanerva-512 (21ep)",   "Kanerva", 744, None, 6.24,  "5k->K512->ZCA, first leg"),
    ("Kanerva-512 (30ep)",   "Kanerva", 700, None, 7.69,  "full horizon"),
    ("pst-769 p7 (graduated)", "809",   885, None, 12.87, "round-2 winner; ceiling relay-confirmed"),
]


def live_rows():
    rows = []
    for name, fam, tag in (("bake kc-raw", "809", "bake_kc-raw"),
                           ("bake kc-zca", "809", "bake_kc-zca"),
                           ("bake pca-320", "PCA", "bake_pca-320"),
                           ("bake nk-512", "Kanerva", "bake_nk-512"),
                           ("bake k809-2048", "Kanerva", "bake_k809"),
                           ("organ surprise", "809", "organ_surprise"),
                           ("organ +GRPO", "809", "organ_grpo"),
                           ("organ +replayT", "809", "organ_replayt"),
                           ("organ DDQN", "809", "organ_ddqn"),
                           ("organ GRPO-raw", "809", "bake2_grporaw"),
                           ("bake2 pst-769", "809", "bake2_pst769"),
                           ("bake2 5k raw", "Kanerva", "bake2_hk5k"),
                           ("bake2 5k std", "Kanerva", "bake2_hk5k-std"),
                           ("bake2 809+conj", "Kanerva", "bake2_concat"),
                           ("bake2 5k PCA-512", "PCA", "bake2_pca5k"),
                           ("organ lr-warmup", "809", "bake3_lrwu"),
                           ("organ phase-mix", "809", "bake3_phase"),
                           ("organ dis-gamma", "809", "bake3_disgamma"),
                           ("bake4 kpst 4-kingbucket", "809", "bake4_kpst"),
                           ("bake4 mlp64 crelu+adam", "capacity", "bake4_mlpcr"),
                           ("bake4 linear adam (ctl)", "capacity", "bake4_linadam")):
        p = f"data/{tag}.log"
        if os.path.exists(p):
            txt = open(p, errors="replace").read()
            m = re.search(r"best elo ([0-9.]+)", txt)
            if m:
                rows.append((name + " (study best)", fam, float(m.group(1)), None, None, "3-trial study"))
            else:                                          # study still running: best-so-far per trial line
                ts = [float(v) for v in re.findall(r"trial \d+: elo (-?[0-9.]+)", txt)]
                ts = [t for t in ts if t > -900]           # drop the -999 crashed-trial sentinel
                if ts:
                    rows.append((f"{name} (LIVE, {len(ts)} trial{'s' if len(ts) > 1 else ''})",
                                 fam, max(ts), None, None, "study running"))
    for name, fam, tag in (("probe linear+adam s25", "capacity", "probe_lin25"),
                           ("probe linear+adam s50", "capacity", "probe_lin50"),
                           ("probe mlp64cr+adam s25", "capacity", "probe_mlp25"),
                           ("probe mlp64cr+adam s50", "capacity", "probe_mlp50")):
        p = f"data/{tag}.log"
        if os.path.exists(p):
            m = re.search(r"KILL-CHECK\s+elo\s+(-?[0-9.]+)", open(p, errors="replace").read())
            if m:
                rows.append((name, fam, float(m.group(1)), None, None,
                             "volume-curve probe (100% pts = bake4 781/783)"))
    if os.path.exists("data/qlearn_results.jsonl"):
        for ln in open("data/qlearn_results.jsonl"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("tag") in ("b0", "mlpb") and r.get("elo") is not None:
                fam = "capacity" if r["tag"] == "mlpb" else "809"
                rows.append((f"{r['tag']} final", fam, float(r["elo"]),
                             (r.get("elo_lo"), r.get("elo_hi")), None, "baseline ladder"))
    return rows


def h2h_rows():
    """Duel-differential results (head2head.py) — OWN scale (duel Elo is compressed);
    never plotted on the pooled axis (statistic-class separation, operator rule)."""
    rows = []
    for p in sorted(glob.glob("data/h2h_*.md")):
        for ln in open(p, errors="replace"):
            m = re.match(r"h2h (\S+): A=(\S+) vs B=(\S+) \| (\d+)g score ([0-9.]+) -> "
                         r"Elo ([+-]?\d+) \(95% ([+-]?\d+)\.\.([+-]?\d+)\) \| decisive (\d+)%", ln)
            if m:
                rows.append(m.groups())
    return rows


def panel(title, rows, key, unit):
    vals = [(a, f, v, ci) for (a, f, v, ci, *_ ) in rows if v is not None]
    if not vals:
        return f"<div class='card'><h2>{title}</h2><p class='sub'>not in yet</p></div>"
    lo = min((ci[0] if ci and ci[0] else v) for _, _, v, ci in vals)
    hi = max((ci[1] if ci and ci[1] else v) for _, _, v, ci in vals)
    lo, hi = lo - (hi - lo) * 0.06 - 1, hi + (hi - lo) * 0.06 + 1
    W, RH, padL, padR = 860, 30, 210, 60
    H = len(vals) * RH + 46
    X = lambda v: padL + (v - lo) / (hi - lo) * (W - padL - padR)
    out = [f"<div class='card'><h2>{title}</h2>",
           f"<svg viewBox='0 0 {W} {H}' style='width:100%;height:auto'>"]
    for gv in range(int(lo // 100 + 1) * 100, int(hi), 100):
        out.append(f"<line x1='{X(gv):.0f}' y1='24' x2='{X(gv):.0f}' y2='{H-22}' stroke='#30363d' stroke-width='1'/>"
                   f"<text x='{X(gv):.0f}' y='{H-8}' fill='#7d8590' font-size='11' text-anchor='middle'>{gv}</text>")
    for i, (arm, fam, v, ci) in enumerate(sorted(vals, key=lambda r: -r[2])):
        y = 34 + i * RH
        c = FAM[fam]
        out.append(f"<text x='{padL-10}' y='{y+4}' fill='#c3c2b7' font-size='12' text-anchor='end'>{arm}</text>")
        if ci and ci[0] is not None and ci[1] is not None:
            out.append(f"<line x1='{X(ci[0]):.0f}' y1='{y}' x2='{X(ci[1]):.0f}' y2='{y}' stroke='{c}' stroke-width='2' opacity='0.5'/>")
        out.append(f"<circle cx='{X(v):.0f}' cy='{y}' r='5' fill='{c}' stroke='#161b22' stroke-width='2'>"
                   f"<title>{arm}: {v:.0f} {unit}</title></circle>"
                   f"<text x='{X(v)+10:.0f}' y='{y+4}' fill='#ffffff' font-size='12'>{v:.0f}</text>")
    out.append("</svg></div>")
    return "".join(out)


def build():
    rows_final = CANONICAL + live_rows()
    rows_crown = [(a, f, cr, None) for (a, f, _pf, _ci, cr, *_ ) in CANONICAL if cr is not None]
    legend = "".join(f"<span style='margin-right:16px'><span style='display:inline-block;width:10px;height:10px;"
                     f"border-radius:5px;background:{c};margin-right:5px'></span>{k}</span>" for k, c in FAM.items())
    tbl = "<table class='lad'><tr><th>arm</th><th>family</th><th>pooled final</th><th>best crown</th><th>note</th></tr>"
    for (a, f, pf, _ci, cr, note) in CANONICAL:
        tbl += f"<tr><td>{a}</td><td>{f}</td><td>{pf or '—'}</td><td>{cr or '—'}</td><td>{note}</td></tr>"
    for r in live_rows():
        tbl += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]:.0f}</td><td>—</td><td>{r[5]}</td></tr>"
    tbl += "</table>"
    h2h_tbl = ("<table class='lad'><tr><th>duel</th><th>A</th><th>B</th><th>games</th>"
               "<th>A−B duel Elo (95%)</th><th>decisive</th></tr>")
    for (tag, a, b, g, _s, e, lo_, hi_, dec) in h2h_rows():
        h2h_tbl += (f"<tr><td>{tag}</td><td>{a}</td><td>{b}</td><td>{g}</td>"
                    f"<td>{e} ({lo_}..{hi_})</td><td>{dec}%</td></tr>")
    h2h_tbl += "</table>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Elo per feature set</title>
<style>body{{background:#0d1117;color:#e6edf3;font:14px/1.5 system-ui;margin:0;padding:24px;max-width:960px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:18px}}
.card h2{{font-size:13px;margin:0 0 10px;color:#7d8590;text-transform:uppercase}}
.sub{{color:#7d8590}} table.lad{{width:100%;border-collapse:collapse;font-size:12px}}
table.lad th{{color:#7d8590;text-align:left;padding:4px 8px;border-bottom:1px solid #30363d}}
table.lad td{{padding:4px 8px;border-bottom:1px solid #21262d}}</style></head><body>
<h1 style='font-size:18px'>Elo per feature set — from-scratch arms, one protocol</h1>
<p class='sub'>{legend}</p>
{panel("pooled final Elo (KILL-CHECK protocol)", [(a,f,v,ci) for (a,f,v,ci,*_ ) in rows_final], "pf", "Elo")}
{panel("best confirmed crown (d2 scale)", rows_crown, "cr", "crown")}
<div class='card'><h2>head-to-head duels (DUEL Elo — compressed scale, A minus B; never comparable to the pooled axis)</h2>{h2h_tbl}</div>
<div class='card'><h2>table view</h2>{tbl}</div>
</body></html>"""
    open("data/compare.html", "w", encoding="utf-8").write(html)
    return html


if __name__ == "__main__":
    build()
    print("data/compare.html written")
