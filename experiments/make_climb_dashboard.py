import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Generate a self-contained HTML dashboard for the NNUE distillation climb.

Embeds the current training_plots/nnue_trend.png (as a data URI) + the models/nnue_trend.jsonl points
as a table + a target readout. Re-run to refresh after each climb cycle. -> training_plots/climb_dashboard.html
"""
import base64
import json

PNG = "training_plots/nnue_trend.png"
TREND = "models/nnue_trend.jsonl"
OUT = "training_plots/climb_dashboard.html"
PST_D2, PST_D3 = 1367, 1672


def main():
    png_b64 = base64.b64encode(open(PNG, "rb").read()).decode()
    pts = [json.loads(l) for l in open(TREND) if l.strip()]
    pts.sort(key=lambda p: p["records"])
    best = max(pts, key=lambda p: p["elo"])

    rows = ""
    for p in pts:
        cls = "dip" if "balanced" in p["label"] else ("best" if p is best else "")
        sc = f"{p['score']:.2f}" if p.get("score") is not None else "—"
        rows += (f'<tr class="{cls}"><td class="mono">{p["records"]//1000}k</td>'
                 f'<td class="mono">d{p.get("depth", 2)}</td>'
                 f'<td class="mono b">{p["elo"]}</td><td class="mono">{p["elo_diff"]:+d}</td>'
                 f'<td class="mono">{sc}</td><td>{p["label"]}</td></tr>\n')

    gap = PST_D3 - best["elo"]
    html = f"""<title>NNUE Distillation Climb</title>
<style>
  :root {{ --ground:#f4f5f7; --surface:#fff; --ink:#1a1d23; --soft:#565b66; --faint:#8b909b;
    --line:#e0e3e9; --accent:#9a6c2b; --good:#3f7d5c; --bad:#a8503f; --s2:#eef0f4; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --ground:#14161b; --surface:#1b1e25; --ink:#e7e9ee;
    --soft:#a2a8b4; --faint:#6b717c; --line:#2a2e38; --accent:#cc9a54; --good:#74b791; --bad:#d4877a; --s2:#22262f; }} }}
  :root[data-theme="dark"] {{ --ground:#14161b; --surface:#1b1e25; --ink:#e7e9ee; --soft:#a2a8b4;
    --faint:#6b717c; --line:#2a2e38; --accent:#cc9a54; --good:#74b791; --bad:#d4877a; --s2:#22262f; }}
  :root[data-theme="light"] {{ --ground:#f4f5f7; --surface:#fff; --ink:#1a1d23; --soft:#565b66;
    --faint:#8b909b; --line:#e0e3e9; --accent:#9a6c2b; --good:#3f7d5c; --bad:#a8503f; --s2:#eef0f4; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ground); color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
  .wrap {{ max-width:860px; margin:0 auto; padding:clamp(24px,5vw,56px) clamp(18px,5vw,36px) 72px; }}
  .mono {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-variant-numeric:tabular-nums; }}
  .eyebrow {{ font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); font-weight:700; }}
  h1 {{ font-family:Charter,"Iowan Old Style",Georgia,serif; font-weight:600; font-size:clamp(26px,4vw,36px);
    margin:10px 0 6px; letter-spacing:-.01em; }}
  .sub {{ color:var(--soft); margin:0 0 24px; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:0 0 24px; }}
  .card {{ flex:1; min-width:150px; background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:16px 18px; }}
  .card .k {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); font-weight:700; }}
  .card .v {{ font-size:26px; font-weight:700; margin-top:4px; }}
  .card .v.accent {{ color:var(--accent); }}
  .card .v.good {{ color:var(--good); }}
  figure {{ margin:0 0 24px; background:var(--surface); border:1px solid var(--line); border-radius:12px;
    padding:14px; }}
  img {{ width:100%; height:auto; display:block; border-radius:6px; }}
  .tablewrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:var(--surface); }}
  table {{ border-collapse:collapse; width:100%; min-width:480px; font-size:14.5px; }}
  th {{ text-align:left; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--faint);
    font-weight:700; padding:12px 16px; border-bottom:1px solid var(--line); background:var(--s2); }}
  td {{ padding:11px 16px; border-bottom:1px solid var(--line); }}
  tr:last-child td {{ border-bottom:none; }}
  td.b {{ font-weight:700; }}
  tr.best td {{ background:color-mix(in srgb, var(--accent) 10%, transparent); }}
  tr.dip td {{ color:var(--faint); }}
  .note {{ color:var(--soft); font-size:13.5px; margin-top:18px; }}
</style>
<div class="wrap">
  <div class="eyebrow">Live · off-policy distillation of the value critic</div>
  <h1>NNUE distillation climb</h1>
  <p class="sub">Measured Elo of the learned eval vs corpus size, climbing toward the pst@d3 bar (~1672).</p>
  <div class="cards">
    <div class="card"><div class="k">best so far</div><div class="v accent">{best['elo']}</div></div>
    <div class="card"><div class="k">at corpus</div><div class="v mono">{best['records']//1000}k</div></div>
    <div class="card"><div class="k">target (pst@d3)</div><div class="v good">&gt;1600</div></div>
    <div class="card"><div class="k">gap to target</div><div class="v mono">{gap:+d}</div></div>
  </div>
  <figure><img alt="Elo vs corpus size" src="data:image/png;base64,{png_b64}"></figure>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Corpus</th><th>depth</th><th>Elo</th><th>vs pst</th><th>score</th><th>run</th></tr></thead>
      <tbody>
{rows}      </tbody>
    </table>
  </div>
  <p class="note">Two levers, both working. <b>Data (d2 track):</b> the 156k point is the balanced-data dip; the
  jump to 1144 came from <b>:Material-coverage:</b> (seeding material-imbalanced positions so the eval learns to
  value material), and the autonomous climb keeps growing corpus + coverage. <b>Depth (d3 track):</b> the SAME
  eval at one extra ply reads <b>{best['elo']}</b> — already above pst@d2, closing on the &gt;1600 zone; +337 Elo/ply.
  Convergence above 1600 comes from the two together (better eval × deeper search). Honest ceiling from the
  literature: strong club-to-IM (~1600–2000) with enough well-covered data — top-Stockfish stays out of reach.</p>
</div>
"""
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"saved {OUT} ({len(pts)} points, best {best['elo']} at {best['records']//1000}k)")


if __name__ == "__main__":
    main()
