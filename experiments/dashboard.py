import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Render the RL-ladder dashboard (spec/observability.spec.md) and open it in the browser.

Reads data/rl_trend.jsonl (written by measure_elo.py), builds a SELF-CONTAINED dashboard.html with the data
embedded (no server, no external libs — opens straight from file://), and opens it. Re-run any time to
refresh after a new measurement. This is the webpage; every merge that appends a trend row shows up here.

Usage: python dashboard.py            # render + open
       python dashboard.py --no-open  # just write dashboard.html
"""
import sys
import os
import json
import webbrowser

TREND = "data/rl_trend.jsonl"
OUT = "dashboard.html"


def load_rows():
    if not os.path.exists(TREND):
        return []
    return [json.loads(l) for l in open(TREND) if l.strip()]


def render(rows):
    data = json.dumps(rows)
    latest = rows[-1] if rows else {}
    elo = latest.get("elo")
    elo_str = "—" if elo is None else str(elo)
    score = latest.get("vs_sf_score")
    score_str = "—" if score is None else f"{score:.2f}"
    wdl = f'{latest.get("vs_sf_W","–")}-{latest.get("vs_sf_D","–")}-{latest.get("vs_sf_L","–")}'
    avglen = latest.get("avg_len", "—")
    agent = latest.get("agent", "—")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>chess RL ladder</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#0d1117; color:#e6edf3; font:14px/1.5 system-ui,sans-serif; margin:0; padding:32px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#7d8590; margin:0 0 24px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:16px; margin-bottom:28px; }}
  .tile {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px 18px; }}
  .tile .k {{ color:#7d8590; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .tile .v {{ font-size:30px; font-weight:600; margin-top:6px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin-bottom:24px; }}
  .card h2 {{ font-size:14px; margin:0 0 14px; color:#7d8590; text-transform:uppercase; letter-spacing:.04em; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ text-align:right; padding:6px 10px; border-bottom:1px solid #21262d; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:#7d8590; font-weight:500; }}
  .floor {{ color:#f0883e; }}
  .empty {{ color:#7d8590; padding:20px 0; }}
</style></head><body>
  <h1>chess RL ladder — observability</h1>
  <p class="sub">latest agent: <b>{agent}</b> · anchor: Stockfish @1320 · <code>data/rl_trend.jsonl</code></p>

  <div class="tiles">
    <div class="tile"><div class="k">Elo (anchored)</div><div class="v">{elo_str}</div></div>
    <div class="tile"><div class="k">Score vs SF@1320</div><div class="v">{score_str}</div></div>
    <div class="tile"><div class="k">W-D-L vs SF</div><div class="v" style="font-size:22px">{wdl}</div></div>
    <div class="tile"><div class="k">Avg game length</div><div class="v">{avglen}</div></div>
  </div>

  <div class="card">
    <h2>Elo over merges</h2>
    <svg id="chart" width="100%" height="220" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
  </div>

  <div class="card">
    <h2>Runs</h2>
    <table id="tbl"><thead><tr>
      <th>merge</th><th>agent</th><th>games</th><th>score vs SF</th><th>Elo</th><th>vs random</th><th>avg len</th>
    </tr></thead><tbody></tbody></table>
    <div class="empty" id="empty" style="display:none">No runs yet — run <code>python measure_elo.py</code> first.</div>
  </div>

<script>
const rows = {data};
const tb = document.querySelector('#tbl tbody');
if (!rows.length) {{ document.getElementById('empty').style.display='block'; }}
for (const r of rows) {{
  const floor = (r.vs_sf_score != null && r.vs_sf_score < 0.05);
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{r.merge}}</td><td>${{r.agent}}</td><td>${{r.games}}</td>`
    + `<td>${{r.vs_sf_score==null?'—':r.vs_sf_score.toFixed(2)}}</td>`
    + `<td class="${{floor?'floor':''}}">${{r.elo==null?'—':r.elo}}${{floor?' (floor)':''}}</td>`
    + `<td>${{r.vs_random_score==null?'—':r.vs_random_score.toFixed(2)}}</td>`
    + `<td>${{r.avg_len}}</td>`;
  tb.appendChild(tr);
}}
// simple Elo line
const pts = rows.map((r,i)=>({{x:i, y:r.elo}})).filter(p=>p.y!=null);
const svg = document.getElementById('chart');
if (pts.length) {{
  const ys = pts.map(p=>p.y), ymin=Math.min(...ys,0), ymax=Math.max(...ys,1600);
  const X=i=>pts.length<2?400:40+i*(720/(pts.length-1));
  const Y=v=>200-((v-ymin)/((ymax-ymin)||1))*180;
  // 1600 goal line
  const gy=Y(1600);
  svg.innerHTML += `<line x1="40" y1="${{gy}}" x2="760" y2="${{gy}}" stroke="#238636" stroke-dasharray="4 4"/>`
    + `<text x="762" y="${{gy+4}}" fill="#238636" font-size="11">1600</text>`;
  let d = pts.map((p,i)=>`${{i?'L':'M'}}${{X(p.x)}} ${{Y(p.y)}}`).join(' ');
  svg.innerHTML += `<path d="${{d}}" fill="none" stroke="#58a6ff" stroke-width="2"/>`;
  pts.forEach((p,i)=>{{ svg.innerHTML += `<circle cx="${{X(p.x)}}" cy="${{Y(p.y)}}" r="4" fill="#58a6ff"/>`; }});
}} else {{
  svg.innerHTML = `<text x="400" y="110" fill="#7d8590" font-size="13" text-anchor="middle">no Elo yet</text>`;
}}
</script>
</body></html>"""


def main():
    rows = load_rows()
    html = render(rows)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {OUT} ({len(rows)} run(s))")
    if "--no-open" not in sys.argv:
        webbrowser.open("file://" + os.path.abspath(OUT))


if __name__ == "__main__":
    main()
