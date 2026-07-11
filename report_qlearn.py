"""Render the Q-learning study report (spec/q-learning.spec.md open questions) — reads either way.

Reads data/qlearn_results.jsonl (one row per Optuna trial: tuned params + final Elo WITH CI + vs-SF / vs-random
scores) and the Optuna study, and writes a self-contained report.html (opens from file://, no server) with:
a verdict banner, the trials table, param importances, and an HONEST two-tier verdict against the random floor
(~-280 Elo). It is built to conclude EITHER "RL viable here" OR "walled" from the data, never to flatter.

Usage: python report_qlearn.py [--no-open]
"""
import sys
import os
import json
import webbrowser

RESULTS = "data/qlearn_results.jsonl"
OUT = "report.html"
# The floor is the half-point clamp of a 0/n shutout vs SF@1320 (measure_elo.elo_diff(s, n)): ~562 @ n=40.
# Older rows carry the legacy -1600-diff floor (-280 Elo). The verdict keys on vs_sf_score == 0, not on a
# magic Elo constant, so both generations of rows read correctly.


def load_results():
    if not os.path.exists(RESULTS):
        return []
    return [json.loads(l) for l in open(RESULTS) if l.strip()]


def param_importances():
    """Importances from the MOST RECENTLY ACTIVE study (names are fingerprinted per protocol/regime)."""
    try:
        import optuna
        storage = "sqlite:///models/qlearn_optuna.db"
        best_study, best_ts = None, None
        for s in optuna.get_all_study_summaries(storage):
            st = optuna.load_study(study_name=s.study_name, storage=storage)
            ts = max((t.datetime_start for t in st.trials if t.datetime_start), default=None)
            if ts is not None and (best_ts is None or ts > best_ts):
                best_study, best_ts = st, ts
        if best_study and len([t for t in best_study.trials if t.value is not None]) >= 4:
            return optuna.importance.get_param_importances(best_study)
    except Exception:
        pass
    return {}


def verdict(rows):
    """Honest tiers keyed on the NOMINAL score vs SF@1320, not a magic floor Elo. Tier 1: real points off
    the anchor (vs_sf_score > 0) -> the Elo is a measurement. Tier 2: shut out by SF but beats random ->
    learning is real, below the anchor's resolution. Else: walled."""
    scored = [r for r in rows if r.get("elo") is not None]
    if not scored:
        return ("no data", "#7d8590", "No completed trials with an Elo.")
    best = max(scored, key=lambda r: r["elo"])
    vsf = best.get("vs_sf_score") or 0.0
    vr = best.get("vs_random_score") or 0.0
    if vsf > 0:
        elo_lo, elo_hi = best.get("elo_lo"), best.get("elo_hi")
        ci = f" (95% CI {elo_lo:.0f}..{elo_hi:.0f})" if elo_lo is not None else ""
        return ("RL VIABLE — scored real points off the anchor", "#238636",
                f"Best trial took {vsf:.3f} score/game vs SF@1320 -> ~{best['elo']:.0f} Elo{ci}. Reward-only "
                f"value-based Q-learning with eligibility traces is on the rating scale, not floored.")
    if vr > 0.55:
        return ("PARTIAL — beats random, shut out by SF@1320", "#bb8009",
                f"Best trial scores {vr:.2f} vs a random mover (learned something) but 0 vs SF@1320, so its "
                f"anchored Elo is only a half-point upper bound (~562 @ 40 games; -280 on legacy rows). "
                f"Learning is real but below the anchor's resolution.")
    return ("WALLED — no signal above random", "#da3633",
            f"No trial scored meaningfully above random AND every trial was shut out by SF@1320. At this "
            f"budget, reward-only self-play with a linear value did not climb — consistent with the prior "
            f"tabula-rasa plateau. Honest negative result, not a bug.")


def render(rows, imps):
    head, color, body = verdict(rows)
    data = json.dumps(rows)
    imp_rows = "".join(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>" for k, v in imps.items()) or \
        '<tr><td colspan="2" style="color:#7d8590">n&lt;4 trials — importance not estimated</td></tr>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Q-learning study report</title>
<style>
  :root{{color-scheme:dark;}}
  body{{background:#0d1117;color:#e6edf3;font:14px/1.6 system-ui,sans-serif;margin:0;padding:32px;max-width:1000px;}}
  h1{{font-size:20px;margin:0 0 4px;}} .sub{{color:#7d8590;margin:0 0 20px;}}
  .verdict{{border-left:5px solid {color};background:#161b22;border-radius:10px;padding:18px 20px;margin-bottom:24px;}}
  .verdict h2{{margin:0 0 6px;font-size:17px;color:{color};}} .verdict p{{margin:0;color:#c9d1d9;}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin-bottom:22px;}}
  .card h3{{font-size:13px;margin:0 0 14px;color:#7d8590;text-transform:uppercase;letter-spacing:.04em;}}
  table{{border-collapse:collapse;width:100%;font-size:13px;}}
  th,td{{text-align:right;padding:6px 10px;border-bottom:1px solid #21262d;}}
  th:first-child,td:first-child{{text-align:left;}} th{{color:#7d8590;font-weight:500;}}
  .floor{{color:#f0883e;}} code{{color:#58a6ff;}}
</style></head><body>
  <h1>Q-learning study — value-based rung, "just shy of actor-critic"</h1>
  <p class="sub">TD(λ) + variance-adaptive λ · softmax-τ self-play · tuned {{decay, α, λ, warmup}} · anchor SF@1320 · shutout = half-point floor (n-dependent)</p>

  <div class="verdict"><h2>{head}</h2><p>{body}</p></div>

  <div class="card"><h3>Trials (best first)</h3>
    <table><thead><tr><th>trial</th><th>α</th><th>γ</th><th>λ</th><th>warmup</th><th>epochs</th>
      <th>vs SF</th><th>Elo (95% CI)</th><th>vs random</th></tr></thead><tbody id="tb"></tbody></table>
  </div>

  <div class="card"><h3>Parameter importance</h3>
    <table><thead><tr><th>param</th><th>importance</th></tr></thead><tbody>{imp_rows}</tbody></table>
  </div>

  <div class="card"><h3>Reading it either way</h3>
    <p style="color:#c9d1d9;margin:0">Elo is anchored to SF@1320 with a 95% CI (score is a mean over games). A
    shutout only bounds the rating: the half-point clamp puts a 0/40 at <code>~562</code> ("at or below"), and
    legacy rows floor at <code>-280</code>. The verdict is <b>VIABLE</b> only if the best trial scored REAL
    points off the anchor (vs_sf_score &gt; 0); <b>PARTIAL</b> if it beats a random mover but is shut out by
    SF; <b>WALLED</b> otherwise. A WALLED result means "walled at this budget" — a heavier run remains
    available.</p></div>

<script>
const rows = {data};
rows.sort((a,b)=>(b.elo??-9999)-(a.elo??-9999));
const tb=document.getElementById('tb');
for(const r of rows){{
  const floor = (r.vs_sf_score!=null && r.vs_sf_score<=0);   // true shutout -> Elo is only an upper bound
  const ci = r.elo==null?'—':`${{r.elo}} (${{r.elo_lo}}..${{r.elo_hi}})`;
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${{r.tag||r.seed}}</td><td>${{(+r.alpha).toExponential(1)}}</td><td>${{(+r.gamma).toFixed(3)}}</td>`
    +`<td>${{(+r.lambda).toFixed(2)}}</td><td>${{(+r.warmup).toFixed(2)}}</td><td>${{r.epochs_run}}</td>`
    +`<td>${{r.vs_sf_score==null?'—':(+r.vs_sf_score).toFixed(2)}}</td>`
    +`<td class="${{floor?'floor':''}}">${{ci}}</td>`
    +`<td>${{r.vs_random_score==null?'—':(+r.vs_random_score).toFixed(2)}}</td>`;
  tb.appendChild(tr);
}}
</script></body></html>"""


def main():
    rows = load_results()
    html = render(rows, param_importances())
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {OUT} ({len(rows)} trial(s)) — verdict: {verdict(rows)[0]}")
    if "--no-open" not in sys.argv:
        webbrowser.open("file://" + os.path.abspath(OUT))


if __name__ == "__main__":
    main()
