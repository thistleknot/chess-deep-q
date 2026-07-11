---
description: 'Cross-cutting observability — anchor the current agent to Stockfish for an Elo, log a trend, render a self-contained dashboard webpage. Pulled forward from Merge 2 so every rung is watchable.'
import:
  - environment
---

***definitions***

- :Elo-harness: (`measure_elo.py`) plays the current merge's agent vs two rungs — uniform random (ordinal baseline) and Stockfish @1320 (the anchored rung) — and reports Elo = 1320 + :Elo-diff:(score). The agent is a move function passed in, so each merge swaps its learner without editing the harness. At Merge 0 the agent is `random_policy`, so the anchored score is ~0 and Elo floors below the anchor.
- :Elo-diff: is `400·log10(s/(1−s))` with s clamped to [1e-4, 1−1e-4], so a shutout maps to a finite floor (~−1600 gap) instead of −∞. A score below 0.05 is flagged as floored (not a resolved rating).
- :Trend-log: is `data/rl_trend.jsonl`, one appended row per measurement: `{merge, agent, ts, games, vs_random_score, vs_sf_W/D/L, vs_sf_score, elo, avg_len}`. Append-only — the ladder's history.
- :Dashboard: (`dashboard.py` → `dashboard.html`) reads the :Trend-log:, embeds it into a self-contained HTML page (no server, no external libraries, opens from `file://`), and opens it in the browser. Shows the latest Elo / score / W-D-L / avg-length tiles, an Elo-over-merges line with the 1600 goal line, and a runs table.

***implementation reqs***

- Observability is CROSS-CUTTING: it imports Merge 0's environment but no learning rung imports it, and it never changes reward or transition semantics. Each rung, after training, calls `measure_elo.py` (or its API) with its agent to append a :Trend-log: row.
- The anchor is the Stockfish binary already in `engines/**/stockfish*.exe` at `UCI_Elo=1320` (its weakest setting). If absent, the harness reports no anchored Elo and the row's `elo` is null — it must not crash.
- The :Dashboard: is regenerated on demand (`python dashboard.py`); it is not a live server. A live server is a later concern if wanted.

***functional specs***

- Given the current agent as a move function, When :Elo-harness: runs N games/rung with alternating colors, Then it returns W/D/L from the agent's perspective and appends exactly one :Trend-log: row.
- Given an anchored score s, When Elo is computed, Then Elo = 1320 + :Elo-diff:(s), and When s < 0.05 Then the result is marked floored (a floor sentinel, not a rating the rung has earned).
- Given a Merge 0 (random) agent, Then the measured result is ~0 vs SF@1320 and ~0.50 vs random — the honest bottom the learning rungs climb from.
- Given the :Trend-log: has ≥1 row, When :Dashboard: renders, Then the page opens offline with the latest tiles, the Elo line (with the 1600 goal line), and the runs table; When the log is empty, Then it shows a "run measure_elo first" prompt instead of crashing.
