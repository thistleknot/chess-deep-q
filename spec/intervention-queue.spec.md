# Intervention queue — autoresearch ordering for the Elo objective

The improvement loop is an optimizer whose evaluation costs 30–60 min (600 graded games +
measures). Automation frameworks (TextGrad/DSPy-style propose→measure→diagnose→propose) add
value here ONLY through disciplined ordering: run the cheapest, most-diagnostic arm first;
every arm is single-variable; every failure must falsify something specific (the "textual
gradient" that reorders the queue). Grading per spec/expectations.spec.md.

## Protocol per arm

1. One variable changed vs the incumbent lineage. 2. Budget: ≤ 600 games or 3 epochs,
   whichever first (extend only on a confirmed crown). 3. Readout: pooled SF Elo + confirmed
   crowns + ladder rung; 60g rung on best if promising. 4. On failure, record WHAT is
   falsified before dequeuing the next arm.

## Queue (2026-07-10, post-:Backed-bootstrap:; reorder on every diagnosis)

| # | Arm | Cost | Grounding | Failure falsifies |
|---|-----|------|-----------|-------------------|
| 0 | RUNNING: depth-3 targets (kc3) | 1 leg | KnightCap d4+ full-width | "target depth was the gap" |
| 1 | Opponent-diet forcing: fixed share of next-rung-up "reach games" regardless of matchmaking window | knob-level code | KnightCap's rising FICS opponents; DRLIA 059 (forgetting: all-weak diet has no above-heuristic signal) | "data distribution was the gap" |
| 1.5 | Texel tuning pass: logistic regression of V (kc features) on OWN-game outcomes over pooled quiet positions from all graded games | ~40 lines, reuses game data | chessprogramming.org Texel's Tuning Method (Österlund 2014) — the community's post-TDLeaf successor for linear evals; pure MC regression = zero bootstrap pathologies; from-scratch compliant (own outcomes, no SF labels) | "the TD family itself was the gap" |
| 2 | Width-16 generation targets | config only | more backed values per position | "target breadth was the gap" |
| 3 | Double-learning bootstrap: frozen net selects the PV leaf, LIVE net evaluates it | ~10 lines | S&B 013–015 (residual max-bias within the frozen net's own noise) | "remaining max-bias was the gap" |
| 4 | Prioritized replay on |TD error| | ~20 lines | DRLIA 149 | "uniform sampling was the gap" |
| 5 | Online per-move updates (drop replay buffer) | restructure | KnightCap-faithful TD; DRLIA 060 | "the replay formulation was the gap" |
| 6 | Distributional value head | new head + loss | DRLIA 129/141/144 | "point estimates were the gap" |

Standing exit: any arm that produces a CONFIRMED crown trend gets extended (ride the
gradient) instead of dequeuing.

## :Sidecar: — the intern (TextGrad/DSPy-style advisor, zero new infrastructure)

A cheap LLM subagent acts as proposal generator; the primary agent stays the decision layer.
Contract per invocation:
- INPUT (the "state"): this queue, the last ~10 results from data/experiments.md (arm, params,
  pooled Elo, verdict), and the current diagnosis.
- OUTPUT (the "textual gradient"): (a) a suggested queue permutation with one-line rationale
  per move; (b) at most ONE novel arm not in the queue, with grounding (paper/skill) and what
  its failure would falsify; JSON.
- The primary agent ACCEPTS/REJECTS each suggestion with a stated reason, logged to
  data/experiments.md. Advice is advisory: no sidecar output ever launches compute directly.
- Cadence: on every arm verdict (not on liveness ticks). Model tier: cheapest available
  (intern, not architect).
- Search hybrid (spec/expectations.spec.md): bidirectional — advisers direct the searches,
  search findings become every adviser's next context.

## :Search-council: — three advisers on WHAT TO SEARCH

On each arm verdict (or Below-graded signal event), spawn three cheap advisers in parallel,
each with a FIXED lens, each returning 1–2 web-search queries + one line of intent:
1. **textgrad lens** (diagnosis-driven): given the latest failure/verdict, what missing
   knowledge does the textual gradient point at? (e.g. "why does X plateau despite Y")
2. **dspy lens** (module/metric-driven): which pipeline module is underperforming its
   contract, and what does the field use instead of that module? (e.g. "alternatives to TD
   for tuning linear chess evals" → surfaced Texel tuning)
3. **autoresearch lens** (exploration): what adjacent SOTA/technique exists that the queue
   has never considered? (novelty search, deliberately outside the current diagnosis)
The primary agent dedupes/ranks the queries, runs the top ~2, and arbitrates findings into
the queue with logged accept/reject. Diversity rule: the three advisers run as separate
agents with no shared draft (independent proposals, council-not-chorus).

## :Texel: (arm 1.5 — texel_fit.py, the non-TD family)

Donor method: chessprogramming.org "Texel's Tuning Method" (Österlund 2014) — logistic
regression of the eval on OWN-game outcomes. Pure Monte Carlo; no bootstrap, no traces, no
max operators anywhere: the bias family that ate the TD arms cannot exist here.

1. Generation: load the kc7 seed as V; play `GAMES` (default 1000) 1-ply ε-greedy (ε=0.1)
   games against a fixed opponent mix (heuristic .4, sf-skill0 .3, skill2 .2, skill5 .1),
   colors alternating. 1-ply keeps generation fast (no search) — Texel labels positions by
   OUTCOME, so behavior strength matters less than position coverage.
2. Dataset: positions after ply 12, QUIET only (side to move not in check, previous move not
   a capture), encoded with `encode_features`, labeled y = game outcome for White (1/0.5/0).
   ~30-40 quiet positions/game → ~30k+ rows for 809 params.
3. Fit: minimize logloss of sigmoid(K·V(x)) vs y, K a learned scalar; full-batch Adam,
   init from the kc7 seed. Checkpoint models/qlearn_texel{,_best}.pt (enc=kc).
4. Measure: 60g rungs at 1-ply greedy AND d2 w8 search → ladder. Accept per queue protocol.
Failure falsifies: "the TD family itself was the gap."

## :Opponent-diet: (arm 1 pre-spec, qlearn.py OpponentLadder)

- `QLEARN_OPP_REACH` (float, default 0.25 when OPP=graded): with this probability a
  generation game is played vs the rung ABOVE the current matchmaking rung (clamped to the
  ladder top). Reach-game scores are EXCLUDED from the matchmaking window — the window
  measures rung-MATCHED performance; feeding expected losses in would bias the ladder down
  and block exactly the climbs the diet exists to enable.
- Guarantee: the training distribution always contains above-current-strength opponents —
  the signal KnightCap's rising FICS pool provided and our boundary-locked ladder does not.
- Identity: part of PROTO when tuning (`-reach{p}`); infrastructure control otherwise.
