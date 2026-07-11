# Self-judgement rubric — the agent's performance class

The operator runs multiple consoles; this one is graded continuously on the ONLY objective
that matters here: measured Elo progress. The agent (Claude Code session driving this repo)
prepends its self-assigned class to EVERY response — visible to the operator at a glance and
to the agent as a standing incentive/reminder.

## Classes

| Class                  | Condition                                                        |
|------------------------|------------------------------------------------------------------|
| Below Expectations     | no Elo improvement within the current 5-minute observation window across batches during epoch training, AND best confirmed measure < 1600 |
| Met Expectations       | Elo trending upward (rising SF samples / confirmed crowns / ladder rung climbs / rising rungs on data/rl_trend.jsonl) |
| Exceeded Expectations  | confirmed measure > 1600                                          |

## Protocol

- Assessment cadence = the 5-minute observer ticks (spec :Confirmed-crown: era metrics:
  sf_pts samples, crowned bests, opp_rung, ladder rungs).
- "Improvement" means MEASURED movement, not activity: code shipped, arms launched, and
  analyses written do not change the class — only the Elo signal does.
- The class is prepended to every agent response: `Below|Met|Exceeded Expectations:`.
- Honesty rule: when in doubt between two classes, take the lower. A noise spike is not a
  trend (:Confirmed-crown: applies to self-grading too).
- The grade is a lever, not a mood: a Below Expectations reading obliges the agent to ask,
  at that tick, whether to pivot (pre-registered pivot ladder in data/experiments.md).
- Below Expectations on a SIGNAL event (new SF sample / epoch verdict / arm verdict — not
  liveness ticks) additionally obliges a targeted WEB SEARCH on the current obstacle: the
  world may already have the answer (this campaign's biggest gains came from literature —
  TDLeaf, the donor features, the Rainbow ranking). Findings and their accept/reject land in
  data/experiments.md like sidecar advice.
