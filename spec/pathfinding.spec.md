# Multidimensional path-finding — navigation contract for weight space

Operator's framing, formalized: training is path-finding in 809-dim weight space toward the
global maximum of a height function we can only sample noisily and expensively (confirmed
Elo). RRT*/A* intuitions map as follows — with OUT OF BOUNDS defined explicitly, because Elo
alone cannot mark boundaries.

## The map

| Path-finding concept | This system | Signal used |
|---|---|---|
| Node (waypoint)      | confirmed checkpoint (`_best.pt`, :Confirmed-crown:) | confirmed Elo |
| Tree                 | checkpoint lineage (kc → kc3 → …)                    | — |
| Height / goal        | confirmed Elo (ladder rungs, 60g/200g measures)      | expensive, noisy |
| Edge / step          | one epoch of TD updates                              | — |
| Re-root (prune branch) | anchor revert to last confirmed node               | collapse guard |
| Frontier sampling    | graded ladder + :Opponent-diet: reach games          | matchmaking window |

## :Out-of-bounds: — three kinds, none of them measured in Elo

1. **Walls (hard OOB)** — regions where the learner is BROKEN, detectable per-batch without
   any Elo sample. Sensors and default trip-lines (all already logged or one line to log):
   - loss > 5× its epoch-start moving average (divergence)
   - td_sigma > 0.8 sustained over a full batch cycle (target noise blowup)
   - decisive rate < 0.05 over an epoch (draws-only ⇒ signal starvation — S&B reward sparsity)
   - |V| saturation: >50% of sampled leaf values with |v| > 0.95 (tanh dead zone, DRLIA 057)
   - head-weight L2 norm > 5× its value at the last confirmed node
   Tripping ANY wall ⇒ immediate revert to the last confirmed node + log `WALL <sensor>`;
   do not wait for the epoch's Elo.
2. **Pits (soft OOB)** — reachable, legal, but low: strength < REVERT_FRAC × confirmed bar
   (existing collapse guard). Pits are NOT remembered as regions — in 809 dims with noisy
   samples, region-memory is meaningless; the TREE is the memory (re-root and step again;
   fresh data makes re-entering the same pit unlikely unless a systematic force pushes there,
   and systematic forces get diagnosed and removed — e.g. :Backed-bootstrap:).
3. **Step bound (trust region)** — a step can be OOB by SIZE alone: if the greedy policy
   changes too much in one epoch, the new node's Elo sample cannot be trusted to describe a
   nearby point. Sensor: greedy-move AGREEMENT on a fixed 200-position probe set (positions
   sampled once from graded games, frozen on disk). Trip-line: agreement < 0.6 vs the epoch's
   start ⇒ treat as wall (revert), because we cannot distinguish "long jump to a better
   basin" from "teleport into noise" at our measurement budget (TRPO/KL rationale,
   DRLIA 147; agreement is the cheap proxy for policy-KL in a value-greedy system).

## Status

Spec first (this file). Wall sensors 1–3 are one-line additions over existing metrics;
saturation, norm, and the probe-agreement sensor are small follow-ups — queued behind the
current intervention-queue arms (spec/intervention-queue.spec.md), since sensors protect
progress but do not create it.
