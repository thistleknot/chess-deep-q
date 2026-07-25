# :Relay: — ordered waves over one lane (operator's "Iwo Jima" doctrine), Merge 19

Never bet on a single combatant finishing the horizon end-to-end. Send LEGS: each resumes
from the furthest HELD position (the baton) with at most ONE diagnosed change of orders.
Not a smarter model — agile restarts. This is the reactive-schedule panel
(spec/schedules.spec.md) at the RUN timescale: move → epoch → run, one panel, three clocks.
Motivating evidence: the p7 monolithic run crowned 12.87 at epoch 2, sagged 4 epochs,
patience-stopped at ~7 epochs, pooled 885 — the beach was taken fast and the advance died.

## Definitions

- **Baton** = (ckpt `_best.pt`, confirmed bar). The furthest position HELD — crowns only
  move via the existing confirmation pooling (QLEARN_CONFIRM=1), never raw epochs.
- **Leg** = one qlearn.py run, RESUME=1 from the baton copy, `RELAY_LEG_EPOCHS` epochs
  (default 3 — sized so a leg lands < 15 min at measured cadence; policy bound),
  PATIENCE=99 (the DRIVER owns stopping; patience is repurposed as the leg boundary).
- **Pair** = tweak arm T + control arm C launched in PARALLEL from the same baton.
  C is a plain resume (no change). T carries the diagnosed order. Attribution rule:
  the tweak SURVIVES only if T's confirmed bar > C's. (PBT's counterfactual, serialized —
  best-practice deviation #1 and its mitigation.)

## Rule table (pre-registered; ONE knob per leg; default = NO CHANGE)

Signature read from the finished pair's epoch series + confirmed bars. BAND = 3.0
strength points (declared: binomial sd of the 48-game SF sample ≈ 3.3, tighter pooled).

| Signature | Test | Order for next T |
|---|---|---|
| ADVANCE | winner confirmed bar > baton bar | keep orders (reset knobs to base) |
| REGRESS | best epoch of leg < baton − BAND | discard leg weights; τ_start ×0.75 (floor ×0.5 base) |
| OSCILL  | ≥2 sign flips around baton bar AND spread > BAND | α ×0.5 (floor ×0.25 base) |
| STALL   | anything else (flat within band, no crown) | τ_start ×1.5 (cap ×4 base) |

Refutation learning: if C ≥ T on confirmed bar, the fired rule is logged REFUTED in
data/relay.md and its threshold raised (spec edit appended here, never silent).

## Contracts

- **Require**: a baton ckpt with `strength` metadata; the winning-protocol env (console
  parity) declared in relay.py in ONE dict.
- **Guarantee**: baton bar is monotone non-decreasing across legs (a losing pair never
  demotes the baton; REGRESS discards the leg, keeps the baton).
- **Maintain**: total epochs across all legs ≤ RELAY_BUDGET (default 30 — the declared
  horizon); every leg disposition ≤ 15 min wall; exactly one knob differs between T and C.
- **Assert**: every pair logs — signature, order, T bar, C bar, baton before/after — as
  numbered plain events + one scoreboard line (status-report format) in data/relay.md.

## Acceptance (before the relay becomes the default lane driver)

Full relay (rules ON) vs control-only relay (every leg plain-resume) over the same
epoch budget from the same baton: compare final confirmed bars. Tie within band → the
control wins by cost (the rule table dies by its own protocol — replicate-before-invent).

## Operator gates

Leg 1 fires on operator go. data/relay.stop halts the driver at the next leg boundary.

## Acceptance verdict (2026-07-12, ran same-day)

Five pairs, 30 epochs, from the 12.87 baton: EVERY pair tied at the baton (no arm crowned);
every tweak refuted by its control; rule table cooled/reheated tau across 0.498..1.181 with
no effect. Verdict: diagnosed restarts did NOT beat plain resume — control wins by cost —
AND nothing beat the baton at all. The 12.87 crown (monolith epoch 2) survived the
monolith tail + 30 relay epochs across 10 arms: the pst-769 linear ceiling is CONFIRMED
~12.9 at this protocol, consistent with the 4-way triangulated linear ceiling (11.2-13.9).
Rule-table defect logged: repeated REGRESS spirals tau downward (0.498) — fresh legs
always score noisy-low epochs vs a strong baton; any revival needs a REGRESS repeat-cap.
