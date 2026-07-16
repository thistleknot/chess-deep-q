---
description: 'The temperature -> absolute-Elo dial: calibrate one trained net to any target strength against the anchor'
import:
  - elo-measurement
  - dynamic-difficulty
---

***definitions***

- :Temperature-elo-curve: is the persisted, monotone mapping from a policy's :Strength-temperature: to its absolute :Measured-elo: against the :Elo-anchor:. It is built per net checkpoint by measuring the policy at a grid of temperatures (argmax at the low end, flatter sampling at the high end) and fitting a monotone-decreasing curve — higher temperature = weaker = lower Elo.
- :Absolute-strength-dial: is the inverse lookup: given a target absolute Elo inside the curve's measured range, it returns the :Strength-temperature: whose measured Elo is closest (interpolated). This lets ONE trained net serve a chosen strength anywhere on the human rating curve (e.g. play a 1400 opponent) — an ABSOLUTE calibration, distinct from :Difficulty-controller:'s RELATIVE setpoint (player mean-regret + offset).
- :Chained-self-anchoring: is how the WEAK end of the curve is placed. A temperature that plays ~900 scores ≈0 against SF@1320 (the anchor's floor) — a shutout bound, not a point (per elo-measurement). So the directly-measurable range bottoms out well above the human range we want to serve. Fix: place mid-strength temperatures directly against the :Elo-anchor:, then place weaker temperatures by matches against the ALREADY-PLACED mid temperatures, chaining Elo downward off self-play rather than off the anchor.
- :Approximate-elo-curve: is a heuristic default :Temperature-elo-curve: present the instant a net is loaded, BEFORE any measurement — a smooth monotone placeholder (temperature → Elo gap below the anchor) that carries an `approximate` flag. It exists so the :Absolute-strength-dial: and the :Estimated-elo-readout: work with ZERO warm-up: a human can start an adjusting or fixed-strength game immediately and see estimated Elo. It is never claimed as measured — every surface that shows it labels it approximate — and it is replaced by the measured curve when (and only when) calibration runs (per the run-contract req below). Calibration is a distinct step from model TRAINING: training produces the net; calibration measures the temperature→Elo map of that net. Neither is a prerequisite for the other, and neither is a prerequisite for playing.
- :Estimated-elo-readout: is the per-turn surfacing of two numbers while a :Difficulty-controller: opponent (auto or fixed) is active: the opponent's Elo from its current :Strength-temperature:, and the human's Elo from their :Player-skill-level: regret (via `player_elo`). Both are read off whichever curve is active and are labeled approximate vs measured accordingly.

***implementation reqs***

- `elo_calibration.py` owns building, persisting, and inverting the curve; this file is its owning spec (previously absent). It reuses `measure_sf` / the elo-measurement machinery to place each grid point.
- Constant: TEMPERATURE_GRID, CALIBRATION_GAMES_PER_POINT — the temperature grid and games measured per point.
- The curve is persisted per checkpoint (e.g. `models/temp_elo.json`); it is re-measured ONLY on gate-clear checkpoints, not every run — a grid of ~8 temperatures × ~20 games does not fit the 5-minute run contract.
- The calibrator MUST expose an `approximate` flag and construct an :Approximate-elo-curve: as its default table, so a curve is always present without measurement. `calibrate()` clears the flag once it has measured the grid; loading a persisted curve restores whichever flag was saved.
- No implicit calibration: game start and difficulty setup MUST NOT trigger a measurement ("warm-up"). Running the real calibration is an explicit, opt-in action (a settings action), never a prompt gating normal play.

***test reqs***

- A grid with noisy raw measurements, to assert the persisted curve is projected monotone-decreasing before use.

***functional specs***

- :Temperature-elo-curve: must be monotone-decreasing in temperature.
  - Given temperatures t1 < t2, Then curve(t1) >= curve(t2); raw measurement noise is projected to the nearest monotone curve before persisting.
- Each point must be absolute Elo, not relative.
  - Given a grid temperature, When its point is measured, Then it is :Measured-elo: against the :Elo-anchor: (absolute), never a regret offset against a human.
- :Absolute-strength-dial: must invert within range and clamp (flagged) outside it.
  - Given a target Elo within the curve's range, Then the dial returns the interpolated :Strength-temperature:.
  - Given a target outside the range, Then it clamps to the nearest endpoint and flags the request as out-of-range (a shutout point is a bound, per elo-measurement).
- The weak end must be placed by :Chained-self-anchoring:, not against the anchor directly.
  - Given a temperature too weak to score against SF@1320 (a shutout bound), When it is placed, Then its Elo is derived from a match against an already-placed mid-strength temperature, chaining downward off self-play.
  - Given the curve extends below the anchor floor, Then it can serve human-range targets (e.g. 900) that direct anchor measurement cannot reach.
- Re-measurement must respect the run contract.
  - Given an ordinary ≤5-minute run, Then the curve is NOT re-measured; Given a gate-clear checkpoint, Then the full temperature grid is re-measured and persisted.
- The absolute dial must compose with, not replace, the relative controller.
  - Given the :Absolute-strength-dial: sets a baseline operating strength and the :Difficulty-controller: then tracks the seated human, Then the absolute curve fixes the operating point and the relative controller adjusts around it. (This is the temperature->Elo proxy the 1200 thread required.)
- The dial must be usable with no warm-up via the :Approximate-elo-curve:.
  - Given a net is loaded but never calibrated, When a human starts an adjusting or fixed-strength game, Then an :Approximate-elo-curve: is already present and the opponent's strength selection and the :Estimated-elo-readout: work immediately.
  - Given the active curve is approximate, When any Elo is shown, Then it is labeled approximate (never presented as measured).
  - Given the human declines or never runs calibration, Then play and the :Estimated-elo-readout: still function for the whole game (the warm-up is optional refinement, not a prerequisite).
- The :Estimated-elo-readout: must be surfaced each turn while an adjusting/fixed opponent is active.
  - Given a :Difficulty-controller: opponent (auto or fixed) is enabled, When a board is shown, Then the opponent's Elo (from :Strength-temperature:) and the human's Elo (from :Player-skill-level:) are displayed together, each labeled approximate or measured per the active curve.
  - Given the opponent is at full strength (:Difficulty-controller: disabled), Then no per-turn readout is required (there is no relative skill signal being tracked).

## Champion wiring (2026-07-14 — the :Strength-temperature: dial for the native engine)

- The champion (`agents.ChampionAgent`) exposes `root_move(board, tau)`: tau <= 0.02 ->
  full-depth deep-search argmax (d9, the 1670-claims engine); tau > 0.02 -> softmax over
  DIFF_DEPTH(=2)-backed child values, mover-perspective, UNNORMALIZED — so low tau only
  randomizes among near-equal moves (flat positions) and locks the best move in sharp
  ones (gate G1: regret monotone in tau, 0.058/0.604/0.621 cold/mid/hot on a
  hanging-queen battery; cold top-hit 38/40). Repetition-refusal per the pinned H2H
  cycle-lock lesson. Argmax is always reported (adapter caches `last_root_best` for
  :Move-regret:).
- Fixed :Absolute-strength-dial: tiers in the menu: ~800/1000/1183 via tau on the curve;
  1572/1670 = argmax at measured depths d7/d9 (claims-grade dial points).
- models/elo_calibration.json is accepted ONLY if its taus separate (the shipped file is
  a degenerate placeholder from a near-random old net); otherwise the
  :Approximate-elo-curve: default is used, anchored at the champion d2 band (~1183),
  flagged approximate/assumed in the readout per this spec. A measured tau-grid
  calibration vs SF@1320 is the follow-on that clears the flag.
