---
description: 'Adapt the computer opponent to the human player by tracking move quality and tuning selection temperature'
import:
  - learned-model
  - search-mcts
---

***definitions***

- :Move-regret: is a move's quality on a position-independent scale: the learned value of the position after the played move minus the learned value after the policy's best move, taken in the mover's perspective; 0 is optimal and more-negative is worse.
- :Player-skill-level: is the exponentially-weighted MEAN of the human's :Move-regret: samples — a single moving estimate of how well the player is currently playing. (Variance/standard-deviation is deliberately not tracked: with one human there is exactly one skill to estimate, and a mean plus an additive offset is the whole signal.)
- :Strength-temperature: is the temperature applied when the computer picks its root move — 0 means argmax (strongest), higher means flatter sampling over visit counts (weaker); it is the single lever difficulty tuning moves. It is NOT the annealing :Prior-bias-temperature:, which softens prior move categories during expansion.
- :Difficulty-controller: is the per-game `entity` that observes both players' :Move-regret:, holds the :Player-skill-band: and the current :Strength-temperature:, and drives the opponent toward a target strength; it carries identity and mutable state across the moves of one game.

***implementation reqs***

- Constant: STRENGTH_TEMP_MIN / STRENGTH_TEMP_MAX — bounds of :Strength-temperature:; at or below MIN the computer plays argmax.
- Constant: DIFFICULTY_OFFSET — additive regret bias on the setpoint; positive makes the opponent play slightly better than the player (harder), negative gives a handicap. Replaces the earlier sigma-based offset.
- Constant: PLAYER_EMA_ALPHA — smoothing of the :Player-skill-level:.
- Constant: DIFFICULTY_GAIN — proportional gain of the temperature controller.
- Constant: USE_DYNAMIC_DIFFICULTY — feature flag, off by default; when off, root selection is plain argmax and no scoring runs.
- Regret reuses `get_q_value` (learned value) and one policy search for the best move; the computer's own search yields its best move for free, so only the human move costs an extra search.

***test reqs***

- A midgame position where the policy's best move and a clearly weaker legal move have distinct values, to assert regret ordering and temperature monotonicity.

***functional specs***

- Where :Difficulty-controller: is enabled, the computer must pick its root move using :Strength-temperature: rather than argmax.
  - Given :Strength-temperature: at or below STRENGTH_TEMP_MIN, When the root move is chosen, Then it is the argmax (identical to today).
  - Given a higher :Strength-temperature:, When the root move is chosen, Then weaker moves gain probability and mean :Move-regret: worsens. (Assert: expected regret is monotonically non-increasing in strength as temperature falls.)
- The :Difficulty-controller: must update the :Player-skill-band: from each human :Move-regret: and drive the opponent toward a setpoint.

  Input: player_move — the human's chosen move, and the position before it
  Parameters: offset ∈ ℝ (additive regret bias), gain ∈ ℝ⁺
  Initialize: player_mean ← seed_mean, temperature ← STRENGTH_TEMP_MAX/2   # per game; seed optional

  Loop over each move pair in the game:
      When it is the human's turn:
          r ← :Move-regret: of player_move
          player_mean ← EMA(player_mean, r)
      Otherwise When it is the computer's turn:
          setpoint ← player_mean + offset                              # transient
          pick the root move at the current temperature
          r_ai ← :Move-regret: of the computer's played move
          temperature ← clamp(temperature + gain · (r_ai − setpoint), STRENGTH_TEMP_MIN, STRENGTH_TEMP_MAX)
      log r or r_ai together with the raw mover-value for validation
  Assert: temperature stays within [STRENGTH_TEMP_MIN, STRENGTH_TEMP_MAX] every move.

  Given a stream of human moves at a steady skill, When several computer moves follow, Then the computer's mean :Move-regret: should converge near player_mean + offset.
  Given offset > 0, Then the computer should end slightly stronger than the player; Given offset < 0, Then weaker.

- The raw mover-value must be logged beside :Move-regret: each move so the regret signal can be validated without a second control path.
- :Difficulty-controller: state resets at game start but may be warm-started from seed_mean; cross-session persistence is out of scope here.
