"""Dynamic difficulty adjustment: track the player's skill and tune the opponent's strength.

See spec/dynamic-difficulty.spec.md. During a human-vs-computer game the controller scores
every move by its *regret* against the learned policy's best move (a position-independent
quality measure), maintains an EMA of the human's mean regret, and adjusts the computer's
root-selection temperature by proportional feedback so the opponent tracks a target of
`player_mean + offset` (offset > 0 = slightly stronger, < 0 = handicap).

Deliberately simple: with one human there is exactly one skill to estimate, so a mean plus an
additive offset is the whole signal -- no variance/standard-deviation. Regret is measured with
the learned value function, so this module never depends on the prior evaluator, and it is
orthogonal to the annealing schedule's prior-bias temperature.
"""

import chess

from constants import (
    USE_DYNAMIC_DIFFICULTY,
    STRENGTH_TEMP_MIN, STRENGTH_TEMP_MAX,
    DIFFICULTY_OFFSET, PLAYER_EMA_ALPHA, DIFFICULTY_GAIN,
)


def _mover_value(board_before, move, value_fn):
    """Learned value of the position after `move`, in the mover's perspective (higher = better).

    value_fn returns a White-absolute score in [-1, 1]; we flip sign for Black so that a larger
    number always means a better move for whoever is on the move.
    """
    board_after = board_before.copy()
    board_after.push(move)
    white_value = value_fn(board_after)
    return white_value if board_before.turn == chess.WHITE else -white_value


def move_regret(board_before, played_move, value_fn, best_move):
    """Return (regret, raw_value): quality of played_move vs the policy's best move.

    regret <= 0 means the played move is no better than best; 0 is optimal. raw_value is the
    mover-perspective value of the played move, logged for validation. Falls back to regret 0
    when best_move is missing or illegal.
    """
    raw_value = _mover_value(board_before, played_move, value_fn)
    if best_move is None or best_move not in board_before.legal_moves:
        return 0.0, raw_value
    best_value = _mover_value(board_before, best_move, value_fn)
    return raw_value - best_value, raw_value


class DifficultyController:
    """Per-game entity that adapts opponent strength to the human player's move quality."""

    def __init__(self, value_fn, best_move_fn, enabled=USE_DYNAMIC_DIFFICULTY,
                 mode="auto", fixed_temperature=0.5,
                 offset=DIFFICULTY_OFFSET, ema_alpha=PLAYER_EMA_ALPHA, gain=DIFFICULTY_GAIN,
                 temp_min=STRENGTH_TEMP_MIN, temp_max=STRENGTH_TEMP_MAX, seed_mean=0.0):
        # value_fn(board) -> White-absolute value; best_move_fn(board) -> policy's best move
        self.value_fn = value_fn
        self.best_move_fn = best_move_fn
        self.enabled = enabled
        self.mode = mode              # 'auto' = track the player; 'fixed' = constant temperature
        self.fixed_temperature = fixed_temperature
        self.offset = offset          # additive regret bias (>0 harder, <0 handicap)
        self.ema_alpha = ema_alpha
        self.gain = gain
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.seed_mean = seed_mean
        self.reset()

    def reset(self):
        """Reset per-game state; the skill level warm-starts from the configured seed."""
        self.player_mean = self.seed_mean
        self.temperature = (self.fixed_temperature if self.mode == "fixed"
                            else 0.5 * (self.temp_min + self.temp_max))
        self.log = []  # per-move records: {who, regret, raw, temp?, setpoint?}

    def setpoint(self):
        """Target regret for the computer: player's mean regret plus a fixed offset."""
        return self.player_mean + self.offset

    def observe_player_move(self, board_before, move):
        """Update the player's mean-regret skill level from the human's move."""
        if not self.enabled:
            return
        best = self.best_move_fn(board_before)
        regret, raw = move_regret(board_before, move, self.value_fn, best)
        a = self.ema_alpha
        self.player_mean = (1.0 - a) * self.player_mean + a * regret
        self.log.append({'who': 'human', 'regret': regret, 'raw': raw})

    def next_temperature(self):
        """Temperature the computer should use for its next root selection (0 = argmax)."""
        if not self.enabled:
            return 0.0
        if self.mode == "fixed":
            return self.fixed_temperature
        return self.temperature

    def observe_ai_move(self, board_before, played_move, best_move=None):
        """Feed back the computer's realized regret and nudge temperature toward the setpoint.

        best_move (the argmax root move) is supplied by the AI's own search to avoid a second
        search; if absent, it is recomputed via best_move_fn.
        """
        if not self.enabled:
            return
        if best_move is None:
            best_move = self.best_move_fn(board_before)
        regret, raw = move_regret(board_before, played_move, self.value_fn, best_move)
        target = self.setpoint()
        # Only the auto mode adapts; fixed mode holds the temperature the player set.
        if self.mode == "auto":
            # Higher temperature -> weaker play -> lower (more negative) regret; move toward target.
            self.temperature = min(self.temp_max,
                                   max(self.temp_min,
                                       self.temperature + self.gain * (regret - target)))
        self.log.append({'who': 'ai', 'regret': regret, 'raw': raw,
                         'temp': self.temperature, 'setpoint': target})
