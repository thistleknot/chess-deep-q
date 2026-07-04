"""Annealing schedule: hands control from the fixed prior to the learned model.

Single source of truth for every progress-driven coefficient in training and search.
See spec/annealing-schedule.spec.md. All endpoints live in constants.py; this module only
interpolates. The service is pure and stateless -- the same training_progress always yields
the same coefficients, so search and training read one consistent schedule.

Annealing here means *less prior control*, never *more randomness*: each knob moves behaviour
toward the learned policy as training_progress goes 0 -> 1.
"""

from constants import (
    LEAF_WEIGHT_START, LEAF_WEIGHT_END,
    SHAPED_WEIGHT_START, SHAPED_WEIGHT_END, SHAPED_WEIGHT_FLOOR,
    PRIOR_TEMP_START, PRIOR_TEMP_END,
    MCTS_EXPLORATION_START, MCTS_EXPLORATION_END,
    MCTS_SAMPLES_START, MCTS_SAMPLES_END,
)


def _clamp01(progress):
    """Clamp training progress into [0, 1]."""
    return max(0.0, min(1.0, progress))


def _lerp(start, end, progress):
    """Linear interpolate start -> end over clamped progress."""
    return start + (end - start) * _clamp01(progress)


class AnnealingSchedule:
    """Stateless service mapping training_progress in [0,1] to prior->learned coefficients."""

    @staticmethod
    def learned_leaf_weight(progress):
        """Share of a search-leaf value taken from the learned model; grows with progress."""
        return _lerp(LEAF_WEIGHT_START, LEAF_WEIGHT_END, progress)

    @staticmethod
    def shaped_reward_weight(progress):
        """Multiplier on prior-derived intermediate reward; decays, never below the floor."""
        return max(SHAPED_WEIGHT_FLOOR, _lerp(SHAPED_WEIGHT_START, SHAPED_WEIGHT_END, progress))

    @staticmethod
    def prior_bias_temperature(progress):
        """Softening of prior move-category weights before sampling; flattens with progress."""
        return _lerp(PRIOR_TEMP_START, PRIOR_TEMP_END, progress)

    @staticmethod
    def mcts_exploration_weight(progress):
        """UCB exploration weight for search; narrows with progress."""
        return _lerp(MCTS_EXPLORATION_START, MCTS_EXPLORATION_END, progress)

    @staticmethod
    def samples_per_level(progress):
        """Per-level Russian Doll sampling widths; narrow with progress.

        Each level interpolates between its start and end width; the end width acts as a floor.
        """
        return [
            max(end, int(round(_lerp(start, end, progress))))
            for start, end in zip(MCTS_SAMPLES_START, MCTS_SAMPLES_END)
        ]
