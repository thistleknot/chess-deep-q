import chess
import numpy as np
import torch
import random

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Constants
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# Colors for visual board
LIGHT_SQUARE_COLOR = '#f0d9b5'
DARK_SQUARE_COLOR = '#b58863'
HIGHLIGHT_COLOR = '#aaa23b'  # Yellow for selected pieces
POSSIBLE_MOVE_COLOR = '#7eb36a'  # Green for possible moves
SECONDARY_MOVE_COLOR = '#e2a853'  # Light orange for secondary moves
THREAT_COLOR = '#e88686'  # Light red for threats
GUARDED_COLOR = '#a8d5a2'  # Light green for guarded/protected pieces

MAX_CACHE_SIZE = 100000  # Limit cache size to avoid memory issues

# ---------------------------------------------------------------------------
# Reinforcement-learning constants (see spec/learned-model.spec.md)
# Developer-tuned: changing these changes what the program means.
# ---------------------------------------------------------------------------
GAMMA = 0.99                 # discount factor for the DQN target
LEARNING_RATE = 0.001        # Adam step size for the value network
BATCH_SIZE = 64              # replay minibatch size
REPLAY_CAPACITY = 10000      # replay buffer maxlen
TARGET_UPDATE_INTERVAL = 5   # sync target network every N games

# AHA (mistake-correction) learning — opt-in, off by default
AHA_BUDGET_PER_GAME = 3
AHA_THRESHOLD = -1.5

# ---------------------------------------------------------------------------
# Annealing schedule endpoints (see spec/annealing-schedule.spec.md)
# Every knob hands control from the fixed prior toward the learned model as
# training_progress goes 0 -> 1. Annealing means less prior control, not more
# randomness.
# ---------------------------------------------------------------------------
# Learned share of a search-leaf value (prior-heavy early, learned-heavy late)
LEAF_WEIGHT_START = 0.2
LEAF_WEIGHT_END = 1.0
# Multiplier on prior-derived shaped reward (decays; never below the floor)
SHAPED_WEIGHT_START = 1.0
SHAPED_WEIGHT_END = 0.2
SHAPED_WEIGHT_FLOOR = 0.1
# Softening of prior move-category weights before sampling (flatter late)
PRIOR_TEMP_START = 1.0
PRIOR_TEMP_END = 2.5
# MCTS UCB exploration weight (narrows with progress)
MCTS_EXPLORATION_START = 1.6
MCTS_EXPLORATION_END = 0.8
# Per-level Russian Doll sampling widths at progress 0 and progress 1
MCTS_SAMPLES_START = [25, 15, 10, 7, 5, 3, 2]
MCTS_SAMPLES_END = [21, 13, 8, 5, 3, 2, 1]

# ---------------------------------------------------------------------------
# Dynamic difficulty adjustment (see spec/dynamic-difficulty.spec.md)
# Opt-in; when disabled the computer plays argmax exactly as before.
# ---------------------------------------------------------------------------
USE_DYNAMIC_DIFFICULTY = False
STRENGTH_TEMP_MIN = 0.05     # at/below this the computer plays argmax (strongest)
STRENGTH_TEMP_MAX = 3.0      # flattest sampling over visit counts (weakest)
# Setpoint = player_mean_regret + DIFFICULTY_OFFSET. Additive (no variance/sigma): with one
# human there is exactly one skill to estimate, so mean + a fixed offset is the whole signal.
DIFFICULTY_OFFSET = 0.02     # >0 = opponent slightly stronger than player; <0 = handicap
PLAYER_EMA_ALPHA = 0.25      # smoothing of the player's mean-regret skill level
DIFFICULTY_GAIN = 0.5        # proportional gain of the temperature controller

# ---------------------------------------------------------------------------
# ELO calibration (see elo_calibration.py). Maps opponent temperature -> ELO and
# the aligned regret -> ELO, so the player can be scored on the same index as the policy.
# ---------------------------------------------------------------------------
# Kept small so calibration finishes in ~1-2 min interactively. Relative strength across
# temperatures is robust to shallow search, so calibration caps MCTS iterations.
ELO_CALIBRATION_TAUS = [0.0, 0.4, 0.8, 1.5]
ELO_CALIBRATION_GAMES = 6         # games per temperature (win-rate -> ELO gap)
ELO_CALIBRATION_MAX_MOVES = 60
ELO_CALIBRATION_ITERATIONS = 24   # capped MCTS iterations per move during calibration
ASSUMED_ARGMAX_ELO = 1200         # fallback full-strength anchor if none measured
ELO_CALIBRATION_PATH = "models/elo_calibration.json"
