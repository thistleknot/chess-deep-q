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
