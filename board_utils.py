import torch
import chess
import threading
import functools
from functools import lru_cache

# Shared cache for board evaluation with lock for thread safety
EVAL_CACHE = {}
CACHE_LOCK = threading.Lock()

# Board Representation & Fast Utility Functions
def board_to_tensor(board):
    """Convert a chess board to a compact tensor representation"""
    tensor = torch.zeros(12, 8, 8)
    
    # Mapping of piece types to channel indices
    piece_to_channel = {
        (chess.PAWN, True): 0,      # White pawn
        (chess.KNIGHT, True): 1,    # White knight
        (chess.BISHOP, True): 2,    # White bishop
        (chess.ROOK, True): 3,      # White rook
        (chess.QUEEN, True): 4,     # White queen
        (chess.KING, True): 5,      # White king
        (chess.PAWN, False): 6,     # Black pawn
        (chess.KNIGHT, False): 7,   # Black knight
        (chess.BISHOP, False): 8,   # Black bishop
        (chess.ROOK, False): 9,     # Black rook
        (chess.QUEEN, False): 10,   # Black queen
        (chess.KING, False): 11     # Black king
    }
    
    # Fill the tensor based on the board state
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row, col = divmod(square, 8)
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            tensor[channel, row, col] = 1.0
    
    return tensor

# Either use lru_cache everywhere OR manual locking, not both
@lru_cache(maxsize=5000)
def get_valid_moves_cached(board_fen):
    board = chess.Board(board_fen)
    return list(board.legal_moves)

def get_move_uci(move):
    """Convert a chess.Move to UCI format string (e.g., 'e2e4')"""
    return move.uci()

def get_legal_moves_from_square(board, square):
    """Get all legal moves from a specific square"""
    moves = set()
    for move in board.legal_moves:
        if move.from_square == square:
            moves.add(move.to_square)
    return moves

def get_secondary_moves(board, from_square):
    """Get potential squares that could be reached with two moves"""
    # This is computationally expensive, so limit it
    secondary_moves = set()
    primary_moves = get_legal_moves_from_square(board, from_square)
    
    # For each primary move, make the move and see what's possible next
    for to_square in primary_moves:
        # Create a copy of the board and make the move
        board_copy = board.copy()
        move = chess.Move(from_square, to_square)
        if move in board_copy.legal_moves:  # Safety check
            board_copy.push(move)
            
            # If it's a promotion move, skip (to avoid complexity)
            if move.promotion:
                continue
                
            # Find all squares this piece could move to next
            piece = board_copy.piece_at(to_square)
            if piece and piece.color == board_copy.turn:
                for second_move in board_copy.legal_moves:
                    if second_move.from_square == to_square:
                        secondary_moves.add(second_move.to_square)
    
    # Remove primary moves from secondary moves (they're already highlighted)
    secondary_moves -= primary_moves
    return secondary_moves

def square_name_to_square(square_name):
    """Convert square name (e.g., 'e4') to a square number"""
    try:
        file_char, rank_char = square_name.lower()
        file_idx = ord(file_char) - ord('a')
        rank_idx = int(rank_char) - 1
        
        if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
            return chess.square(file_idx, rank_idx)
        return None
    except:
        return None


# Game Playing Functions
def print_board(board):
    """Print the chess board with Unicode symbols"""
    board_str = str(board)
    piece_symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    
    for old, new in piece_symbols.items():
        board_str = board_str.replace(old, new)
    
    # Add rank numbers and file letters
    rows = board_str.split('\n')
    rows_with_numbers = []
    for i, row in enumerate(rows):
        rows_with_numbers.append(f"{8-i} {row}")
    board_str = '\n'.join(rows_with_numbers)
    board_str += '\n  a b c d e f g h'
    
    print(board_str)

