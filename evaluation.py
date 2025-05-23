import numpy as np
import chess
import random
import threading
from board_utils import CACHE_LOCK, EVAL_CACHE

# Piece values for material evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King's value is not counted in material
}

# Maximum size for evaluation cache to avoid memory issues
MAX_CACHE_SIZE = 100000

def format_score(score):
    """Format a score properly without special symbols for checkmate"""
    # We'll use a consistent numerical format for all scores
    # Even for large values that might represent checkmate
    if score > 0:
        return f"+{score:.2f}"
    else:
        return f"{score:.2f}"


def fast_evaluate_position(board, ignore_checkmate=False):
    """
    Enhanced evaluation function that provides consistent, comparable numerical values
    even in checkmate/terminal positions.
    
    Args:
        board: The chess board to evaluate
        ignore_checkmate: If True, will calculate regular evaluation even in checkmate positions
    """
    # Check cache first
    board_hash = board.fen()
    
    with CACHE_LOCK:
        if board_hash in EVAL_CACHE:
            return EVAL_CACHE[board_hash]
    
    # Game over checks - BUT keep the raw evaluation instead of special values
    is_terminal = False
    terminal_eval = 0
    
    if board.is_checkmate() and not ignore_checkmate:
        # Instead of extreme ±10000, use a large but still comparable value
        terminal_eval = -50.0 if board.turn == chess.WHITE else 50.0
        is_terminal = True
    
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
        terminal_eval = 0.0  # Draw
        is_terminal = True
    
    if is_terminal:
        with CACHE_LOCK:
            EVAL_CACHE[board_hash] = terminal_eval
            if len(EVAL_CACHE) > MAX_CACHE_SIZE:
                EVAL_CACHE.clear()
        return terminal_eval
    
    # 1. Material balance
    white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                      for piece_type, value in PIECE_VALUES.items())
    black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                      for piece_type, value in PIECE_VALUES.items())
    material_score = white_material - black_material
        
    # 2. Mobility - count legal moves and threatened squares
    white_mobility = 0
    black_mobility = 0

    # Create board copies to calculate mobility for each side
    white_board = board.copy()
    white_board.turn = chess.WHITE
    white_mobility = len(list(white_board.legal_moves))

    black_board = board.copy()
    black_board.turn = chess.BLACK
    black_mobility = len(list(black_board.legal_moves))

    # Calculate attacked squares by both sides (no need to change turns)
    white_attacked = set()
    black_attacked = set()

    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_attacked.add(square)
        if board.is_attacked_by(chess.BLACK, square):
            black_attacked.add(square)
    
    mobility_score = (white_mobility - black_mobility) * 0.1
    control_score = (len(white_attacked) - len(black_attacked)) * 0.05
    
    # 3. King safety
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    white_king_attackers = 0
    black_king_attackers = 0
    
    if white_king_square:
        # Count attackers to white king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == chess.BLACK:
                if board.is_attacked_by(chess.BLACK, white_king_square):
                    white_king_attackers += 1
    
    if black_king_square:
        # Count attackers to black king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                if board.is_attacked_by(chess.WHITE, black_king_square):
                    black_king_attackers += 1
    
    # King safety score (negative means under attack)
    king_safety_score = (black_king_attackers - white_king_attackers) * 0.5
    
    # Check if kings are castled or in the center
    white_king_castled = white_king_square in [chess.G1, chess.C1]
    black_king_castled = black_king_square in [chess.G8, chess.C8]
    white_king_center = white_king_square in [chess.D4, chess.E4, chess.D5, chess.E5]
    black_king_center = black_king_square in [chess.D4, chess.E4, chess.D5, chess.E5]
    
    # Penalize kings in center, reward castled kings
    if white_king_castled:
        king_safety_score += 1
    if black_king_castled:
        king_safety_score -= 1
    if white_king_center:
        king_safety_score -= 2
    if black_king_center:
        king_safety_score += 2
    
    # 4. Pawn structure
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    
    # Count doubled pawns (same file)
    white_pawn_files = [chess.square_file(sq) for sq in white_pawns]
    black_pawn_files = [chess.square_file(sq) for sq in black_pawns]
    
    white_doubled = sum(white_pawn_files.count(f) - 1 for f in set(white_pawn_files))
    black_doubled = sum(black_pawn_files.count(f) - 1 for f in set(black_pawn_files))
    
    # Find isolated pawns (no friendly pawns on adjacent files)
    white_isolated = 0
    black_isolated = 0
    
    for pawn in white_pawns:
        file = chess.square_file(pawn)
        has_neighbor = False
        for adj_file in [file-1, file+1]:
            if 0 <= adj_file < 8:
                if any(chess.square_file(p) == adj_file for p in white_pawns):
                    has_neighbor = True
                    break
        if not has_neighbor:
            white_isolated += 1
    
    for pawn in black_pawns:
        file = chess.square_file(pawn)
        has_neighbor = False
        for adj_file in [file-1, file+1]:
            if 0 <= adj_file < 8:
                if any(chess.square_file(p) == adj_file for p in black_pawns):
                    has_neighbor = True
                    break
        if not has_neighbor:
            black_isolated += 1
    
    pawn_structure_score = (black_doubled - white_doubled) * 0.3 + (black_isolated - white_isolated) * 0.2
    
    # 5. Space control and central control
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    extended_center = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D6, 
                       chess.E3, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6]
    
    white_center = sum(1 for sq in center_squares if sq in white_attacked)
    black_center = sum(1 for sq in center_squares if sq in black_attacked)
    white_ext_center = sum(1 for sq in extended_center if sq in white_attacked)
    black_ext_center = sum(1 for sq in extended_center if sq in black_attacked)
    
    space_score = (white_center - black_center) * 0.5 + (white_ext_center - black_ext_center) * 0.1
    
    # 6. Piece coordination - count defended pieces and attackers per square
    white_defended = 0
    black_defended = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE and board.is_attacked_by(chess.WHITE, square):
                white_defended += 1
            elif piece.color == chess.BLACK and board.is_attacked_by(chess.BLACK, square):
                black_defended += 1
    
    coordination_score = (white_defended - black_defended) * 0.1
    
    # Check bonus
    check_score = 0
    if board.is_check():
        check_score = 50 if board.turn == chess.BLACK else -50
    
    # Combine all evaluation components
    eval_score = (
        material_score * 1.0 +        # Material value
        mobility_score * 0.3 +        # Freedom of movement
        control_score * 0.3 +         # Square control
        king_safety_score * 0.5 +     # King safety 
        pawn_structure_score * 0.4 +  # Pawn structure
        space_score * 0.3 +           # Space control
        coordination_score * 0.4 +    # Piece coordination
        check_score                   # Check bonus
    )
    
    # Cache the result
    with CACHE_LOCK:
        EVAL_CACHE[board_hash] = eval_score
        if len(EVAL_CACHE) > MAX_CACHE_SIZE:
            EVAL_CACHE.clear()
    
    return eval_score


def categorize_moves(board):
    """
    Categorize moves by tactical significance and assign sampling weights.
    Returns a dictionary of categorized moves and their weights.
    """
    moves = list(board.legal_moves)
    categorized_moves = {
        'captures_high_value': [],      # Weight: 10
        'checks': [],                   # Weight: 9
        'captures_equal_value': [],     # Weight: 8
        'threatened_piece_moves': [],   # Weight: 7
        'developing_moves': [],         # Weight: 6
        'captures_low_value': [],       # Weight: 5
        'center_control': [],           # Weight: 4
        'king_safety': [],              # Weight: 3
        'pawn_structure': [],           # Weight: 2
        'other_moves': []               # Weight: 1
    }
    
    # Weights for each category
    category_weights = {
        'captures_high_value': 10,
        'checks': 9,
        'captures_equal_value': 8,
        'threatened_piece_moves': 7,
        'developing_moves': 6,
        'captures_low_value': 5,
        'center_control': 4,
        'king_safety': 3,
        'pawn_structure': 2,
        'other_moves': 1
    }
    
    # Process each move and assign to appropriate category
    for move in moves:
        from_piece = board.piece_at(move.from_square)
        to_piece = board.piece_at(move.to_square)
        from_square_rank, from_square_file = divmod(move.from_square, 8)
        to_square_rank, to_square_file = divmod(move.to_square, 8)
        
        # Check if the move is a capture
        if to_piece is not None:
            # Capturing a higher value piece
            if PIECE_VALUES[to_piece.piece_type] > PIECE_VALUES[from_piece.piece_type]:
                categorized_moves['captures_high_value'].append(move)
                continue
            # Capturing an equal value piece
            elif PIECE_VALUES[to_piece.piece_type] == PIECE_VALUES[from_piece.piece_type]:
                categorized_moves['captures_equal_value'].append(move)
                continue
            # Capturing a lower value piece
            else:
                categorized_moves['captures_low_value'].append(move)
                continue
        
        # Check if the move gives check
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            categorized_moves['checks'].append(move)
            continue
        
        # Check if the piece is threatened and this move escapes
        if board.is_attacked_by(not board.turn, move.from_square):
            categorized_moves['threatened_piece_moves'].append(move)
            continue
        
        # Check if the move develops minor pieces (knights/bishops)
        if from_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # White knights and bishops on home squares
            if board.turn == chess.WHITE and from_square_rank == 0 and from_square_file in [1, 2, 5, 6]:
                categorized_moves['developing_moves'].append(move)
                continue
            # Black knights and bishops on home squares
            elif board.turn == chess.BLACK and from_square_rank == 7 and from_square_file in [1, 2, 5, 6]:
                categorized_moves['developing_moves'].append(move)
                continue
        
        # Center control moves (moves to or attacking e4, d4, e5, d5)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        if move.to_square in center_squares:
            categorized_moves['center_control'].append(move)
            continue
        
        # King safety moves (castling or moving king to safer positions)
        if from_piece.piece_type == chess.KING:
            # Castling moves
            if abs(from_square_file - to_square_file) > 1:
                categorized_moves['king_safety'].append(move)
                continue
            # King moving away from center
            if board.turn == chess.WHITE and from_square_rank == 0:
                categorized_moves['king_safety'].append(move)
                continue
            if board.turn == chess.BLACK and from_square_rank == 7:
                categorized_moves['king_safety'].append(move)
                continue
        
        # Pawn structure moves
        if from_piece.piece_type == chess.PAWN:
            # Pawn chain formation or advancement
            categorized_moves['pawn_structure'].append(move)
            continue
        
        # Other moves
        categorized_moves['other_moves'].append(move)
    
    return categorized_moves, category_weights


def select_weighted_moves(categorized_moves, category_weights, num_samples=25):
    # Flatten categorized moves with their weights
    weighted_moves = []
    for category, moves in categorized_moves.items():
        for move in moves:
            weighted_moves.append((move, category_weights[category]))
    
    # If not enough moves, return all
    if len(weighted_moves) <= num_samples:
        return [move for move, _ in weighted_moves]
    
    # Convert weights to probabilities
    total_weight = sum(weight for _, weight in weighted_moves)
    probabilities = [weight / total_weight for _, weight in weighted_moves]
    
    # This is the ONLY random sampling in the move selection process
    selected_indices = np.random.choice(
        len(weighted_moves), 
        size=num_samples,
        replace=False,
        p=probabilities
    )
    
    selected_moves = [weighted_moves[i][0] for i in selected_indices]
    return selected_moves



def find_threatened_squares(board):
    """Find all squares that are under threat/attack by the opponent"""
    threatened_squares = set()
    
    # Current player's color
    current_color = board.turn
    opponent_color = not current_color
    
    # Check all squares on the board
    for square in chess.SQUARES:
        # If the square is attacked by the opponent, it's threatened
        if board.is_attacked_by(opponent_color, square):
            threatened_squares.add(square)
    
    return threatened_squares

def find_guarded_squares(board):
    """Find all squares that are protected/controlled by friendly pieces"""
    guarded_squares = set()
    
    # Current player's color
    current_color = board.turn
    
    # Check all squares on the board
    for square in chess.SQUARES:
        # If the square is defended by friendly pieces, it's guarded
        if board.is_attacked_by(current_color, square):
            guarded_squares.add(square)
    
    return guarded_squares



def calculate_attack_range(board, color):
    """Calculate all squares a player of given color could attack in one move"""
    attack_squares = set()
    
    # For each square on the board
    for square in chess.SQUARES:
        # If the square is attacked by the specified color
        if board.is_attacked_by(color, square):
            attack_squares.add(square)
    
    return attack_squares

# REPLACE the calculate_movement_range function in evaluation.py with this:

def calculate_movement_range(board, color):
    """Calculate all squares a player of given color could move to in one move"""
    movement_squares = set()
    
    # Create a board copy and set the turn to the desired color
    temp_board = board.copy()
    temp_board.turn = color
    
    # For each legal move, add the destination square
    for move in temp_board.legal_moves:
        movement_squares.add(move.to_square)
    
    return movement_squares
def format_score(score):
    """Format a score properly without special symbols for checkmate"""
    # We'll use a consistent numerical format for all scores
    # Even for large values that might represent checkmate
    if score > 0:
        return f"+{score:.2f}"
    else:
        return f"{score:.2f}"
