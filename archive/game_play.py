def enhanced_human_move(board, current_input=""):
    """
    Enhanced move input function that shows possible moves as user types
    Returns either a complete move or current_input for further processing
    """
    # Process the current input
    if len(current_input) == 2:  # First square entered (e.g., "e2")
        from_square = square_name_to_square(current_input)
        if from_square is not None and board.piece_at(from_square):
            if board.piece_at(from_square).color == board.turn:
                possible_moves = get_legal_moves_from_square(board, from_square)
                secondary_moves = get_secondary_moves(board, from_square)
                return from_square, possible_moves, secondary_moves, current_input
    
    elif len(current_input) == 4:  # Complete move entered (e.g., "e2e4")
        try:
            move = chess.Move.from_uci(current_input)
            if move in board.legal_moves:
                return move, set(), set(), ""  # Return the move and reset input
        except ValueError:
            pass  # Invalid move format
    
    # If we get here, the input is incomplete or invalid
    return None, set(), set(), current_input

