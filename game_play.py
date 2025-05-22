import chess
import time
import os
import datetime
import pickle
import threading
from board_utils import print_board, get_legal_moves_from_square, square_name_to_square
from evaluation import find_threatened_squares, find_guarded_squares, format_score, fast_evaluate_position
#from ui import ChessBoardVisualizer, NonClickableChessBoard



def get_human_move(board):
    """Get a move from human input"""
    legal_moves = [move.uci() for move in board.legal_moves]
    while True:
        try:
            uci = input("Enter your move in UCI format (e.g., e2e4): ")
            if uci == "help":
                print("Legal moves:", ", ".join(legal_moves))
                continue
            if uci == "resign":
                print("You resigned the game.")
                return chess.Move.null()
            
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except ValueError:
            print("Invalid format. Use UCI format (e.g., e2e4).")

def play_game(ai, human_color=chess.WHITE, display=True):
    """Play a game against the AI"""
    ai.reset_board()
    move_count = 0
    
    while not ai.board.is_game_over() and move_count < 200:
        if display:
            print_board(ai.board)
            print(f"Move: {move_count+1}")
        
        if ai.board.turn == human_color:
            # Human's turn
            move = get_human_move(ai.board)
        else:
            # AI's turn
            print("AI is thinking...")
            move = ai.get_best_move()
            print(f"AI plays: {get_move_uci(move)}")
        
        ai.make_move(move)
        move_count += 1
    
    # Game over
    if display:
        print_board(ai.board)
        print(f"Game over. Result: {ai.board.result()}")
        
        if ai.board.is_checkmate():
            winner = "White" if ai.board.turn == chess.BLACK else "Black"
            print(f"{winner} wins by checkmate!")
        elif ai.board.is_stalemate():
            print("Draw by stalemate.")
        elif ai.board.is_insufficient_material():
            print("Draw by insufficient material.")
        elif ai.board.is_repetition(3):
            print("Draw by threefold repetition.")
        else:
            print("Draw by fifty-move rule or other reason.")
            
    return ai.board.result()

def visual_play_game_with_features(ai, human_color=chess.WHITE):
    """Play a game against the AI with enhanced visual features"""
    ai.reset_board()
    move_count = 0
    last_move = None
    current_input = ""
    selected_square = None
    possible_moves = set()
    secondary_moves = set()
    
    plt.ion()  # Turn on interactive mode
    
    print("Enhanced Chess Game")
    print("------------------")
    print("Type the coordinates (e.g., e2e4) to make a move.")
    print("As you type, you'll see highlights for possible moves.")
    print("Press 'Escape' to cancel your current move entry.")
    print("Type 'resign' to resign the game.")
    print("Color coding:")
    print("- Light red: Squares under opponent control/attack")
    print("- Light green: Squares under your control/defense")
    print("- Yellow: Selected piece and last move")
    print("- Green: Possible moves for selected piece")
    print("- Light orange: Secondary moves (two-move sequences)")
    print("- Pieces are highlighted more brightly when under threat or defended")
    print("- Contested squares show both colors")
    
    while not ai.board.is_game_over() and move_count < 200:
        # Get evaluation score
        evaluation = fast_evaluate_position(ai.board)
        # Flip the sign for displaying from black's perspective
        display_eval = evaluation if human_color == chess.WHITE else -evaluation
        
        # Find threatened and guarded squares from the player's perspective
        # Always calculate for the current player's turn, not necessarily the human
        threatened_squares = find_threatened_squares(ai.board)
        guarded_squares = find_guarded_squares(ai.board)
        
        # Display the board with current features
        fig, ax = create_visual_chess_board(
            ai.board, 
            last_move=last_move,
            threatened_squares=threatened_squares,
            guarded_squares=guarded_squares,
            selected_square=selected_square,
            possible_moves=possible_moves,
            secondary_moves=secondary_moves,
            evaluation=display_eval
        )
        plt.draw()
        plt.pause(0.1)
        
        print(f"Move: {move_count+1}")
        
        if ai.board.turn == human_color:
            # Human's turn - enhanced input with visual feedback
            print(f"Current input: {current_input}")
            key = input("Enter next key or complete move: ")
            
            if key.lower() == 'resign':
                print("You resigned the game.")
                break
            
            # Handle keyboard input
            if key.lower() == 'esc' or key.lower() == 'escape':
                # Reset the input
                current_input = ""
                selected_square = None
                possible_moves = set()
                secondary_moves = set()
                print("Input cleared.")
            else:
                # Add the key to current input
                current_input += key
                
                # Process the current input
                result, new_possible_moves, new_secondary_moves, current_input = enhanced_human_move(
                    ai.board, current_input
                )
                
                if isinstance(result, chess.Move):
                    # Complete move entered
                    move = result
                    ai.make_move(move)
                    last_move = move
                    move_count += 1
                    selected_square = None
                    possible_moves = set()
                    secondary_moves = set()
                    print(f"You played: {move.uci()}")
                elif isinstance(result, int):
                    # Square selected, update highlights
                    selected_square = result
                    possible_moves = new_possible_moves
                    secondary_moves = new_secondary_moves
        else:
            # AI's turn
            print("AI is thinking...")
            start_time = time.time()
            move = ai.get_best_move()
            end_time = time.time()
            ai.make_move(move)
            last_move = move
            move_count += 1
            print(f"AI plays: {move.uci()} (took {end_time - start_time:.2f}s)")
        
        # Close the figure before showing the next one
        plt.close(fig)
    
    # Show the final position
    evaluation = fast_evaluate_position(ai.board)
    display_eval = evaluation if human_color == chess.WHITE else -evaluation
    threatened_squares = find_threatened_squares(ai.board)
    guarded_squares = find_guarded_squares(ai.board)
    
    fig, ax = create_visual_chess_board(
        ai.board, 
        last_move=last_move,
        threatened_squares=threatened_squares,
        guarded_squares=guarded_squares,
        evaluation=display_eval
    )
    plt.draw()
    
    # Game over
    print(f"Game over. Result: {ai.board.result()}")
    
    if ai.board.is_checkmate():
        winner = "White" if ai.board.turn == chess.BLACK else "Black"
        print(f"{winner} wins by checkmate!")
    elif ai.board.is_stalemate():
        print("Draw by stalemate.")
    elif ai.board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif ai.board.is_repetition(3):
        print("Draw by threefold repetition.")
    else:
        print("Draw by fifty-move rule or other reason.")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final board displayed
    
    return ai.board.result()

def visual_play_game(ai, human_color=chess.WHITE):
    """Play a game against the AI with visual board"""
    # Create a non-clickable board (input through CLI)
    board = NonClickableChessBoard(chess.Board(), ai, human_color=human_color)
    board.start()
    return ai.board.result()


# Save and Load Game Functions
def save_game_to_pgn(board, move_history, filename="saved_game.pgn", player_name="Human", ai_name="Chess AI"):
    """Save the current game to a PGN file"""
    # Create a new game
    game = chess.pgn.Game()
    
    # Set headers
    game.headers["Event"] = "Chess Game"
    game.headers["Site"] = "Enhanced Chess UI"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = player_name if board.turn == chess.BLACK else ai_name
    game.headers["Black"] = ai_name if board.turn == chess.BLACK else player_name
    game.headers["Result"] = board.result()
    
    # Reconstruct the game from move history
    node = game
    for move in move_history:
        node = node.add_variation(chess.Move.from_uci(move))
    
    # Create the save directory if it doesn't exist
    save_dir = "saved_games"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save to file
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    
    # Also save the board state for direct loading
    state_filename = os.path.splitext(filename)[0] + ".state"
    state_path = os.path.join(save_dir, state_filename)
    with open(state_path, "wb") as f:
        state = {
            "fen": board.fen(),
            "moves": move_history
        }
        pickle.dump(state, f)
    
    return save_path

def load_game_from_pgn(filename="saved_game.pgn"):
    """Load a game from a PGN file"""
    # Construct the path
    save_dir = "saved_games"
    path = os.path.join(save_dir, filename)
    
    # Try to load the state file first (more reliable)
    state_filename = os.path.splitext(filename)[0] + ".state"
    state_path = os.path.join(save_dir, state_filename)
    
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            board = chess.Board(state["fen"])
            move_history = state["moves"]
            return board, move_history
    
    # If state file doesn't exist, try loading from PGN
    if os.path.exists(path):
        with open(path) as f:
            game = chess.pgn.read_game(f)
            
        # Recreate the board and moves
        board = game.board()
        moves = []
        
        # Traverse the mainline moves
        for move in game.mainline_moves():
            moves.append(move.uci())
            board.push(move)
            
        return board, moves
    
    return None, None

def list_saved_games():
    """List all saved games"""
    save_dir = "saved_games"
    if not os.path.exists(save_dir):
        return []
    
    pgn_files = [f for f in os.listdir(save_dir) if f.endswith(".pgn")]
    return pgn_files


# Save and Load Game Functions
def save_game_to_pgn(board, move_history, filename="saved_game.pgn", player_name="Human", ai_name="Chess AI"):
    """Save the current game to a PGN file"""
    # Create a new game
    game = chess.pgn.Game()
    
    # Set headers
    game.headers["Event"] = "Chess Game"
    game.headers["Site"] = "Enhanced Chess UI"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = player_name if board.turn == chess.BLACK else ai_name
    game.headers["Black"] = ai_name if board.turn == chess.BLACK else player_name
    game.headers["Result"] = board.result()
    
    # Reconstruct the game from move history
    node = game
    for move in move_history:
        node = node.add_variation(chess.Move.from_uci(move))
    
    # Create the save directory if it doesn't exist
    save_dir = "saved_games"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save to file
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    
    # Also save the board state for direct loading
    state_filename = os.path.splitext(filename)[0] + ".state"
    state_path = os.path.join(save_dir, state_filename)
    with open(state_path, "wb") as f:
        state = {
            "fen": board.fen(),
            "moves": move_history
        }
        pickle.dump(state, f)
    
    return save_path

def load_game_from_pgn(filename="saved_game.pgn"):
    """Load a game from a PGN file"""
    # Construct the path
    save_dir = "saved_games"
    path = os.path.join(save_dir, filename)
    
    # Try to load the state file first (more reliable)
    state_filename = os.path.splitext(filename)[0] + ".state"
    state_path = os.path.join(save_dir, state_filename)
    
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            board = chess.Board(state["fen"])
            move_history = state["moves"]
            return board, move_history
    
    # If state file doesn't exist, try loading from PGN
    if os.path.exists(path):
        with open(path) as f:
            game = chess.pgn.read_game(f)
            
        # Recreate the board and moves
        board = game.board()
        moves = []
        
        # Traverse the mainline moves
        for move in game.mainline_moves():
            moves.append(move.uci())
            board.push(move)
            
        return board, moves
    
    return None, None

def list_saved_games():
    """List all saved games"""
    save_dir = "saved_games"
    if not os.path.exists(save_dir):
        return []
    
    pgn_files = [f for f in os.listdir(save_dir) if f.endswith(".pgn")]
    return pgn_files
