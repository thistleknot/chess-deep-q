import chess
import colorama
import os
import time
from colorama import Fore, Back, Style
from evaluation import (fast_evaluate_position, find_threatened_squares,
                        find_guarded_squares, evaluate_by_player)
from board_utils import get_legal_moves_from_square, get_secondary_moves, get_move_uci
from difficulty import DifficultyController

# Initialize colorama for cross-platform terminal colors
colorama.init()

class TerminalChessBoard:
    def __init__(self, board, ai, human_color=chess.WHITE):
        self.board = board
        self.ai = ai
        self.human_color = human_color
        self.selected_square = None
        self.possible_moves = set()
        self.secondary_moves = set()
        self.last_move = None
        self.move_history = []
        self.board_history = [board.copy()]
        self.highlighted_hint = None

        # Dynamic difficulty controller (off unless enabled). Scores moves with the learned
        # policy and adjusts the AI's move-selection temperature toward the player's skill band.
        self.difficulty_controller = DifficultyController(
            value_fn=self._policy_value,
            best_move_fn=self._policy_best_move,
        )
        # Apply any difficulty settings the user chose in the menu (persisted on the AI).
        settings = getattr(ai, 'difficulty_settings', None)
        if settings:
            self.difficulty_controller.enabled = settings.get('enabled', self.difficulty_controller.enabled)
            self.difficulty_controller.mode = settings.get('mode', self.difficulty_controller.mode)
            self.difficulty_controller.fixed_temperature = settings.get(
                'fixed_temperature', self.difficulty_controller.fixed_temperature)
            self.difficulty_controller.offset = settings.get('offset', self.difficulty_controller.offset)
            self.difficulty_controller.offset_sdev = settings.get(
                'offset_sdev', self.difficulty_controller.offset_sdev)   # :Sigma-offset: dial
            self.difficulty_controller.reset()   # apply mode (fixed sets the starting temperature)

        # ELO calibrator (temperature/regret -> ELO), if one was loaded onto the AI.
        self.elo_calibrator = getattr(ai, 'elo_calibrator', None)

    def _difficulty_status_line(self):
        """One-line ELO readout while dynamic difficulty is active (estimates)."""
        ctrl = self.difficulty_controller
        cal = self.elo_calibrator
        if not ctrl.enabled or cal is None or not cal.is_calibrated():
            return None
        opp = cal.policy_elo(ctrl.temperature)
        you = cal.player_elo(ctrl.player_mean)
        anchor = "measured" if cal.anchor_measured else "assumed"
        approx = "approx curve" if getattr(cal, "approximate", False) else "calibrated"
        parts = []
        if opp is not None:
            parts.append(f"Opponent ≈ {opp:.0f} ELO")
        if you is not None:
            parts.append(f"your play ≈ {you:.0f} ELO")
        if not parts:
            return None
        return f"[difficulty] {' | '.join(parts)} ({approx}; anchor {anchor})"

    def _policy_value(self, board):
        """Learned value (White-absolute, [-1,1]) of a position; feeds difficulty regret."""
        return self.ai.dqn_agent.get_q_value(board)

    def _policy_best_move(self, board):
        """The policy's strongest (argmax) move for a position, used to score a player's move."""
        saved_board = self.ai.board
        self.ai.board = board.copy()
        try:
            return self.ai.get_best_move()  # temperature 0 => argmax
        finally:
            self.ai.board = saved_board
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    def prompt_load_game(self):
        """Prompt the user to select a saved game"""
        # Import the list_saved_games function if not already imported
        from game_play import list_saved_games, load_game_from_pgn
        
        games = list_saved_games()
        if not games:
            print("No saved games found.")
            return
            
        print("Available saved games:")
        for i, game in enumerate(games):
            print(f"{i+1}. {game}")
            
        try:
            choice = int(input("Enter the number of the game to load (0 to cancel): "))
            if 1 <= choice <= len(games):
                self.load_game(games[choice-1])
                return True
            elif choice == 0:
                return False
            else:
                print("Invalid choice.")
                return False
        except ValueError:
            print("Invalid choice. Load cancelled.")
            return False

    def load_game(self, filename="game.pgn"):
        """Load a saved game"""
        # Import the load_game_from_pgn function if not already imported
        from game_play import load_game_from_pgn
        
        board, moves = load_game_from_pgn(filename)
        if board and moves:
            self.board = board
            self.move_history = moves
            
            # Reconstruct board history
            self.board_history = [chess.Board()]
            temp_board = chess.Board()
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                temp_board.push(move)
                self.board_history.append(temp_board.copy())
            
            # Set last move if available
            if moves:
                last_move_uci = moves[-1]
                self.last_move = chess.Move.from_uci(last_move_uci)
            else:
                self.last_move = None
                
            self.selected_square = None
            self.possible_moves = set()
            self.secondary_moves = set()
            
            # Update the AI's board
            self.ai.board = board.copy()
            
            self.clear_screen()
            self.display_board()
            print(f"Game loaded from {filename}")
            return True
        else:
            print(f"Failed to load game from {filename}")
            return False

    def save_game(self, filename=None):
        """Save the current game"""
        # Import the save_game_to_pgn function if not already imported
        from game_play import save_game_to_pgn
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"chess_game_{timestamp}.pgn"
        
        path = save_game_to_pgn(self.board, self.move_history, filename)
        print(f"Game saved to {path}")
        return path
    def display_board(self):
        """Display the chess board with colored squares and pieces"""
        # Get evaluation score
        evaluation = fast_evaluate_position(self.board)
        
        # Find threatened and guarded squares
        threatened_squares = find_threatened_squares(self.board)
        guarded_squares = find_guarded_squares(self.board)
        
        # Find contested squares (both threatened and guarded)
        contested_squares = threatened_squares & guarded_squares
        
        # Print turn and evaluation at the top
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        eval_prefix = "+" if evaluation > 0 else ""
        
        print(f"\n{turn} to move | Eval: {eval_prefix}{evaluation:.2f}")

        # Per-player breakdown: piece (material) value and board-position value.
        scores = evaluate_by_player(self.board)
        w, b = scores['white'], scores['black']
        print(f"  {'':7}{'Pieces':>8} {'Position':>9} {'Total':>8}")
        print(f"  {'White':7}{w['material']:>8.1f} {w['position']:>9.2f} {w['total']:>8.2f}")
        print(f"  {'Black':7}{b['material']:>8.1f} {b['position']:>9.2f} {b['total']:>8.2f}")

        # Estimated ELO of the opponent and of your play, shown every turn when an
        # adjusting/fixed opponent is active (uses the approximate curve until calibrated).
        status = self._difficulty_status_line()
        if status:
            print(status)
        print("  " + "-" * 17)  # Adjusted for proper alignment
        
        # Unicode chess pieces
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
            '.': ' '
        }
        
        # Display board - with white at the bottom
        for rank in range(7, -1, -1):
            print(f"{rank+1} |", end="")
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                square_symbol = piece_symbols[piece.symbol()] if piece else piece_symbols['.']
                
                # Determine background color based on square type and highlights
                bg_color = Back.LIGHTBLACK_EX if (file + rank) % 2 == 1 else Back.BLACK
                
                # Priority-based highlighting (most important first)
                if self.selected_square == square:
                    bg_color = Back.YELLOW
                elif square in contested_squares:
                    bg_color = Back.LIGHTYELLOW_EX
                elif square in threatened_squares:
                    bg_color = Back.RED
                elif square in self.possible_moves:
                    bg_color = Back.GREEN
                elif square in self.secondary_moves:
                    bg_color = Back.MAGENTA
                elif self.last_move and (square == self.last_move.from_square or square == self.last_move.to_square):
                    bg_color = Back.BLUE
                elif square in guarded_squares:
                    bg_color = Back.CYAN
                
                # Determine text color based on piece color
                if piece:
                    text_color = Fore.WHITE if piece.color == chess.WHITE else Fore.BLACK
                    # For better contrast on certain backgrounds
                    if bg_color in [Back.YELLOW, Back.LIGHTYELLOW_EX, Back.CYAN]:
                        text_color = Fore.BLACK
                else:
                    text_color = Fore.WHITE
                
                # Print the square (piece + space for alignment)
                print(f"{bg_color}{text_color}{square_symbol} {Style.RESET_ALL}", end="")
            print("|")
        
        print("  " + "-" * 17)  # Adjusted for proper alignment
        print("   a b c d e f g h")  # Properly aligned coordinates
        
        # Enhanced Legend
        print("\nHighlight Legend:")
        print(f"{Back.YELLOW}   {Style.RESET_ALL} Selected piece     ", end="")
        print(f"{Back.GREEN}   {Style.RESET_ALL} Possible moves     ", end="")
        print(f"{Back.MAGENTA}   {Style.RESET_ALL} Secondary moves")
        print(f"{Back.BLUE}   {Style.RESET_ALL} Last move          ", end="")
        print(f"{Back.LIGHTYELLOW_EX}   {Style.RESET_ALL} Contested (T+G)    ", end="")
        print(f"{Back.RED}   {Style.RESET_ALL} Threatened only")
        print(f"{Back.CYAN}   {Style.RESET_ALL} Guarded only")
        
        # Available commands (single-letter shortcuts in brackets)
        print("\nCommands: '(h)int', '(u)ndo', '(s)ave', '(l)oad', '(r)esign', '(c)ancel', "
              "or enter move (e.g., e2e4)")
        
        # Show current selection info if any
        if self.selected_square is not None:
            square_name = chess.square_name(self.selected_square)
            piece = self.board.piece_at(self.selected_square)
            piece_name = piece.symbol().upper() if piece else "Empty"
            print(f"\nSelected: {square_name} ({piece_name})")
            
            if self.possible_moves:
                move_names = [chess.square_name(sq) for sq in sorted(self.possible_moves)]
                print(f"Possible moves: {', '.join(move_names)}")
                
            if self.secondary_moves:
                secondary_names = [chess.square_name(sq) for sq in sorted(self.secondary_moves)]
                print(f"Secondary moves: {', '.join(secondary_names)}")
    
    def enhanced_human_move(self, current_input=""):
        """
        Enhanced move input function that shows possible moves as user types
        This is the ASCII version of the matplotlib enhanced_human_move
        """
        if len(current_input) == 2:  # First square entered (e.g., "e2")
            from_square = self.square_name_to_square(current_input)
            if from_square is not None and self.board.piece_at(from_square):
                piece = self.board.piece_at(from_square)
                if piece.color == self.board.turn:
                    # Update the display state
                    self.selected_square = from_square
                    self.possible_moves = get_legal_moves_from_square(self.board, from_square)
                    self.secondary_moves = get_secondary_moves(self.board, from_square)
                    
                    # Refresh the display
                    self.clear_screen()
                    self.display_board()
                    
                    return from_square, self.possible_moves, self.secondary_moves, current_input
        
        elif len(current_input) == 4:  # Complete move entered (e.g., "e2e4")
            try:
                move = chess.Move.from_uci(current_input)
                if move in self.board.legal_moves:
                    return move, set(), set(), ""  # Return the move and reset input
            except ValueError:
                pass  # Invalid move format
        
        # If we get here, the input is incomplete or invalid
        return None, set(), set(), current_input
    
    def square_name_to_square(self, square_name):
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
    
    def process_input(self, user_input):
        """Process user input for a move or command with enhanced features"""
        # Split input to handle both "e2" and "e2e4" formats
        parts = user_input.lower().strip().split()
        command = parts[0] if parts else ""
        
        # Handle commands. Each accepts its full word or its first-letter shortcut
        # (h/u/s/l/r/c); single letters never collide with moves, which are 2 or 4 chars.
        if command in ('hint', 'h'):
            self.show_move_hint()
            return False
        elif command in ('undo', 'u'):
            self.undo_move()
            return False
        elif command in ('save', 's'):
            self.save_game()
            return False
        elif command in ('load', 'l'):
            self.prompt_load_game()
            return False
        elif command in ('resign', 'r'):
            print("You resigned the game.")
            return True  # Signal game over
        elif command in ('cancel', 'c'):
            self.selected_square = None
            self.possible_moves = set()
            self.secondary_moves = set()
            self.clear_screen()
            self.display_board()
            return False
        
        # Use enhanced move processing
        result, new_possible_moves, new_secondary_moves, updated_input = self.enhanced_human_move(command)
        
        if isinstance(result, chess.Move):
            # Complete move entered
            move = result
            
            # Save the board state before making the move
            self.board_history.append(self.board.copy())
            
            # Make the move
            self.board.push(move)
            self.last_move = move
            self.move_history.append(move.uci())
            
            # Reset selection state
            self.selected_square = None
            self.possible_moves = set()
            self.secondary_moves = set()
            self.highlighted_hint = None
            
            return True  # Move successfully made
            
        elif isinstance(result, int):
            # Square selected, highlights already updated by enhanced_human_move
            return False
            
        elif result is None and len(command) == 2:
            # Potential destination after selecting a piece
            if self.selected_square is not None:
                dest_square = self.square_name_to_square(command)
                if dest_square is not None and dest_square in self.possible_moves:
                    move = chess.Move(self.selected_square, dest_square)
                    
                    # Check for promotion
                    piece = self.board.piece_at(self.selected_square)
                    if piece.piece_type == chess.PAWN:
                        if (self.human_color == chess.WHITE and chess.square_rank(dest_square) == 7) or \
                           (self.human_color == chess.BLACK and chess.square_rank(dest_square) == 0):
                            valid_promotions = {'q': chess.QUEEN, 'r': chess.ROOK, 
                                              'b': chess.BISHOP, 'n': chess.KNIGHT}
                            promotion = input("Promote to (q/r/b/n, default=q): ").lower() or 'q'
                            if promotion in valid_promotions:
                                move.promotion = valid_promotions[promotion]
                            else:
                                move.promotion = chess.QUEEN
                    
                    # Save board state and make move
                    self.board_history.append(self.board.copy())
                    self.board.push(move)
                    self.last_move = move
                    self.move_history.append(move.uci())
                    
                    # Reset selection state
                    self.selected_square = None
                    self.possible_moves = set()
                    self.secondary_moves = set()
                    self.highlighted_hint = None
                    
                    return True  # Move successfully made
                else:
                    print(f"Invalid destination. {command} is not a legal move.")
                    return False
        
        # Handle complete move format (e.g., "e2e4")
        if len(command) == 4:
            try:
                move = chess.Move.from_uci(command)
                if move in self.board.legal_moves:
                    # Save the board state before making the move
                    self.board_history.append(self.board.copy())
                    
                    # Make the move
                    self.board.push(move)
                    self.last_move = move
                    self.move_history.append(move.uci())
                    
                    # Reset selection state
                    self.selected_square = None
                    self.possible_moves = set()
                    self.secondary_moves = set()
                    self.highlighted_hint = None
                    
                    return True  # Move successfully made
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid move format. Use format like 'e2e4'.")
            
            return False
        
        print("Invalid input. Enter a coordinate like 'e2' to select a piece, or 'e2e4' to make a move.")
        return False
        
    def make_ai_move(self):
        """Let the AI make a move"""
        if self.board.is_game_over():
            return
            
        if self.board.turn != self.human_color:
            print("AI is thinking...")
            start_time = time.time()

            # Difficulty: pick the strength temperature for this move (0 = strongest/argmax).
            temperature = self.difficulty_controller.next_temperature()

            # Get AI's move
            state_before = self.board.copy()
            self.ai.board = state_before.copy()  # Make sure AI has current board
            move = self.ai.get_best_move(temperature=temperature)

            end_time = time.time()
            print(f"AI plays: {move.uci()} (took {end_time - start_time:.2f}s)")

            # Feed the AI's realized regret back to the controller (best move came free from the
            # same search via last_root_best), nudging temperature toward the player's band.
            self.difficulty_controller.observe_ai_move(
                state_before, move, self.ai.dqn_agent.last_root_best)

            # (The estimated-ELO readout is rendered every turn in display_board's header.)

            # Save board state before AI move
            self.board_history.append(self.board.copy())

            # Update the board
            self.board.push(move)
            self.last_move = move
            self.move_history.append(move.uci())
        
    def undo_move(self):
        """Undo the last move pair (human + AI) with improved state management"""
        # Need at least one move to undo
        if len(self.move_history) == 0:
            print("No moves to undo.")
            return False
        
        # Determine how many moves to undo based on whose turn it is
        if self.board.turn == self.human_color:
            # It's human's turn, so AI just moved
            # We want to undo both the AI's move and the human's move before that
            if len(self.move_history) >= 2:
                moves_to_undo = 2
                target_move_count = len(self.move_history) - 2
            else:
                print("Not enough moves to undo a complete turn pair.")
                return False
        else:
            # It's AI's turn, so human just moved
            # We only want to undo the human's move
            moves_to_undo = 1
            target_move_count = len(self.move_history) - 1
        
        # Remove the moves from history
        for _ in range(moves_to_undo):
            if self.move_history:
                self.move_history.pop()
        
        # Reconstruct the board from the beginning with remaining moves
        self.board = chess.Board()  # Start fresh
        
        # Replay all remaining moves
        for move_uci in self.move_history:
            try:
                move = chess.Move.from_uci(move_uci)
                self.board.push(move)
            except ValueError:
                print(f"Error replaying move: {move_uci}")
                # Fallback: reset to initial position
                self.board = chess.Board()
                self.move_history = []
                break
        
        # Rebuild board history to match current state
        self.board_history = [chess.Board()]  # Start with initial position
        temp_board = chess.Board()
        for move_uci in self.move_history:
            move = chess.Move.from_uci(move_uci)
            temp_board.push(move)
            self.board_history.append(temp_board.copy())
        
        # Set last move if there are any moves left
        if self.move_history:
            self.last_move = chess.Move.from_uci(self.move_history[-1])
        else:
            self.last_move = None
        
        # Reset selection state
        self.selected_square = None
        self.possible_moves = set()
        self.secondary_moves = set()
        
        # Update the AI's board to match current state
        self.ai.board = self.board.copy()
        
        # Provide feedback
        if moves_to_undo == 2:
            print("Undid last move pair (human + AI).")
        else:
            print("Undid last move.")
        
        return True
    
    def show_move_hint(self):
        """Get a hint for the best move from the AI"""
        if self.board.turn == self.human_color and not self.board.is_game_over():
            # Temporarily make a copy of the board to not affect the game state
            board_copy = self.board.copy()
            self.ai.board = board_copy
            
            # Get AI's move suggestion
            hint_move = self.ai.get_best_move()
            self.ai.board = self.board  # Restore the original board
            
            # Get the move notation
            from_square_name = chess.square_name(hint_move.from_square)
            to_square_name = chess.square_name(hint_move.to_square)
            
            print(f"Hint: Move {from_square_name} to {to_square_name}")
            
            # Store the hint and highlight it
            self.highlighted_hint = (hint_move.from_square, hint_move.to_square)
            
            # Temporarily highlight the hint
            temp_selected = self.selected_square
            temp_possible = self.possible_moves.copy()
            temp_secondary = self.secondary_moves.copy()
            
            self.selected_square = hint_move.from_square
            self.possible_moves = {hint_move.to_square}
            self.secondary_moves = set()
            
            self.clear_screen()
            self.display_board()
            
            input("Press Enter to continue...")
            
            # Restore previous state
            self.selected_square = temp_selected
            self.possible_moves = temp_possible
            self.secondary_moves = temp_secondary
            
            return hint_move
        return None
        
    def start(self):
        """Start the chess game"""
        self.clear_screen()
        
        print("Enhanced Terminal Chess Game")
        print("---------------------------")
        
        # Display the initial board
        self.display_board()
        
        # If AI goes first (human is black), let AI make first move
        if self.human_color == chess.BLACK:
            self.make_ai_move()
            self.clear_screen()
            self.display_board()
        
        # Main game loop
        game_over = False
        while not game_over:
            # Get user input for move
            user_input = input("\nEnter move (e.g., e2e4) or command: ")
            
            # Process the input
            move_made = self.process_input(user_input)
            
            # Clear screen and redisplay board
            self.clear_screen()
            self.display_board()
            
            # Check for game over
            if self.board.is_game_over():
                result = self.board.result()
                if self.board.is_checkmate():
                    winner = "White" if self.board.turn == chess.BLACK else "Black"
                    print(f"\nCheckmate! {winner} wins.")
                elif self.board.is_stalemate():
                    print("\nGame over. Stalemate.")
                else:
                    print(f"\nGame over. Result: {result}")
                
                game_over = True
                continue
            
            # If a move was made, let AI respond
            if move_made:
                # Difficulty: score the human's move against the policy before the AI replies.
                # board_history[-1] is the position before this human move; last_move is the move.
                if self.difficulty_controller.enabled and self.last_move is not None \
                        and self.board_history:
                    self.difficulty_controller.observe_player_move(
                        self.board_history[-1], self.last_move)

                # AI makes its move
                self.make_ai_move()
                
                # Clear screen and redisplay board
                self.clear_screen()
                self.display_board()
                
                # Check for game over again
                if self.board.is_game_over():
                    result = self.board.result()
                    if self.board.is_checkmate():
                        winner = "White" if self.board.turn == chess.BLACK else "Black"
                        print(f"\nCheckmate! {winner} wins.")
                    elif self.board.is_stalemate():
                        print("\nGame over. Stalemate.")
                    else:
                        print(f"\nGame over. Result: {result}")
                    
                    game_over = True
                    
        # :Played-buffer: (operator 2026-07-14) — every FINISHED human game is archived
        # as training experience. Human = opponent rung (same legal class as the SF
        # ladder; KnightCap's original FICS recipe); targets stay self-generated — the
        # human's moves shape the states, never the labels.
        if game_over and self.move_history:
            try:
                import chess.pgn as _pgn
                os.makedirs("data/human_games", exist_ok=True)
                g = _pgn.Game()
                g.headers["Result"] = self.board.result()
                g.headers["White"] = "human" if self.human_color == chess.WHITE else "champion"
                g.headers["Black"] = "champion" if self.human_color == chess.WHITE else "human"
                g.headers["Date"] = time.strftime("%Y.%m.%d")
                node = g
                for u in self.move_history:
                    node = node.add_variation(chess.Move.from_uci(u))
                path = f"data/human_games/{time.strftime('%Y%m%d_%H%M%S')}.pgn"
                with open(path, "w") as fh:
                    fh.write(str(g))
                print(f"(archived to the played buffer: {path})")
            except Exception as e:
                print(f"(played-buffer archive failed: {e})")
        print("\nThanks for playing!")