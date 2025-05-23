import chess
import colorama
import os
import time
from colorama import Fore, Back, Style
from evaluation import fast_evaluate_position, find_threatened_squares, find_guarded_squares
from board_utils import get_legal_moves_from_square, get_secondary_moves, get_move_uci

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
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
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
        
        # Available commands
        print("\nCommands: 'hint', 'undo', 'save', 'load', 'resign', 'cancel', or enter move (e.g., e2e4)")
        
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
        
        # Handle commands
        if command == 'hint':
            self.show_move_hint()
            return False
        elif command == 'undo':
            self.undo_move()
            return False
        elif command == 'save':
            self.save_game()
            return False
        elif command == 'load':
            self.prompt_load_game()
            return False
        elif command == 'resign':
            print("You resigned the game.")
            return True  # Signal game over
        elif command == 'cancel':
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
            
            # Get AI's move
            self.ai.board = self.board.copy()  # Make sure AI has current board
            move = self.ai.get_best_move()
            
            end_time = time.time()
            print(f"AI plays: {move.uci()} (took {end_time - start_time:.2f}s)")
            
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
                    
        print("\nThanks for playing!")