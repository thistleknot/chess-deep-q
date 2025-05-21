import pygame
import threading
import queue
import chess
import time
import os
import math
from evaluation import find_threatened_squares, find_guarded_squares, fast_evaluate_position, calculate_attack_range, calculate_movement_range
from board_utils import get_legal_moves_from_square, square_name_to_square, get_secondary_moves
from constants import (
    LIGHT_SQUARE_COLOR, DARK_SQUARE_COLOR, HIGHLIGHT_COLOR, 
    POSSIBLE_MOVE_COLOR, SECONDARY_MOVE_COLOR, THREAT_COLOR, GUARDED_COLOR
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# At the beginning of your ui.py file
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require tkinter
import threading
import queue
import time

class NonClickableChessBoard:
    def __init__(self, board, ai, human_color=chess.WHITE):
        self.board = board
        self.ai = ai
        self.human_color = human_color
        self.selected_square = None
        self.possible_moves = set()
        self.secondary_moves = set()
        self.last_move = None
        self.move_history = []
        # Store board states for undo feature
        self.board_history = [board.copy()]
        self.fig = None
        self.ax = None
        self.move_hint = None
        self.highlighted_hint = None
        
    def process_input(self):
        """Handle keyboard input to select piece and destination"""
        if self.board.turn != self.human_color:
            return False
            
        # Print available pieces for clarity
        print("\nYour pieces:")
        available_pieces = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                square_name = chess.square_name(square)
                piece_symbol = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
                available_pieces.append(f"{square_name}:{piece_symbol}")
        
        # Print in a nice grid
        for i in range(0, len(available_pieces), 8):
            print(" ".join(available_pieces[i:i+8]))
        
        # First prompt for piece selection
        piece_coord = input("\nEnter piece to move (e.g., e2) or 'undo' or 'hint': ").lower()
        
        # Handle special commands
        if piece_coord == 'undo':
            self.undo_move()
            return False
        elif piece_coord == 'hint':
            self.show_move_hint()
            return False
        elif piece_coord == 'save':
            self.save_game()
            return False
        elif piece_coord == 'load':
            self.prompt_load_game()
            return False
        elif piece_coord == 'resign':
            print("You resigned the game.")
            return True  # Signal game over
        
        # Convert input to square number
        square = square_name_to_square(piece_coord) if len(piece_coord) == 2 else None
        
        # Validate piece selection
        if square is None:
            print("Invalid square notation. Please use format like 'e2'.")
            return False
            
        piece = self.board.piece_at(square)
        if not piece:
            print(f"No piece at {piece_coord}. Please select a square with your piece.")
            return False
            
        if piece.color != self.human_color:
            print(f"That's not your piece. Please select one of your pieces.")
            return False
        
        # Valid piece selected, show possible moves
        self.selected_square = square
        self.possible_moves = get_legal_moves_from_square(self.board, square)
        self.secondary_moves = get_secondary_moves(self.board, square)
        
        # Handle special case for castling
        castling_options = self.get_castling_options(square)
        
        # Update board to show highlights
        self.update_board()
        
        # Show available destinations
        print("\nPossible moves:")
        destination_list = [chess.square_name(move) for move in self.possible_moves]
        
        # Add castling options to the display if available
        if castling_options:
            for castle_type, castle_square in castling_options.items():
                if castle_square not in self.possible_moves:
                    destination_list.append(f"{chess.square_name(castle_square)} (Castle {castle_type})")
        
        # Print in a nice grid
        for i in range(0, len(destination_list), 8):
            print(" ".join(destination_list[i:i+8]))
        
        # Get destination selection
        dest_coord = input("\nEnter destination square (or 'cancel'): ").lower()
        
        if dest_coord == 'cancel':
            # Cancel the selection
            self.selected_square = None
            self.possible_moves = set()
            self.secondary_moves = set()
            self.update_board()
            return False
        
        # Parse the destination
        dest_square = square_name_to_square(dest_coord) if len(dest_coord) == 2 else None
        
        if dest_square is None:
            print("Invalid square notation. Please use format like 'e4'.")
            self.selected_square = None
            self.possible_moves = set()
            self.secondary_moves = set()
            return False
        
        # Check if destination is a valid move
        if dest_square in self.possible_moves:
            move = chess.Move(self.selected_square, dest_square)
            
            # Check for promotion - default to queen but allow choice
            if self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                # Check if pawn is moving to the last rank
                if (self.human_color == chess.WHITE and chess.square_rank(dest_square) == 7) or \
                   (self.human_color == chess.BLACK and chess.square_rank(dest_square) == 0):
                    # Ask for promotion piece
                    valid_promotions = {'q': chess.QUEEN, 'r': chess.ROOK, 
                                      'b': chess.BISHOP, 'n': chess.KNIGHT}
                    promotion = input("Promote to (q=Queen, r=Rook, b=Bishop, n=Knight, default=q): ").lower() or 'q'
                    if promotion in valid_promotions:
                        move.promotion = valid_promotions[promotion]
                    else:
                        move.promotion = chess.QUEEN
                        print("Invalid choice. Promoting to Queen.")
            
            # Check if this is a castling move
            if piece.piece_type == chess.KING and abs(self.selected_square - dest_square) == 2:
                # This is a castling move
                if dest_square > self.selected_square:
                    print("Castling kingside")
                else:
                    print("Castling queenside")
            
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
            self.move_hint = None
            self.highlighted_hint = None
            
            # Update board after the move
            self.update_board()
            return True  # Move successfully made
        else:
            # Handle castling case
            if castling_options and dest_square in castling_options.values():
                # Find which castling option was selected
                for castle_type, castle_square in castling_options.items():
                    if dest_square == castle_square:
                        # This is a castling move, but we need to create the right king move
                        king_square = self.get_king_square()
                        if king_square:
                            if castle_type == "kingside":
                                castle_move = chess.Move(king_square, king_square + 2)
                            else:  # queenside
                                castle_move = chess.Move(king_square, king_square - 2)
                                
                            # Save state and make move
                            self.board_history.append(self.board.copy())
                            self.board.push(castle_move)
                            self.last_move = castle_move
                            self.move_history.append(castle_move.uci())
                            print(f"Castling {castle_type}")
                            
                            # Reset selection state
                            self.selected_square = None
                            self.possible_moves = set()
                            self.secondary_moves = set()
                            self.move_hint = None
                            self.highlighted_hint = None
                            
                            # Update board after the move
                            self.update_board()
                            return True
            
            print(f"Invalid destination. {dest_coord} is not a legal move.")
            self.selected_square = None
            self.possible_moves = set()
            self.secondary_moves = set()
            return False
    
    def get_king_square(self):
        """Get the square of the current player's king"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.KING and piece.color == self.board.turn:
                return square
        return None
    
    def get_castling_options(self, square):
        """Get castling options when selecting a rook or king"""
        castling_options = {}
        piece = self.board.piece_at(square)
        
        # If it's a king, check for castling possibilities
        if piece and piece.piece_type == chess.KING:
            if self.board.turn == chess.WHITE:
                # White castling
                if self.board.has_kingside_castling_rights(chess.WHITE):
                    castling_options["kingside"] = chess.G1
                if self.board.has_queenside_castling_rights(chess.WHITE):
                    castling_options["queenside"] = chess.C1
            else:
                # Black castling
                if self.board.has_kingside_castling_rights(chess.BLACK):
                    castling_options["kingside"] = chess.G8
                if self.board.has_queenside_castling_rights(chess.BLACK):
                    castling_options["queenside"] = chess.C8
        
        # If it's a rook, check which castling it corresponds to
        elif piece and piece.piece_type == chess.ROOK:
            if self.board.turn == chess.WHITE:
                # White rook
                if square == chess.H1 and self.board.has_kingside_castling_rights(chess.WHITE):
                    # Find the king square
                    king_square = self.get_king_square()
                    if king_square:
                        castling_options["kingside"] = chess.G1
                elif square == chess.A1 and self.board.has_queenside_castling_rights(chess.WHITE):
                    king_square = self.get_king_square()
                    if king_square:
                        castling_options["queenside"] = chess.C1
            else:
                # Black rook
                if square == chess.H8 and self.board.has_kingside_castling_rights(chess.BLACK):
                    king_square = self.get_king_square()
                    if king_square:
                        castling_options["kingside"] = chess.G8
                elif square == chess.A8 and self.board.has_queenside_castling_rights(chess.BLACK):
                    king_square = self.get_king_square()
                    if king_square:
                        castling_options["queenside"] = chess.C8
                        
        return castling_options
    
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
            self.update_board()
    
    def undo_move(self):
        """Undo the last move pair (human + AI)"""
        # Need at least one move to undo
        if len(self.move_history) == 0:
            print("No moves to undo.")
            return False
            
        # If it's the human's turn, we need to undo the last AI move and the human move before that
        if self.board.turn == self.human_color:
            # Make sure we have at least two moves to undo (human and AI)
            if len(self.move_history) >= 2:
                # Undo the last AI move and the human move before that
                self.move_history.pop()  # Remove AI move
                self.move_history.pop()  # Remove human move
                
                # Load the board state from before the human's move
                if len(self.board_history) >= 2:
                    self.board_history.pop()  # Current state
                    self.board_history.pop()  # AI's move state
                    self.board = self.board_history[-1].copy()
                else:
                    # Fallback if history is missing
                    self.board = chess.Board()
                    for move_uci in self.move_history:
                        self.board.push(chess.Move.from_uci(move_uci))
                
                # Set last move if there are any moves left
                if self.move_history:
                    self.last_move = chess.Move.from_uci(self.move_history[-1])
                else:
                    self.last_move = None
                
                print("Undid last move pair.")
                
                # Reset selection state
                self.selected_square = None
                self.possible_moves = set()
                self.secondary_moves = set()
                
                # Update the AI's board
                self.ai.board = self.board.copy()
                
                self.update_board()
                return True
            else:
                print("Not enough moves to undo.")
                return False
        # If it's the AI's turn, we just need to undo the human's last move
        else:
            # Pop the last move
            self.move_history.pop()
            
            # Load previous board state
            if len(self.board_history) >= 1:
                self.board = self.board_history.pop().copy()
            else:
                # Fallback if history is missing
                self.board = chess.Board()
                for move_uci in self.move_history:
                    self.board.push(chess.Move.from_uci(move_uci))
            
            # Set last move if there are any moves left
            if self.move_history:
                self.last_move = chess.Move.from_uci(self.move_history[-1])
            else:
                self.last_move = None
            
            print("Undid last move.")
            
            # Reset selection state
            self.selected_square = None
            self.possible_moves = set()
            self.secondary_moves = set()
            
            # Update the AI's board
            self.ai.board = self.board.copy()
            
            self.update_board()
            return True
    
    def show_move_hint(self):
        """Get a hint for the best move from the AI (optional feature)"""
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
            self.move_hint = hint_move
            self.highlighted_hint = (hint_move.from_square, hint_move.to_square)
            self.update_board()
            
            return hint_move
        return None
    
    def save_game(self, filename=None):
        """Save the current game"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"chess_game_{timestamp}.pgn"
        
        path = save_game_to_pgn(self.board, self.move_history, filename)
        print(f"Game saved to {path}")
        return path
    
    def prompt_load_game(self):
        """Prompt the user to select a saved game"""
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
            self.move_hint = None
            self.highlighted_hint = None
            
            # Update the AI's board
            self.ai.board = board.copy()
            
            self.update_board()
            print(f"Game loaded from {filename}")
            return True
        else:
            print(f"Failed to load game from {filename}")
            return False
        
    def update_board(self):
        """Update the board display"""
        # Clear previous figure if it exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Get evaluation score
        evaluation = fast_evaluate_position(self.board)
        # Display evaluation from white's perspective (positive = white advantage)
        display_eval = evaluation
        
        # Find threatened and guarded squares
        threatened_squares = find_threatened_squares(self.board)
        guarded_squares = find_guarded_squares(self.board)
        
        # Create the visual board - now the function always shows white at bottom
        self.fig, self.ax = create_visual_chess_board(
            self.board, 
            last_move=self.last_move,
            threatened_squares=threatened_squares,
            guarded_squares=guarded_squares,
            selected_square=self.selected_square,
            possible_moves=self.possible_moves,
            secondary_moves=self.secondary_moves,
            highlighted_hint=self.highlighted_hint,
            evaluation=display_eval
            # perspective parameter removed
        )
        
        # Add control buttons (non-clickable, just for reference)
        self.add_control_buttons()
        
        plt.draw()
        plt.pause(0.1)
        
        # Check for game over
        if self.board.is_game_over():
            result = self.board.result()
            if self.board.is_checkmate():
                winner = "White" if self.board.turn == chess.BLACK else "Black"
                print(f"Checkmate! {winner} wins.")
            elif self.board.is_stalemate():
                print("Game over. Stalemate.")
            else:
                print(f"Game over. Result: {result}")
            
            return True  # Game is over
        return False  # Game continues
    
    def add_control_buttons(self):
        """Add control buttons to the board (visual reference only)"""
        # Adjust the figure to make room for reference text
        self.fig.subplots_adjust(bottom=0.15)
        
        # Add text showing available commands
        commands_text = "Commands: 'hint', 'undo', 'save', 'load', 'resign', 'cancel'"
        plt.figtext(0.5, 0.05, commands_text, ha='center', fontsize=12)
        
    def start(self):
        """Start the interactive chess game without clicking"""
        plt.ion()  # Turn on interactive mode
        
        print("Interactive Chess Game (No-Click Version)")
        print("----------------------------------------")
        print("Enter coordinates to select pieces and make moves.")
        print("Available commands:")
        print("- hint: Get a move suggestion")
        print("- undo: Take back your last move")
        print("- save: Save the current game")
        print("- load: Load a previously saved game")
        print("- resign: Resign the current game")
        print("- cancel: Cancel your current piece selection")
        
        # Display the initial board
        game_over = self.update_board()
        
        # If AI goes first (human is black), let AI make first move
        if self.human_color == chess.BLACK:
            self.make_ai_move()
        
        try:
            # Main game loop
            while not game_over:
                if self.board.turn == self.human_color:
                    # Human's turn - get input
                    move_made = self.process_input()
                    
                    # If a move was made, let AI respond
                    if move_made:
                        game_over = self.update_board()
                        if not game_over:
                            # Pause briefly before AI move
                            plt.pause(0.5)
                            self.make_ai_move()
                            game_over = self.update_board()
                else:
                    # AI's turn
                    self.make_ai_move()
                    game_over = self.update_board()
            
            # Keep the figure open
            plt.ioff()
            plt.show(block=True)
        finally:
            # Ensure proper cleanup when the game ends or is interrupted
            plt.close('all')
def create_visual_chess_board(board, last_move=None, threatened_squares=None, 
                             guarded_squares=None, selected_square=None, possible_moves=None, 
                             secondary_moves=None, highlighted_hint=None, evaluation=None, fig_size=(10, 10)):
    """Create a visual chess board with improved highlighting system"""
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Initialize or use provided sets
    possible_moves = possible_moves or set()
    secondary_moves = secondary_moves or set()
    
    # Create color constants for the new scheme
    OPPONENT_ATTACK_COLOR = '#e88686'  # Red
    OPPONENT_SECONDARY_COLOR = '#f0b6b6'  # Light red
    PLAYER_MOVE_COLOR = '#7eb36a'  # Green
    PLAYER_SECONDARY_COLOR = '#b8e3ae'  # Light green
    
    # Calculate attack and movement ranges
    opponent_attack_range = calculate_attack_range(board, not board.turn)
    player_movement_range = calculate_movement_range(board, board.turn)
    
    # Draw the board squares
    for row in range(8):
        for col in range(8):
            # Set base square color (checkerboard pattern)
            is_light = (row + col) % 2 == 0
            base_color = LIGHT_SQUARE_COLOR if is_light else DARK_SQUARE_COLOR
            
            # Draw the base square
            ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=base_color))
            
            # Map to chess square
            square = chess.square(col, row)
            
            # Determine if the square is under attack or can be moved to
            is_opponent_attack = square in opponent_attack_range
            is_player_move = square in player_movement_range
            
            # Apply the appropriate highlighting
            if is_opponent_attack and is_player_move:
                # Contested square - use a blend of colors
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor='#d9c27a', alpha=0.4))  # Blended color
            elif is_opponent_attack:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=OPPONENT_ATTACK_COLOR, alpha=0.4))
            elif is_player_move:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=PLAYER_MOVE_COLOR, alpha=0.4))
            
            # Apply other highlights on top
            if last_move and (square == last_move.from_square or square == last_move.to_square):
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=HIGHLIGHT_COLOR, alpha=0.5))
                
            if selected_square is not None and square == selected_square:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=HIGHLIGHT_COLOR, alpha=0.6))
                
            if square in possible_moves:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=POSSIBLE_MOVE_COLOR, alpha=0.6))
                
            if square in secondary_moves:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=SECONDARY_MOVE_COLOR, alpha=0.4))
    
    # Place pieces on the board
    piece_symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            
            symbol = piece_symbols[piece.symbol()]
            ax.text(file_idx + 0.5, rank_idx + 0.5, symbol, fontsize=30, 
                   ha='center', va='center')
    
    # Add coordinate labels
    for i in range(8):
        ax.text(-0.3, i+0.5, str(i+1), fontsize=14, va='center')
        ax.text(i+0.5, -0.3, chr(97+i), fontsize=14, ha='center')
    
    # Set up the board view
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title with turn and evaluation
    turn = "White" if board.turn == chess.WHITE else "Black"
    title = f"{turn} to move"
    if evaluation is not None:
        eval_prefix = "+" if evaluation > 0 and abs(evaluation) < 100 else ""
        title += f" | Eval: {eval_prefix}{evaluation:.2f}"
    
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    
    return fig, ax

