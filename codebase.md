# Table of Contents
- .gitignore
- board_utils.py
- chessv3.py
- chess_ai.py
- constants.py
- evaluation.py
- file_structure.txt
- game_play.py
- LICENSE
- main.py
- mcts.py
- menu.py
- neural_network.py
- README.md
- terminal_board.py
- threaded_board.py
- ui.py

## File: .gitignore

- Extension: 
- Language: unknown
- Size: 3617 bytes
- Created: 2025-05-19 18:55:47
- Modified: 2025-05-19 18:55:47

### Code

```unknown
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# UV
#   Similar to Pipfile.lock, it is generally recommended to include uv.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#uv.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/latest/usage/project/#working-with-version-control
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

# Ruff stuff:
.ruff_cache/

# PyPI configuration file
.pypirc

```

## File: board_utils.py

- Extension: .py
- Language: python
- Size: 4781 bytes
- Created: 2025-05-19 19:22:00
- Modified: 2025-05-20 19:53:21

### Code

```python
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

def boards_to_tensor_batch(boards):
    """Convert a list of boards to a batch tensor"""
    batch_size = len(boards)
    tensor_batch = torch.zeros(batch_size, 12, 8, 8)
    
    for b_idx, board in enumerate(boards):
        tensor_batch[b_idx] = board_to_tensor(board)
    
    return tensor_batch


@lru_cache(maxsize=5000)
def get_valid_moves_cached(board_fen):
    """Get valid moves for a board position, using caching for efficiency"""
    board = chess.Board(board_fen)
    return list(board.legal_moves)

def get_valid_moves(board):
    """Get valid moves with caching"""
    return get_valid_moves_cached(board.fen())

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


```

## File: chessv3.py

- Extension: .py
- Language: python
- Size: 523 bytes
- Created: 2025-05-20 19:02:25
- Modified: 2025-05-20 19:53:36

### Code

```python


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import time
import chess
from collections import defaultdict, deque, OrderedDict
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Add this line
from matplotlib.widgets import Button

```

## File: chess_ai.py

- Extension: .py
- Language: python
- Size: 35045 bytes
- Created: 2025-05-19 19:10:15
- Modified: 2025-05-20 19:41:30

### Code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import time
import chess
from collections import deque
import multiprocessing
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from neural_network import ChessQNetwork, DQNAgent
from mcts import ParallelRussianDollMCTS
from evaluation import fast_evaluate_position, format_score
from board_utils import board_to_tensor, EVAL_CACHE, CACHE_LOCK

# Optimized Chess AI class
class OptimizedChessAI:
    def __init__(self, training_games=20, verbose=False):
        self.dqn_agent = DQNAgent()
        self.training_games = training_games
        self.board = chess.Board()
        self.game_history = []
        self.evaluation_history = []
        self.loss_history = []
        self.epsilon_history = []
        self.verbose = verbose
        self.move_count_history = []
        
        # Set number of CPU cores to use
        self.num_cpu_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {self.num_cpu_cores} CPU cores for parallel processing")
    
    def reset_board(self):
        """Reset the chess board to the starting position"""
        self.board = chess.Board()
    
    def set_board_from_fen(self, fen):
        """Set the board to a specific position using FEN notation"""
        self.board = chess.Board(fen)
    
    def make_move(self, move):
        """Make a move on the board"""
        self.board.push(move)
        
    def get_best_move(self, training_progress=0.0, is_training=False, current_move=0, max_moves=200):
        """Get the best move for the current position"""
        return self.dqn_agent.select_move(
            self.board, 
            training_progress, 
            is_training, 
            current_move, 
            max_moves
        )
    def plot_evaluation_trends_across_games(self):
        """Plot the evaluation trends across all games"""
        if not self.game_history:
            print("No game history available.")
            return
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Game numbers for x-axis
        game_numbers = list(range(1, len(self.game_history) + 1))
        
        # Extract the final raw evaluation from each game
        final_evals = [game['final_score'] for game in self.game_history]
        
        # Plot the raw evaluation values
        plt.plot(game_numbers, final_evals, 'b-', marker='o', linewidth=2)
        
        # Add reference line at 0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Color code by game result
        results = [game['result'] for game in self.game_history]
        for i, result in enumerate(results):
            if result == '1-0':  # White win
                plt.plot(game_numbers[i], final_evals[i], 'bo', markersize=8)
            elif result == '0-1':  # Black win
                plt.plot(game_numbers[i], final_evals[i], 'ro', markersize=8)
            else:  # Draw
                plt.plot(game_numbers[i], final_evals[i], 'go', markersize=8)
        
        # Add a trend line
        from scipy import stats
        if len(game_numbers) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(game_numbers, final_evals)
            regression_line = [slope * x + intercept for x in game_numbers]
            plt.plot(game_numbers, regression_line, 'k--', alpha=0.7)
        
        # Add labels and title
        plt.title('Final Position Evaluation Across Training Games')
        plt.xlabel('Game Number')
        plt.ylabel('Evaluation (+ favors White, - favors Black)')
        plt.grid(True, alpha=0.3)
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='White Win'),
            Patch(facecolor='red', label='Black Win'),
            Patch(facecolor='green', label='Draw')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
    # 2. Modify the self_play_game method to display scores and track them for plotting
    def self_play_game(self, max_moves=200):
        """Play a game against itself and store the transitions for learning"""
        self.reset_board()
        move_count = 0
        game_moves = []
        
        # For score tracking and plotting
        white_scores = []
        black_scores = []
        
        # Calculate training progress (0 to 1)
        training_progress = min(len(self.game_history) / self.training_games, 1.0)
        
        # Display setup - but don't create the plot yet
        if self.verbose:
            print(f"\rGame {len(self.game_history)+1}: Move 0/{max_moves}", end="", flush=True)
        
        while not self.board.is_game_over() and move_count < max_moves:
            # Remember the state before the move
            state_before = self.board.copy()
            
            # Select a move with advanced annealing
            move = self.get_best_move(
                training_progress, 
                is_training=True,
                current_move=move_count,
                max_moves=max_moves
            )
            
            if move is None:
                break
                
            self.make_move(move)
            game_moves.append(get_move_uci(move))
            move_count += 1
            
            # Get the raw evaluation with existing function
            raw_score = fast_evaluate_position(self.board)
            
            # Store the raw values
            white_scores.append(raw_score)
            black_scores.append(-raw_score)  # Black's score is opposite of white's
            
            # Display properly formatted scores
            if self.verbose:
                white_score_str = format_score(raw_score)
                black_score_str = format_score(-raw_score)
                print(f"\rGame {len(self.game_history)+1}: Move {move_count}/{max_moves} - White: {white_score_str} | Black: {black_score_str}", end="", flush=True)
            
            # Get the reward
            if self.board.is_checkmate():
                reward = -1.0
                done = True
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                reward = 0.0
                done = True
            else:
                # Use evaluation function for reward signal
                eval_score = fast_evaluate_position(self.board)
                reward = 0.01 * math.tanh(eval_score / 10.0)  # Normalize and scale down
                done = False
            
            # Store the transition
            self.dqn_agent.store_transition(state_before, move, reward, self.board.copy(), done)
            
            # Train the network more frequently but with smaller batches
            if len(self.dqn_agent.memory) >= self.dqn_agent.batch_size:
                loss = self.dqn_agent.train()
                self.loss_history.append(loss)
        
        # Store game result and evaluation
        result = self.board.result()
        # At the end of the game
        final_evaluation = fast_evaluate_position(self.board)
        
        # Display final scores properly
        if self.verbose:
            white_score_str = format_score(final_evaluation)
            black_score_str = format_score(-final_evaluation)  # Black's score is opposite
            print(f"\nFinal scores - White: {white_score_str} | Black: {black_score_str}")
            print(f"Result: {result}, Moves: {move_count}")
        
        final_evaluation = fast_evaluate_position(self.board, ignore_checkmate=True)
        
        # Store the game data with the proper evaluation
        self.game_history.append({
            'moves': game_moves,
            'result': result,
            'move_count': move_count,
            'white_scores': white_scores,
            'black_scores': black_scores,
            'final_score': final_evaluation  # Store the actual numerical value
        })
        
        self.evaluation_history.append(final_evaluation)
        self.epsilon_history.append(self.dqn_agent.epsilon)
        self.move_count_history.append(move_count)
        
        # Display final scores
        if self.verbose:
            black_score_str = format_score(-final_evaluation)
            print(f"\nFinal scores - White: {white_score_str} | Black: {black_score_str}")
            print(f"Result: {result}, Moves: {move_count}")
            
            # Plot the game progress AFTER completion
            plt.figure(figsize=(10, 6))
            move_numbers = list(range(1, len(white_scores) + 1))
            plt.plot(move_numbers, white_scores, 'b-', label="White")
            plt.plot(move_numbers, black_scores, 'r-', label="Black")
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f"Game {len(self.game_history)} Score Progression")
            plt.xlabel("Move Number")
            plt.ylabel("Score")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Periodically update the target network
        if len(self.game_history) % 5 == 0:
            self.dqn_agent.update_target_network()
            
            # Clear evaluation cache periodically to avoid memory bloat
            global EVAL_CACHE
            EVAL_CACHE = {}
        
        return result, move_count

    # 3. Add a method to plot the final game scores after training
    def plot_final_game_scores(self):
        """Plot the final scores for all games after training"""
        if not self.game_history:
            print("No game history available. Please train the AI first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Extract data from game history
        game_numbers = list(range(1, len(self.game_history) + 1))
        final_scores = [game['final_score'] for game in self.game_history]
        results = [game['result'] for game in self.game_history]
        
        # Create color map based on game results
        colors = []
        for result in results:
            if result == '1-0':  # White win
                colors.append('blue')
            elif result == '0-1':  # Black win
                colors.append('red')
            else:  # Draw or unfinished
                colors.append('green')
        
        # Plot the scores
        plt.bar(game_numbers, final_scores, color=colors)
        
        # Add a horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Create a custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='White Win'),
            Patch(facecolor='red', label='Black Win'),
            Patch(facecolor='green', label='Draw')
        ]
        plt.legend(handles=legend_elements)
        
        # Add labels and title
        plt.xlabel('Game Number')
        plt.ylabel('Final Score')
        plt.title('Final Scores Across All Training Games')
        plt.grid(True, alpha=0.3)
        
        # Add score values above each bar
        for i, score in enumerate(final_scores):
            plt.text(i + 1, score + (1 if score >= 0 else -1), 
                     f"{score:.2f}", ha='center')
        
        plt.tight_layout()
        plt.show()
    # 2. Now, let's add a function to plot the number of moves per game
    def plot_move_counts(self):
        """Plot the number of moves per game"""
        if not self.move_count_history:
            print("No move count data to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.move_count_history) + 1), self.move_count_history, 'o-')
        plt.title('Moves per Game')
        plt.xlabel('Game Number')
        plt.ylabel('Number of Moves')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 3. Let's add a function to save training data to CSV
    def save_training_data_to_csv(self, filename="training_data.csv"):
        """Save training data to a CSV file"""
        import pandas as pd
        
        # Create data dictionary
        data = {
            'game_number': list(range(1, len(self.game_history) + 1)),
            'result': [game['result'] for game in self.game_history],
            'move_count': self.move_count_history,
            'final_score': [game['final_score'] for game in self.game_history],
            'white_win': [1 if game['result'] == '1-0' else 0 for game in self.game_history],
            'black_win': [1 if game['result'] == '0-1' else 0 for game in self.game_history],
            'draw': [1 if game['result'] == '1/2-1/2' else 0 for game in self.game_history]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Training data saved to {filename}")
        
        return df
    
    # 4. Finally, let's modify the train method to automatically show plots at the end of training
    def train(self, progress_interval=1):
        """Train the AI through self-play with optimized performance"""
        print(f"Training for {self.training_games} games...")
        start_time = time.time()
        
        # Create a progress bar for the training
        with tqdm(total=self.training_games, desc="Training Progress") as pbar:
            for i in range(self.training_games):
                result, moves = self.self_play_game()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Game': f"{i+1}/{self.training_games}",
                    'Result': result,
                    'Moves': moves,
                    'Epsilon': f"{self.dqn_agent.epsilon:.4f}"
                })
                
                # Print detailed stats at intervals
                if (i + 1) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    
                    print(f"\nGame {i+1}/{self.training_games} completed in {elapsed:.2f}s")
                    print(f"Result: {result}, Moves: {moves}")
                    
                    if self.loss_history:
                        recent_loss = self.loss_history[-min(50, len(self.loss_history)):]
                        avg_loss = sum(recent_loss) / len(recent_loss)
                        print(f"Average loss: {avg_loss:.4f}")
                    
                    print(f"Exploration rate (epsilon): {self.dqn_agent.epsilon:.4f}")
                    
                    avg_moves = sum(self.move_count_history[-min(10, len(self.move_count_history)):]) / \
                                min(10, len(self.move_count_history))
                    print(f"Average moves per game: {avg_moves:.1f}")
                    
                    # Estimate time remaining
                    time_per_game = elapsed / (i + 1)
                    remaining_games = self.training_games - (i + 1)
                    eta = remaining_games * time_per_game
                    print(f"Estimated time remaining: {eta:.2f}s")
                    print(f"Average time per game: {time_per_game:.2f}s")
        
        # Final stats
        total_time = time.time() - start_time
        print("\nTraining completed!")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Average time per game: {total_time / self.training_games:.2f}s")
        
        # Automatically show training summary plots without user interaction
        print("\nGenerating training summary plots...")
        
        # Show evaluation trends across all games
        self.plot_evaluation_trends_across_games()
        
        # Show other training metrics
        self.plot_training_progress()
        self.plot_final_game_scores()
        self.plot_move_counts()
        
        # Save training data to CSV
        save_csv = input("Save training data to CSV? (y/n, default: n): ").lower() or 'n'
        if save_csv == 'y':
            filename = input("Enter CSV filename (default: training_data.csv): ") or "training_data.csv"
            self.save_training_data_to_csv(filename)
        
        return
    
    def plot_training_progress(self):
        """Plot the training progress"""
        if not self.loss_history or not self.evaluation_history:
            print("No training data to plot.")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot loss history
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history)
        plt.title('Q-Network Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        
        # Plot evaluation history
        plt.subplot(2, 2, 2)
        plt.plot(self.evaluation_history)
        plt.title('Board Evaluation')
        plt.xlabel('Games')
        plt.ylabel('Evaluation Score')
        
        # Plot epsilon history
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilon_history)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Games')
        plt.ylabel('Epsilon')
        
        # Plot move count history
        plt.subplot(2, 2, 4)
        plt.plot(self.move_count_history)
        plt.title('Moves per Game')
        plt.xlabel('Games')
        plt.ylabel('Move Count')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename='chess_ai_model.pth'):
        """Save the Q-network model and training state"""
        torch.save({
            'q_network_state_dict': self.dqn_agent.q_network.state_dict(),
            'target_network_state_dict': self.dqn_agent.target_q_network.state_dict(),
            'optimizer_state_dict': self.dqn_agent.optimizer.state_dict(),
            'epsilon': self.dqn_agent.epsilon,
            'game_history': self.game_history,
            'evaluation_history': self.evaluation_history,
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'move_count_history': self.move_count_history,
            'total_games_trained': len(self.game_history),
            'training_games': self.training_games,
            'save_timestamp': time.time()
        }, filename)
    
    def load_model(self, filename='chess_ai_model.pth', continue_training=False):
        """
        Load the Q-network model and optionally training state
        
        Args:
            filename: Path to the saved model file
            continue_training: If True, also load training history and state
        """
        """Load the Q-network model"""
        checkpoint = torch.load(filename)
        self.dqn_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.dqn_agent.target_q_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn_agent.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filename}")


    def evaluate_elo_rating(self, num_games=20, starting_elo=1500, step_size=100, min_elo=800, max_elo=3000):
        """
        Evaluate the ELO rating of the model by playing against Stockfish at different levels.
        
        Args:
            num_games: Number of games to play at each ELO level
            starting_elo: Initial Stockfish ELO to test against
            step_size: How much to adjust Stockfish ELO between tests
            min_elo: Minimum Stockfish ELO to test
            max_elo: Maximum Stockfish ELO to test
        
        Returns:
            Estimated ELO rating
        """
        try:
            import chess.engine
        except ImportError:
            print("Please install the chess.engine module: pip install python-chess[engine]")
            return None
        
        # Find Stockfish engine
        stockfish_path = self._find_stockfish_path()
        if not stockfish_path:
            print("Stockfish engine not found. Please install Stockfish and ensure it's in your PATH.")
            return None
        
        print(f"Found Stockfish at: {stockfish_path}")
        print(f"Starting ELO calibration process with {num_games} games per level...")
        
        # Initialize engine
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        # Binary search for ELO rating
        low_elo = min_elo
        high_elo = max_elo
        current_elo = starting_elo
        
        best_win_ratio = 0
        best_elo_estimate = 0
        results = []
        
        while high_elo - low_elo > step_size:
            # Set Stockfish ELO
            engine.configure({"UCI_Elo": current_elo, "UCI_LimitStrength": True})
            
            # Play games
            wins = 0
            draws = 0
            losses = 0
            
            print(f"\nTesting against Stockfish ELO {current_elo}...")
            
            for game_num in range(num_games):
                # Alternate colors
                model_plays_white = game_num % 2 == 0
                
                result = self._play_against_stockfish(
                    engine, 
                    model_plays_white=model_plays_white,
                    time_per_move=0.5  # 500ms per move
                )
                
                if result == 1:  # Model win
                    wins += 1
                    print("W", end="", flush=True)
                elif result == 0:  # Draw
                    draws += 1
                    print("D", end="", flush=True)
                else:  # Loss
                    losses += 1
                    print("L", end="", flush=True)
            
            print(f"\nResults vs ELO {current_elo}: {wins}W-{draws}D-{losses}L")
            
            # Calculate win ratio (counting draws as 0.5)
            win_ratio = (wins + 0.5 * draws) / num_games
            results.append((current_elo, win_ratio, wins, draws, losses))
            
            # Store best approximation
            if abs(win_ratio - 0.5) < abs(best_win_ratio - 0.5):
                best_win_ratio = win_ratio
                best_elo_estimate = current_elo
            
            # Binary search next ELO to test
            if win_ratio > 0.5:  # Model is stronger, increase Stockfish ELO
                low_elo = current_elo
                current_elo = (current_elo + high_elo) // 2
            else:  # Model is weaker, decrease Stockfish ELO
                high_elo = current_elo
                current_elo = (current_elo + low_elo) // 2
        
        # Close engine
        engine.quit()
        
        # Display results summary
        print("\nELO Calibration Results:")
        print("------------------------")
        for elo, ratio, w, d, l in results:
            print(f"Stockfish ELO {elo}: Win ratio {ratio:.2f} ({w}W-{d}D-{l}L)")
        
        # Plot results
        self._plot_elo_results(results)
        
        print(f"\nEstimated ELO rating: {best_elo_estimate}")
        return best_elo_estimate
    
    def _find_stockfish_path(self):
        """Find the Stockfish executable path"""
        # Common locations for Stockfish
        possible_paths = [
            "stockfish",  # If in PATH
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "C:/Program Files/Stockfish/stockfish.exe",
            "stockfish.exe"
        ]
        
        import shutil
        import os
        
        # Check user-defined path from environment variable
        if "STOCKFISH_PATH" in os.environ:
            if os.path.exists(os.environ["STOCKFISH_PATH"]):
                return os.environ["STOCKFISH_PATH"]
        
        # Try common paths
        for path in possible_paths:
            stockfish_path = shutil.which(path)
            if stockfish_path:
                return stockfish_path
        
        return None
    
    def _play_against_stockfish(self, engine, model_plays_white=True, time_per_move=0.5, max_moves=200):
        """
        Play a game against Stockfish
        
        Returns:
            1 for model win, 0 for draw, -1 for model loss
        """
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            if (board.turn == chess.WHITE) == model_plays_white:
                # Model's turn
                self.board = board.copy()
                move = self.get_best_move()
            else:
                # Stockfish's turn
                result = engine.play(board, chess.engine.Limit(time=time_per_move))
                move = result.move
            
            # Make the move
            if move in board.legal_moves:
                board.push(move)
                move_count += 1
            else:
                # Illegal move (shouldn't happen, but just in case)
                print(f"Illegal move attempted: {move}")
                return -1 if model_plays_white else 1
        
        # Game over - determine result
        if board.is_checkmate():
            # If white is checkmated, black wins
            if board.turn == chess.WHITE:
                return -1 if model_plays_white else 1
            else:
                return 1 if model_plays_white else -1
        else:
            # Draw
            return 0
        
    def _plot_elo_results(self, results):
        """Plot the ELO calibration results"""
        import matplotlib.pyplot as plt
        
        elos = [r[0] for r in results]
        ratios = [r[1] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(elos, ratios, 'o-', linewidth=2, markersize=10)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        
        # Find closest point to 0.5 win ratio
        closest_idx = min(range(len(ratios)), key=lambda i: abs(ratios[i] - 0.5))
        plt.plot(elos[closest_idx], ratios[closest_idx], 'ro', markersize=12)
        
        plt.annotate(f'ELO ≈ {elos[closest_idx]}', 
                    xy=(elos[closest_idx], ratios[closest_idx]),
                    xytext=(10, -30),
                    textcoords='offset points',
                    fontsize=12,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.title('ELO Calibration Results')
        plt.xlabel('Stockfish ELO')
        plt.ylabel('Win Ratio')
        plt.grid(True, alpha=0.3)
        
        # Add 50% line label
        plt.text(elos[0], 0.51, '50% win ratio', fontsize=10, va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    # Fix the plot_final_game_scores method to better show actual evaluation
    def plot_final_game_scores(self):
        """Plot the final scores for all games after training"""
        if not self.game_history:
            print("No game history available. Please train the AI first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Extract data from game history
        game_numbers = list(range(1, len(self.game_history) + 1))
        final_scores = [game['final_score'] for game in self.game_history]
        results = [game['result'] for game in self.game_history]
        
        # Create color map based on game results
        colors = []
        for result in results:
            if result == '1-0':  # White win
                colors.append('blue')
            elif result == '0-1':  # Black win
                colors.append('red')
            else:  # Draw or unfinished
                colors.append('green')
        
        # Plot the scores
        plt.bar(game_numbers, final_scores, color=colors)
        
        # Add a horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Create a custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='White Win'),
            Patch(facecolor='red', label='Black Win'),
            Patch(facecolor='green', label='Draw')
        ]
        plt.legend(handles=legend_elements)
        
        # Add labels and title
        plt.xlabel('Game Number')
        plt.ylabel('Final Score (+ favors White, - favors Black)')
        plt.title('Final Position Evaluation Across All Training Games')
        plt.grid(True, alpha=0.3)
        
        # Add score values above/below each bar
        for i, score in enumerate(final_scores):
            y_pos = score + 0.5 if score >= 0 else score - 1.5
            plt.text(i + 1, y_pos, f"{score:.2f}", ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_game_evaluation(self, game_index=-1):
        """Plot both evaluation and material balance during a game
        
        Args:
            game_index: Index of the game to plot, defaults to the most recent game
        """
        if not self.game_history:
            print("No game history available.")
            return
        
        # Default to the last game if no index specified
        if game_index == -1:
            game_index = len(self.game_history) - 1
        
        # Make sure the index is valid
        if game_index < 0 or game_index >= len(self.game_history):
            print(f"Invalid game index. Must be between 0 and {len(self.game_history) - 1}.")
            return
        
        game = self.game_history[game_index]
        white_scores = game['white_scores']
        result = game['result']
        moves = game['moves']
        
        # Reconstruct the board positions to calculate material balance at each move
        board = chess.Board()
        material_balance = [0]  # Start with initial balanced position
        
        # Define piece values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Calculate initial material balance
        initial_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                              for piece_type, value in piece_values.items()) - \
                           sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                              for piece_type, value in piece_values.items())
        material_balance[0] = initial_material
        
        # Calculate material balance after each move
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            
            # Calculate material difference (positive = white advantage)
            white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                               for piece_type, value in piece_values.items())
            black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                               for piece_type, value in piece_values.items())
            
            material_balance.append(white_material - black_material)
        
        # Create the plot with two Y-axes
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Primary axis - Evaluation score
        move_numbers = list(range(1, len(white_scores) + 1))
        ax1.plot(move_numbers, white_scores, 'b-', linewidth=2, label='Evaluation')
        ax1.set_xlabel('Move Number')
        ax1.set_ylabel('Evaluation Score', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Secondary axis - Material balance
        ax2 = ax1.twinx()
        ax2.plot(range(len(material_balance)), material_balance, 'g-', linewidth=2, label='Material')
        ax2.set_ylabel('Material Balance', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Highlight significant material changes
        for i in range(1, len(material_balance)):
            if abs(material_balance[i] - material_balance[i-1]) >= 1:
                ax2.plot(i, material_balance[i], 'ro', markersize=6)
                
                # Annotate significant captures
                change = material_balance[i] - material_balance[i-1]
                if abs(change) >= 3:  # Only annotate major piece captures
                    piece_name = ""
                    if abs(change) >= 9:
                        piece_name = "Queen"
                    elif abs(change) >= 5:
                        piece_name = "Rook"
                    elif abs(change) >= 3:
                        piece_name = "Minor piece"
                    
                    ax2.annotate(f"{piece_name} ({change:+.0f})",
                                (i, material_balance[i]),
                                xytext=(0, 10 if change > 0 else -20),
                                textcoords="offset points",
                                ha='center',
                                fontsize=8,
                                arrowprops=dict(arrowstyle='->', color='red'))
        
        # Add background shading based on advantage
        for i in range(len(white_scores)-1):
            if white_scores[i] > 1.0:  # Significant white advantage
                plt.axvspan(i+1, i+2, alpha=0.1, color='blue')
            elif white_scores[i] < -1.0:  # Significant black advantage
                plt.axvspan(i+1, i+2, alpha=0.1, color='red')
        
        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Add title with game result
        plt.title(f"Game {game_index+1} Evaluation & Material Balance - Result: {result}")
        
        # Add a marker for the final position
        ax1.plot(len(white_scores), white_scores[-1], 'bo', markersize=8)
        ax2.plot(len(material_balance)-1, material_balance[-1], 'go', markersize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print a summary of material exchanges
        print("\nMaterial Exchange Summary:")
        print("-------------------------")
        significant_changes = [(i, material_balance[i] - material_balance[i-1]) 
                              for i in range(1, len(material_balance)) 
                              if abs(material_balance[i] - material_balance[i-1]) >= 1]
        
        if significant_changes:
            for move_num, change in significant_changes:
                side = "White" if change > 0 else "Black"
                print(f"Move {move_num}: {side} gained {abs(change):.1f} points of material")
        else:
            print("No significant material exchanges in this game.")

    

```

## File: constants.py

- Extension: .py
- Language: python
- Size: 769 bytes
- Created: 2025-05-19 19:22:39
- Modified: 2025-05-20 19:54:16

### Code

```python
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

```

## File: evaluation.py

- Extension: .py
- Language: python
- Size: 16692 bytes
- Created: 2025-05-19 19:20:50
- Modified: 2025-05-20 19:10:39

### Code

```python
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
    
    # Save original turn
    original_turn = board.turn
    
    # Calculate white's mobility
    if board.turn != chess.WHITE:
        board.push(chess.Move.null())  # Switch to white's turn
    white_mobility = len(list(board.legal_moves))
    
    # Calculate attacked squares by white
    white_attacked = set()
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_attacked.add(square)
    
    if board.turn != original_turn:
        board.push(chess.Move.null())  # Restore turn
    
    # Calculate black's mobility
    if board.turn != chess.BLACK:
        board.push(chess.Move.null())  # Switch to black's turn
    black_mobility = len(list(board.legal_moves))
    
    # Calculate attacked squares by black
    black_attacked = set()
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.BLACK, square):
            black_attacked.add(square)
            
    if board.turn != original_turn:
        board.push(chess.Move.null())  # Restore turn
    
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

def calculate_movement_range(board, color):
    """Calculate all squares a player of given color could move to in one move"""
    movement_squares = set()
    
    # Save original turn
    original_turn = board.turn
    
    # Set turn to the color we're calculating for
    if board.turn != color:
        board.push(chess.Move.null())
    
    # For each legal move, add the destination square
    for move in board.legal_moves:
        movement_squares.add(move.to_square)
    
    # Restore original turn if needed
    if original_turn != board.turn:
        board.push(chess.Move.null())
    
    return movement_squares

def format_score(score):
    """Format a score properly without special symbols for checkmate"""
    # We'll use a consistent numerical format for all scores
    # Even for large values that might represent checkmate
    if score > 0:
        return f"+{score:.2f}"
    else:
        return f"{score:.2f}"

```

## File: file_structure.txt

- Extension: .txt
- Language: plaintext
- Size: 2479 bytes
- Created: 2025-05-20 19:00:33
- Modified: 2025-05-20 19:00:33

### Code

```plaintext
# Chess AI Project File Structure Recommendation

Based on analyzing your script, I recommend breaking it down into the following files to improve organization and maintainability:

## 1. constants.py
- All constants: `PIECE_VALUES`
- UI color constants (`LIGHT_SQUARE_COLOR`, `DARK_SQUARE_COLOR`, etc.)
- `MAX_CACHE_SIZE`

## 2. board_utils.py
- `board_to_tensor()`, `boards_to_tensor_batch()`
- `get_valid_moves_cached()`, `get_valid_moves()`
- `get_move_uci()`
- `get_legal_moves_from_square()`, `get_secondary_moves()`
- `square_name_to_square()`
- `print_board()`
- Global caching variables: `EVAL_CACHE` and `CACHE_LOCK`

## 3. evaluation.py
- `fast_evaluate_position()`
- `categorize_moves()`, `select_weighted_moves()`
- `find_threatened_squares()`, `find_guarded_squares()`
- `calculate_attack_range()`, `calculate_movement_range()`
- `format_score()`

## 4. mcts.py
- `RussianDollMCTS` class
- `ParallelRussianDollMCTS` class

## 5. neural_network.py
- `ChessQNetwork` class
- `DQNAgent` class

## 6. chess_ai.py
- Core `OptimizedChessAI` class with all its methods:
  - `reset_board()`, `set_board_from_fen()`, `make_move()`
  - `get_best_move()`
  - `self_play_game()`, `train()`
  - `save_model()`, `load_model()`
  - `evaluate_elo_rating()`
  - All plotting and analysis methods

## 7. game_play.py
- `play_game()`, `get_human_move()`
- `visual_play_game()`, `visual_play_game_with_features()`
- `enhanced_human_move()`
- Game persistence: `save_game_to_pgn()`, `load_game_from_pgn()`, `list_saved_games()`

## 8. ui.py
- `create_visual_chess_board()`
- `ClickableChessBoard` class
- `NonClickableChessBoard` class
- All UI-related helper functions

## 9. menu.py
- `display_full_menu()`
- `handle_menu_selections()`
- `main()`

## 10. main.py
- Import everything and call `main()` function
- Set up common imports and random seeds

This organization groups related functionality together, making the codebase more manageable and easier to maintain. You'll need to adjust imports between these files to account for dependencies.

When implementing this structure, I recommend:
1. Create each file with proper imports
2. Start with constants.py since it has no dependencies
3. Implement board_utils.py and evaluation.py next
4. Continue with the other files in dependency order
5. Test each module individually when possible

Would you like me to help you implement any of these specific files?
```

## File: game_play.py

- Extension: .py
- Language: python
- Size: 15186 bytes
- Created: 2025-05-19 19:41:32
- Modified: 2025-05-20 20:01:39

### Code

```python
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

```

## File: LICENSE

- Extension: 
- Language: unknown
- Size: 1113 bytes
- Created: 2025-05-19 18:55:47
- Modified: 2025-05-19 18:55:47

### Code

```unknown
MIT License

Copyright (c) 2025 Turning out data tricks since 2006!

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## File: main.py

- Extension: .py
- Language: python
- Size: 2593 bytes
- Created: 2025-05-19 19:26:10
- Modified: 2025-05-21 06:29:27

### Code

```python
import numpy as np
import torch
import random
import chess
import os
import sys

# Set matplotlib backend BEFORE any other matplotlib imports
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend which is interactive

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Ensure modules in the current directory can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main menu function
from menu import main as menu_main

def setup_environment():
    """Setup the environment for the chess AI"""
    # Check if required packages are installed
    required_packages = ['numpy', 'torch', 'chess', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA not available. Using CPU for computation.")
        print("Training may be slow. Consider enabling CUDA if available.")
    
    # Create directories for saved data if they don't exist
    os.makedirs("saved_games", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    return device

def main():
    """Main entry point for the chess AI application"""
    try:
        # Setup the environment
        device = setup_environment()
        
        # Display welcome message
        print("\n" + "="*78)
        print(" "*20 + "Chess AI with Deep Q-Learning and MCTS" + " "*20)
        print("="*78 + "\n")
        
        # Start the application
        menu_main()
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        
        # In debug mode, show the full traceback
        if os.environ.get('DEBUG', '0') == '1':
            import traceback
            traceback.print_exc()
        
    print("\nExiting Chess AI. Goodbye!")

if __name__ == "__main__":
    main()
```

## File: mcts.py

- Extension: .py
- Language: python
- Size: 7223 bytes
- Created: 2025-05-19 19:16:29
- Modified: 2025-05-20 20:10:47

### Code

```python
import numpy as np
import chess
import math
import random
import torch
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque, OrderedDict
from evaluation import fast_evaluate_position, categorize_moves, select_weighted_moves
from board_utils import board_to_tensor

# Parallel Russian Doll MCTS implementation
# Modify ParallelRussianDollMCTS class to incorporate annealing
class ParallelRussianDollMCTS:
    def __init__(self, board, iterations=20, exploration_weight=1.0, samples_per_level=None, num_workers=None):
        self.board = board.copy()
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.samples_per_level = samples_per_level or [21, 13, 8, 5, 3, 2, 1]
        self.num_workers = num_workers or min(8, multiprocessing.cpu_count())
        
        # Thread-safe data structures
        self.Q = defaultdict(lambda: defaultdict(float))
        self.N = defaultdict(lambda: defaultdict(int))
        self.children = {}  # {state: [actions]}
        self.node_locks = defaultdict(threading.Lock)
        self.global_lock = threading.Lock()
        self.total_positions_evaluated = 0
        self._evaluation_lock = threading.Lock()
    
    def _select_action(self, state, board, training_progress=0.0, current_move=0, max_moves=200):
        """Select the best action from a state using UCB formula with annealing"""
        with self.node_locks[state]:
            actions = self.children.get(state)
            if not actions:
                return None
            
            # Apply annealing to UCB exploration parameter
            # More exploration early in training/games, less later
            alpha = 1 + training_progress  # 1 to 2
            beta = current_move / max_moves if max_moves > 0 else 0.5  # 0 to 1
            
            # Dynamic exploration weight that decreases with game progress and training progress
            effective_exploration = self.exploration_weight * (1.0 - 0.5 * alpha * beta)
            
            # Check for unexplored actions - always prioritize these
            unexplored = [a for a in actions if self.N[state][a] == 0]
            if unexplored:
                return random.choice(unexplored)
            
            # UCB selection with dynamic exploration parameter
            total_visits = sum(self.N[state][a] for a in actions)
            log_total = math.log(total_visits + 1e-10)
            
            ucb_scores = [
                (self.Q[state][a] / (self.N[state][a] + 1e-10)) +
                effective_exploration * math.sqrt(log_total / (self.N[state][a] + 1e-10))
                for a in actions
            ]
            
            return actions[np.argmax(ucb_scores)]
    
    def search(self, neural_net=None, device=None, training_progress=0.0, current_move=0, max_moves=200):
        """Parallel MCTS search with annealing parameters"""
        state = self.board.fen()
        
        # Initialize root if not seen before
        with self.global_lock:
            if state not in self.children:
                self._expand_node(state)
        
        # Parallel simulations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for _ in range(self.iterations):
                board_copy = self.board.copy()
                futures.append(executor.submit(
                    self._parallel_simulation,
                    board_copy,
                    neural_net,
                    device,
                    training_progress,
                    current_move,
                    max_moves
                ))
            
            # Wait for all simulations to complete
            for future in futures:
                future.result()
        
        # Select best move
        with self.global_lock:
            actions = self.children[state]
            if not actions:
                return None
            
            visit_counts = [self.N[state][a] for a in actions]
            return actions[np.argmax(visit_counts)]
            
    def _parallel_simulation(self, board, neural_net, device, training_progress=0.0, 
                            current_move=0, max_moves=200):
        """Run a single parallel MCTS simulation with annealing parameters"""
        states_actions = []
        current_depth = 0
        local_move = current_move
        
        # Selection and expansion phase
        while current_depth < len(self.samples_per_level):
            state = board.fen()
            
            with self.global_lock:
                if state not in self.children:
                    self._expand_node(state)
            
            # Select action with annealing parameters
            action = self._select_action(state, board, training_progress, local_move, max_moves)
            
            # No actions available or terminal state
            if not action:
                break
            
            # Execute action and record
            board.push(action)
            states_actions.append((state, action))
            current_depth += 1
            local_move += 1  # Increment local move counter
            
            # Stop if terminal state
            if board.is_game_over() or current_depth >= len(self.samples_per_level):
                break
        
        # Evaluate final position
        value = self._evaluate_position(board, neural_net, device)
        
        # Backpropagation
        for state, action in reversed(states_actions):
            with self.node_locks[state]:
                self.N[state][action] += 1
                self.Q[state][action] += (value - self.Q[state][action]) / self.N[state][action]
                value = -value  # Flip for opponent's perspective
    
    def _expand_node(self, state):
        """Expand a node in the tree"""
        board = chess.Board(state)
        categorized_moves, category_weights = categorize_moves(board)
        level_samples = min(
            self.samples_per_level[0],
            sum(len(moves) for moves in categorized_moves.values())
        )
        
        if level_samples > 0:
            self.children[state] = select_weighted_moves(categorized_moves, category_weights, level_samples)
        else:
            self.children[state] = []
    
    def _evaluate_position(self, board, neural_net, device):
        """Evaluate a board position"""
        with self._evaluation_lock:
            self.total_positions_evaluated += 1
            
            if board.is_checkmate():
                return -1.0 if board.turn == chess.WHITE else 1.0
            
            if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
                return 0.0
            
            if neural_net:
                with torch.no_grad():
                    board_tensor = board_to_tensor(board).unsqueeze(0)
                    if device:
                        board_tensor = board_tensor.to(device)
                    return neural_net(board_tensor).item()
            else:
                return math.tanh(fast_evaluate_position(board) / 10.0)

```

## File: menu.py

- Extension: .py
- Language: python
- Size: 23543 bytes
- Created: 2025-05-19 19:43:42
- Modified: 2025-05-21 06:57:09

### Code

```python
import chess
import matplotlib.pyplot as plt
from game_play import play_game, visual_play_game, visual_play_game_with_features, list_saved_games
from board_utils import print_board, get_move_uci
from chess_ai import OptimizedChessAI
from evaluation import fast_evaluate_position, format_score
from ui import NonClickableChessBoard
#from threaded_board import ThreadedChessBoard
from terminal_board import TerminalChessBoard

# Find the display_full_menu function and modify it to prevent repetition
def display_full_menu(chess_ai):
    """Display the full menu of options after the initial workflow"""
    print("\nChess AI Menu:")
    print("1. Play against AI (human as white)")
    print("2. Play against AI (human as black)")
    print("3. Watch AI play against itself")
    print("4. Train more")
    print("5. Continue training from current state")
    print("6. Save model")
    print("7. Load model")
    print("8. Set up custom position (FEN)")
    print("9. Toggle verbose output")
    print("10. Toggle visual board")
    print("11. Toggle enhanced features")
    print("12. Plot final game scores")
    print("13. Evaluate ELO rating")
    print("14. Exit")
    # Removed the infinite while loop that was here


# Update this function in the main code
def handle_menu_selections(chess_ai, verbose, use_visual_board, use_enhanced_features):
    """Handle menu selections and return updated feature flags"""

    # Main menu
    while True:
        display_full_menu(chess_ai)
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            if use_enhanced_features and use_visual_board:
                # Use the non-clickable board with enhanced features
                terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
                terminal_board.start()
            elif use_visual_board:
                visual_play_game(chess_ai, human_color=chess.WHITE)
            else:
                play_game(chess_ai, human_color=chess.WHITE)
        elif choice == '2':
            if use_enhanced_features and use_visual_board:
                # Use the non-clickable board with enhanced features
                threaded_board = ThreadedChessBoard(chess.Board(), chess_ai, human_color=chess.BLACK)
                terminal_board.start()
            elif use_visual_board:
                visual_play_game(chess_ai, human_color=chess.BLACK)
            else:
                play_game(chess_ai, human_color=chess.BLACK)
        elif choice == '3':
            # Watch AI play against itself
            ai1 = OptimizedChessAI(verbose=verbose)
            ai2 = OptimizedChessAI(verbose=verbose)
            
            # Load the same model for both if available
            try:
                ai1.load_model("chess_model.pth")
                ai2.load_model("chess_model.pth")
            except FileNotFoundError:
                print("Model file not found. Using untrained models.")
            
            # Create a single shared board
            shared_board = chess.Board()
            move_count = 0
            
            while not shared_board.is_game_over() and move_count < 200:
                print_board(shared_board)
                
                current_ai = ai1 if shared_board.turn == chess.WHITE else ai2
                ai_name = "White AI" if shared_board.turn == chess.WHITE else "Black AI"
                
                print(f"{ai_name} is thinking...")
                
                # Temporarily set the AI's board to the shared board
                current_ai.board = shared_board.copy()  # Use a copy to avoid reference issues
                move = current_ai.get_best_move()
                
                # Validate the move is legal before making it
                if move in shared_board.legal_moves:
                    shared_board.push(move)
                    move_count += 1
                    print(f"{ai_name} plays: {get_move_uci(move)}")
                else:
                    print(f"Error: {ai_name} attempted illegal move {get_move_uci(move)}. Stopping game.")
                    break
            
            print_board(shared_board)
            print(f"Game over. Result: {shared_board.result()}")
        # Other menu options remain the same
        elif choice == '4':
            # Fresh training
            num_games = int(input("Enter number of training games: "))
            chess_ai.training_games = num_games
            progress_interval = max(1, min(num_games // 10, 10))
            chess_ai.train(progress_interval=progress_interval)
            chess_ai.plot_training_progress()
        elif choice == '5':
            # Continue training from current state
            additional_games = int(input("Enter number of additional training games: "))
            original_games = chess_ai.training_games
            chess_ai.training_games += additional_games
            progress_interval = max(1, min(additional_games // 10, 10))
            
            print(f"Continuing training from game {len(chess_ai.game_history)+1} to {chess_ai.training_games}")
            chess_ai.train(progress_interval=progress_interval)
            chess_ai.plot_training_progress()
        elif choice == '6':
            filename = input("Enter filename to save model (default: chess_model.pth): ") or "chess_model.pth"
            chess_ai.save_model(filename)
        elif choice == '7':
            filename = input("Enter filename to load model (default: chess_model.pth): ") or "chess_model.pth"
            continue_training = input("Load training state as well? (y/n): ").lower() == 'y'
            try:
                chess_ai.load_model(filename, continue_training=continue_training)
                print("Model loaded successfully!")
            except FileNotFoundError:
                print(f"Model file {filename} not found.")
        elif choice == '8':
            fen = input("Enter FEN position (e.g., 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'): ")
            try:
                chess_ai.set_board_from_fen(fen)
                print("Position set successfully!")
                print_board(chess_ai.board)
                
                # Ask if user wants to play from this position
                play_choice = input("Do you want to play from this position? (y/n): ").lower()
                if play_choice == 'y':
                    human_color_choice = input("Play as white or black? (w/b): ").lower()
                    human_color = chess.WHITE if human_color_choice == 'w' else chess.BLACK
                    
                    if use_enhanced_features and use_visual_board:
                        terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
                        terminal_board.start()
                    elif use_visual_board:
                        visual_play_game(chess_ai, human_color=human_color)
                    else:
                        play_game(chess_ai, human_color=human_color)
                
            except ValueError:
                print("Invalid FEN position. Please try again.")
        elif choice == '9':
            verbose = not verbose
            chess_ai.verbose = verbose
            print(f"Verbose output {'enabled' if verbose else 'disabled'}")
        elif choice == '10':
            use_visual_board = not use_visual_board
            print(f"Visual board {'enabled' if use_visual_board else 'disabled'}")
        elif choice == '11':
            use_enhanced_features = not use_enhanced_features
            print(f"Enhanced features {'enabled' if use_enhanced_features else 'disabled'}")
            if use_enhanced_features and not use_visual_board:
                use_visual_board = True
                print("Visual board automatically enabled to support enhanced features.")
        elif choice == '12':
            chess_ai.plot_final_game_scores()
        elif choice == '13':
            print("\nELO Rating Evaluation")
            print("--------------------")
            print("This will play your model against Stockfish at various strength levels")
            print("to determine an approximate ELO rating.")
            print("Note: You need Stockfish installed on your system.")
            
            games_per_level = int(input("Number of games per ELO level (default: 20): ") or 20)
            starting_elo = int(input("Starting ELO to test (default: 1500): ") or 1500)
            
            estimated_elo = chess_ai.evaluate_elo_rating(
                num_games=games_per_level,
                starting_elo=starting_elo
            )
            
            if estimated_elo:
                print(f"\nYour model's estimated ELO rating: {estimated_elo}")
        elif choice == '14':
            print("Exiting the Chess AI program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 14.")
    
    # Return the updated feature flags
    return verbose, use_visual_board, use_enhanced_features


def main():
    print("Optimized Chess AI with Russian Doll MCTS and Deep Q-Learning")
    print("------------------------------------------------------------")
    
    # Create the AI with default settings
    chess_ai = OptimizedChessAI(training_games=20, verbose=False)
    
    # First, determine the user's primary goal
    print("\nWhat would you like to do?")
    print("1. Play chess against the AI")
    print("2. Train or improve the AI")
    print("3. Analyze AI performance")
    
    primary_goal = input("Enter your choice (1-3): ")
    
    # Default visual settings
    use_visual_board = True
    use_enhanced_features = True
    verbose = False
    
    # PLAY CHESS PATHWAY
    if primary_goal == '1':
        # Load a pre-trained model by default
        load_model = input("Would you like to load a pre-trained model? (y/n, default: y): ").lower() or 'y'
        if load_model == 'y':
            filename = input("Enter model filename (default: chess_model.pth): ") or "chess_model.pth"
            try:
                chess_ai.load_model(filename)
                print(f"Model loaded from {filename}")
            except FileNotFoundError:
                print(f"Model file {filename} not found. Using untrained model.")
        
        # Visual settings
        visual_board = input("Use visual board for gameplay? (y/n, default: y): ").lower() or 'y'
        use_visual_board = visual_board == 'y'
        
        enhanced_features = 'n'
        if use_visual_board:
            enhanced_features = input("Use enhanced features (score tracking, threat highlighting)? (y/n, default: y): ").lower() or 'y'
        use_enhanced_features = enhanced_features == 'y'
        
        # Choose color
        color_choice = input("Play as white or black? (w/b, default: w): ").lower() or 'w'
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        
        # Start the game
        if use_enhanced_features and use_visual_board:
            terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
            terminal_board.start()
        elif use_visual_board:
            visual_play_game_with_features(chess_ai, human_color=human_color)
        else:
            play_game(chess_ai, human_color=human_color)
    
    # TRAIN AI PATHWAY
    elif primary_goal == '2':
        # Ask about training from existing model
        continue_training = input("Continue training from an existing model? (y/n, default: n): ").lower() or 'n'
        
        if continue_training == 'y':
            filename = input("Enter model filename (default: chess_model.pth): ") or "chess_model.pth"
            try:
                chess_ai.load_model(filename, continue_training=True)
                print(f"Continuing training from {filename}")
            except FileNotFoundError:
                print(f"Model file {filename} not found. Starting with untrained model.")
        
        # Training parameters
        num_games = int(input("Enter number of training games (default: 20): ") or "20")
        chess_ai.training_games = num_games
        
        verbose = input("Enable verbose output during training? (y/n, default: y): ").lower() or 'y'
        chess_ai.verbose = verbose == 'y'
        verbose = chess_ai.verbose  # Update local variable for menu
        
        # Start training
        print(f"\nStarting training for {chess_ai.training_games} games...")
        progress_interval = max(1, min(chess_ai.training_games // 10, 10))
        chess_ai.train(progress_interval=progress_interval)
        
        # Training results are now shown automatically after training, including the final game scores plot
        
        # Save model option
        save_model = input("Save the trained model? (y/n, default: y): ").lower() or 'y'
        if save_model == 'y':
            filename = input("Enter filename (default: chess_model.pth): ") or "chess_model.pth"
            chess_ai.save_model(filename)
            print(f"Model saved to {filename}")
        
        # Play option after training
        play_after_training = input("Would you like to play against the trained AI? (y/n, default: y): ").lower() or 'y'
        if play_after_training == 'y':
            visual_board = input("Use visual board for gameplay? (y/n, default: y): ").lower() or 'y'
            use_visual_board = visual_board == 'y'
            
            enhanced_features = 'n'
            if use_visual_board:
                enhanced_features = input("Use enhanced features? (y/n, default: y): ").lower() or 'y'
            use_enhanced_features = enhanced_features == 'y'
            
            color_choice = input("Play as white or black? (w/b, default: w): ").lower() or 'w'
            human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
            
            if use_enhanced_features and use_visual_board:
                terminal_board = TerminalChessBoard(chess.Board(), chess_ai, human_color=chess.WHITE)
                terminal_board.start()
            elif use_visual_board:
                visual_play_game_with_features(chess_ai, human_color=human_color)
            else:
                play_game(chess_ai, human_color=human_color)
    
    # ANALYZE AI PATHWAY
    elif primary_goal == '3':
        # Load a model for analysis
        load_model = input("Load a specific model for analysis? (y/n, default: y): ").lower() or 'y'
        if load_model == 'y':
            filename = input("Enter model filename (default: chess_model.pth): ") or "chess_model.pth"
            try:
                chess_ai.load_model(filename)
                print(f"Model loaded from {filename}")
            except FileNotFoundError:
                print(f"Model file {filename} not found. Using untrained model.")
        
        # Analysis options
        print("\nAnalysis options:")
        print("1. Watch AI play against itself")
        print("2. Plot training progress")
        print("3. Plot final game scores")
        print("4. Evaluate ELO rating")
        print("5. Return to main menu")
        
        while True:
            analysis_choice = input("Choose an analysis option (1-5): ")
            
            if analysis_choice == '1':
                # Self-play
                print("\nAI vs AI game:")
                ai1 = chess_ai  # Use the loaded model
                ai2 = OptimizedChessAI(verbose=False)
                try:
                    ai2.load_model(filename)  # Use the same model for opponent
                except:
                    print("Using untrained model for opponent")
                
                # Visual options
                visual = input("Use visual board for self-play? (y/n, default: y): ").lower() or 'y'
                show_scores = input("Display and track scores during play? (y/n, default: y): ").lower() or 'y'
                
                # Create a shared board
                shared_board = chess.Board()
                move_count = 0
                
                # For score tracking
                white_scores = []
                black_scores = []
                
                # Setup score plot if requested
                if show_scores and visual == 'y':
                    plt.ion()  # Turn on interactive mode
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.set_title("Game Progress Scores")
                    ax.set_xlabel("Move Number")
                    ax.set_ylabel("Score")
                    ax.grid(True)
                    white_line, = ax.plot([], [], 'b-', label="White")
                    black_line, = ax.plot([], [], 'r-', label="Black")
                    ax.legend()
                    plt.show(block=False)
                
                # Game loop
                while not shared_board.is_game_over() and move_count < 200:
                    if visual == 'y':
                        print_board(shared_board)
                    
                    current_ai = ai1 if shared_board.turn == chess.WHITE else ai2
                    ai_name = "White AI" if shared_board.turn == chess.WHITE else "Black AI"
                    
                    print(f"{ai_name} is thinking...")
                    current_ai.board = shared_board.copy()
                    move = current_ai.get_best_move()
                    
                    if move in shared_board.legal_moves:
                        shared_board.push(move)
                        move_count += 1
                        
                        # Score evaluation and display
                        if show_scores:
                            raw_score = fast_evaluate_position(shared_board)
                            white_scores.append(raw_score)
                            black_scores.append(-raw_score)
                            
                            # Format scores for display
                            white_score_str = format_score(raw_score)
                            black_score_str = format_score(-raw_score)
                            
                            print(f"{ai_name} plays: {get_move_uci(move)} - White: {white_score_str} | Black: {black_score_str}")
                            
                            # Update plot if visual
                            if visual == 'y' and move_count % 3 == 0:  # Update every 3 moves for efficiency
                                move_numbers = list(range(1, len(white_scores) + 1))
                                white_line.set_data(move_numbers, white_scores)
                                black_line.set_data(move_numbers, black_scores)
                                
                                # Adjust plot limits
                                ax.set_xlim(0, max(move_count + 5, 50))
                                ax.set_ylim(min(min(white_scores), min(black_scores)) - 2, 
                                            max(max(white_scores), max(black_scores)) + 2)
                                
                                # Update plot
                                fig.canvas.draw_idle()
                                fig.canvas.flush_events()
                        else:
                            print(f"{ai_name} plays: {get_move_uci(move)}")
                    else:
                        print(f"Error: {ai_name} attempted illegal move. Stopping game.")
                        break
                
                # Final position
                if visual == 'y':
                    print_board(shared_board)
                
                # Display final result
                result = shared_board.result()
                print(f"Game over. Result: {result}")
                
                # Final scores
                if show_scores:
                    final_score = fast_evaluate_position(shared_board)
                    print(f"Final evaluation - White: {format_score(final_score)} | Black: {format_score(-final_score)}")
                    
                    # Plot final score distribution if requested
                    if input("Plot game score progression? (y/n): ").lower() == 'y':
                        plt.figure(figsize=(12, 6))
                        move_nums = list(range(1, len(white_scores) + 1))
                        plt.plot(move_nums, white_scores, 'b-', label="White")
                        plt.plot(move_nums, black_scores, 'r-', label="Black")
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        plt.grid(True, alpha=0.3)
                        plt.title("Score Progression During Game")
                        plt.xlabel("Move Number")
                        plt.ylabel("Evaluation Score")
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                
                # Close any open plots
                if show_scores and visual == 'y':
                    plt.ioff()
                    plt.close()
            
            elif analysis_choice == '2':
                # Plot training progress
                if hasattr(chess_ai, 'loss_history') and chess_ai.loss_history:
                    chess_ai.plot_training_progress()
                else:
                    print("No training data available. Train the model first.")
            
            elif analysis_choice == '3':
                # Plot final game scores
                if hasattr(chess_ai, 'game_history') and chess_ai.game_history:
                    chess_ai.plot_final_game_scores()
                else:
                    print("No game history available. Train the model first.")
            
            elif analysis_choice == '4':
                # ELO rating evaluation
                print("\nELO Rating Evaluation")
                print("This will play your model against Stockfish at various strength levels.")
                print("Note: You need Stockfish installed on your system.")
                
                games_per_level = int(input("Number of games per ELO level (default: 10): ") or "10")
                starting_elo = int(input("Starting ELO to test (default: 1500): ") or "1500")
                
                try:
                    estimated_elo = chess_ai.evaluate_elo_rating(
                        num_games=games_per_level,
                        starting_elo=starting_elo
                    )
                    if estimated_elo:
                        print(f"\nYour model's estimated ELO rating: {estimated_elo}")
                except Exception as e:
                    print(f"Error evaluating ELO rating: {e}")
                    print("Make sure Stockfish is installed and accessible.")
            
            elif analysis_choice == '5':
                break
            
            print()  # Add spacing between analyses
    
    # INVALID CHOICE
    else:
        print("Invalid choice. Please restart the program and select a valid option.")
    
    show_menu = input("\nWould you like to see the full menu of options? (y/n, default: n): ").lower() or 'n'
    if show_menu == 'y':
        # Now we call our menu handler which has its own loop
        verbose, use_visual_board, use_enhanced_features = handle_menu_selections(
            chess_ai, verbose, use_visual_board, use_enhanced_features
        )

if __name__ == "__main__":
    main()
```

## File: neural_network.py

- Extension: .py
- Language: python
- Size: 9788 bytes
- Created: 2025-05-19 19:19:12
- Modified: 2025-05-20 20:10:12

### Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import deque
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading
from board_utils import board_to_tensor
from mcts import ParallelRussianDollMCTS

# Simpler Neural Network
class ChessQNetwork(nn.Module):
    def __init__(self):
        super(ChessQNetwork, self).__init__()
        
        # Streamlined network architecture for faster inference
        # Input shape: [batch_size, 12, 8, 8]
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)  # Single output for value of position
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Output in range [-1, 1]
        
        return x

# DQN Agent Implementation
class DQNAgent:
    def __init__(self, gamma=0.99, epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.001,
                 batch_size=64):
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Q-Networks
        self.q_network = ChessQNetwork().to(self.device)
        self.target_q_network = ChessQNetwork().to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)  # replay buffer
        
        # Number of CPU cores for parallel processing
        self.num_cpu_cores = max(1, multiprocessing.cpu_count() - 1)
        
        # Clear evaluation cache
        global EVAL_CACHE
        EVAL_CACHE = {}

    def select_move_with_aha_learning(self, board, training_progress=0.0, is_training=True, undo_budget=3, eval_threshold=-1.5):
        """Efficient implementation of 'aha moment' learning"""
        if not is_training or undo_budget <= 0:
            # Regular move selection during gameplay or when out of undos
            return self.select_move(board, is_training=is_training)
        
        # Get current evaluation
        current_eval = fast_evaluate_position(board)
        
        # Standard move selection
        move = self.select_move(board, is_training=is_training)
        
        # Quick look-ahead to check if this move is a mistake
        test_board = board.copy()
        test_board.push(move)
        new_eval = fast_evaluate_position(test_board)
        
        # Calculate evaluation change from player's perspective
        eval_change = current_eval - new_eval if board.turn == chess.WHITE else new_eval - current_eval
        
        # If not a significant mistake, just return the original move
        if eval_change >= eval_threshold:
            return move
        
        # At this point, we've detected a significant mistake
        
        # Step 1: Create immediate learning signal (most important part)
        state_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        next_state_tensor = board_to_tensor(test_board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get current Q-value for this state-action pair
            current_q = self.q_network(state_tensor).item()
            
            # Get target value (immediate negative reward)
            target_q = -1.0  # Strong negative reward for mistake
        
        # Perform direct Q-value update (bypassing replay buffer for efficiency)
        self.optimizer.zero_grad()
        q_value = self.q_network(state_tensor)
        loss = F.mse_loss(q_value, torch.tensor([target_q], device=self.device))
        loss.backward()
        self.optimizer.step()
        
        # Step 2: Find a better alternative move
        better_moves = []
        for alt_move in board.legal_moves:
            if alt_move.uci() == move.uci():
                continue  # Skip the mistake move
                
            # Quick evaluation of alternative
            alt_board = board.copy()
            alt_board.push(alt_move)
            alt_eval = fast_evaluate_position(alt_board)
            
            # From player's perspective
            relevant_eval = current_eval - alt_eval if board.turn == chess.WHITE else alt_eval - current_eval
            
            # If this move is better than our threshold
            if relevant_eval > eval_threshold:
                better_moves.append((alt_move, relevant_eval))
        
        # If no better moves, return original despite the mistake
        if not better_moves:
            return move
        
        # Sort by evaluation (best first) and pick the best
        better_moves.sort(key=lambda x: x[1], reverse=True)
        better_move = better_moves[0][0]
        
        # Return the better move
        return better_move

    def update_target_network(self):
        """Copy weights from the Q-network to the target network"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def get_q_value(self, board):
        """Get the Q-value for a given board position"""
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state).item()
                
    def select_move(self, board, training_progress=0.0, is_training=False, current_move=0, max_moves=200):
        """
        Select a move using Russian Doll MCTS with neural network guidance
        """
        # For gameplay (not training), always use MCTS with no exploration
        if not is_training:
            training_progress = 1.0  # Force minimum exploration
            current_move = max_moves  # Force minimum exploration
        
        # Adjust MCTS iterations and parameters based on training progress
        base_iterations = int(50 + 150 * training_progress)  # 50 to 200 iterations
        exploration_weight = max(1.6 * (1 - 0.5 * training_progress), 0.8)
        
        # Adjust sampling width at each level based on training progress
        samples_per_level = [
            max(21, int(25 * (1 - 0.3 * training_progress))),  # Level 1
            max(13, int(15 * (1 - 0.3 * training_progress))),  # Level 2
            max(8, int(10 * (1 - 0.3 * training_progress))),   # Level 3
            max(5, int(7 * (1 - 0.3 * training_progress))),    # Level 4
            max(3, int(5 * (1 - 0.3 * training_progress))),    # Level 5
            max(2, int(3 * (1 - 0.3 * training_progress))),    # Level 6
            max(1, int(2 * (1 - 0.3 * training_progress)))     # Level 7
        ]
        
        # Use parallel MCTS with annealing parameters
        mcts = ParallelRussianDollMCTS(
            board, 
            iterations=base_iterations,
            exploration_weight=exploration_weight,
            samples_per_level=samples_per_level,
            num_workers=self.num_cpu_cores
        )
        
        return mcts.search(
            neural_net=self.q_network, 
            device=self.device,
            training_progress=training_progress,
            current_move=current_move,
            max_moves=max_moves
        )
        
    # In OptimizedChessAI class
    def get_best_move(self, training_progress=0.0, is_training=False):
        """Get the best move for the current position"""
        return self.dqn_agent.select_move(self.board, training_progress, is_training)
        
    def store_transition(self, state, move, reward, next_state, done):
        """Store a transition in the replay buffer"""
        state_tensor = board_to_tensor(state)
        next_state_tensor = board_to_tensor(next_state) if next_state else torch.zeros_like(state_tensor)
        self.memory.append((state_tensor, move, reward, next_state_tensor, done))
    
    def train(self):
        """Train the Q-network using a batch of experiences from the replay buffer"""
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples for training
        
        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.stack([s[0] for s in minibatch]).to(self.device)
        rewards = torch.tensor([s[2] for s in minibatch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([s[3] for s in minibatch]).to(self.device)
        dones = torch.tensor([s[4] for s in minibatch], dtype=torch.float32).to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(states).squeeze()
        
        # Get next Q-values from the target network
        next_q_values = self.target_q_network(next_states).squeeze().detach()
        
        # Calculate target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

```

## File: README.md

- Extension: .md
- Language: markdown
- Size: 44991 bytes
- Created: 2025-05-19 18:55:47
- Modified: 2025-05-21 07:00:27

### Code

```markdown
#python 3.10
python main.py

# chess-deep-q
Chess based on training a deep q learning model

1. constants.py

All constants: PIECE_VALUES
UI color constants (LIGHT_SQUARE_COLOR, DARK_SQUARE_COLOR, etc.)
MAX_CACHE_SIZE

2. board_utils.py

board_to_tensor(), boards_to_tensor_batch()
get_valid_moves_cached(), get_valid_moves()
get_move_uci()
get_legal_moves_from_square(), get_secondary_moves()
square_name_to_square()
print_board()
Global caching variables: EVAL_CACHE and CACHE_LOCK

3. evaluation.py

fast_evaluate_position()
categorize_moves(), select_weighted_moves()
find_threatened_squares(), find_guarded_squares()
calculate_attack_range(), calculate_movement_range()
format_score()

4. mcts.py

RussianDollMCTS class
ParallelRussianDollMCTS class

5. neural_network.py

ChessQNetwork class
DQNAgent class

6. chess_ai.py

Core OptimizedChessAI class with all its methods:

reset_board(), set_board_from_fen(), make_move()
get_best_move()
self_play_game(), train()
save_model(), load_model()
evaluate_elo_rating()
All plotting and analysis methods



7. game_play.py

play_game(), get_human_move()
visual_play_game(), visual_play_game_with_features()
enhanced_human_move()
Game persistence: save_game_to_pgn(), load_game_from_pgn(), list_saved_games()

8. ui.py

create_visual_chess_board()
ClickableChessBoard class
NonClickableChessBoard class
All UI-related helper functions

9. menu.py

display_full_menu()
handle_menu_selections()
main()

10. main.py

Import everything and call main() function
Set up common imports and random seeds

## Reqs

Chess

Training should be fast
  - I should be able to train 20 example games in about 30 minutes overall (using cuda)
  -	If this means we need to truncate games at move depth and count points
  - 'continue' training (i.e. load an existing model and start training from there)

Future feature
- You should evaluate the ELO level of your model, play it against stockfish and then dial tune the stockfish to find the threshold where your model begins to lose. That way you can find the correct ELO level for it.

Well thought out menu
	Think of user use cases with this system
	
	loading model
		training
			continued
		playing
		evaluating
			stockfish
	
	Update the code to have the model ask to save csv after saving the model state, default is y for both

Requirements:
  - Python
  - Uses the python-chess library for move generation and validation
  - MCTS to efficiently explore game tree
    - sample score weighted permutations until checkmate
	  - see scoring method below
	  - should also allow for non score purely random move (outside of the scored potential moves)
	    - this will ensure the game isn't only learning the rule (symbolic) based policy
    - possible moves
    - handles exponential explosion of move possibilities
  - Reinforcement Learning: State-action-reward triplets for Q-learning or policy gradients
    - deep [q learning] neural network to guide the search (similar to AlphaZero's approach)
	  - Differentiates between phyrric victory and not
	- learns over mcts selected for permutations of possibilities rather than exhaustive
	- learns within games based on score
	- final outcome is score * checkmate status (interaction).
	- 2 opportunities to learn (from both sides).
      - the game learn effectively from two vantage points (each player is like it’s own ‘view’ of the q-learning process.  I.e. what moves were favorable to one side vs disfavorable to the other, and this rotates back and forth).  This should simply be an index operation on the tensor of past moves (i.e. odd, even).
  - Implements a scoring hueristic
  - tensor of board
    - game as a set of 2d tensor's of board positions extended with moves
      - a time series sequence to sequence problem (best path of moves)
	- CUDA
	  - Parallelization
	  - mcts search using descending phi ratio’s could be done in parallel somehow (i.e. top 21 moves running parallel 13 moves and those 13 running parallel 8, etc)
  - uses python libraries for valid chess moves
- ask an llm to do the deep learning over the q agent for me
- that part is fuzzy.  I just know standard q learning is exponentially expensive.
- q learning objective (loss function?):
    - checkmate opponent in the least amount of moves (min)
    - with max score
- Status bar showing # of moves iterated over in current game
- Cap trainng steps (i.e. evaluted moves) per game at n (default 100, -1 is no cap), that's 50 white, 50 black.
- After training, when playing the game, use a visual map of the chess board
- Board
  - use a common python library
Playing
- threats are highlighted in light red on the board
- when a player inputs a piece coordinate for a move, when the player inputs the first two characters (would be nice if they merely type it in, vs hit enter).  the piece is highlighted in yellow, and all available moves are highlighted in green, secondary moves (two moves out) are highlighted in light orange
- how what pieces are guarded, this could be light green
- what areas are under threat should be a light red (not just what pieces are under threat, but what areas if moved into would create a threat immediately due to those areas being guarded).  I suppose on the flip side, we should also what areas are guarded (not just what pieces).  So we should extend the light green guarded pieces logic to light green guarded areas, and light red would be areas under threat (to include pieces under threat).  that would make things much simpler.
- have the board clickable be clickable in addition to the ability to input move coordinates
- the ability to save a game
- continue from a saved game
- user can request a move hint (as if assuming the policy model), we can call this 'policy model hint'
- white should always be on the bottom
- the ability to undo a move (such as making a mistake such as releasing a click too early, or more pragmatically didn't see a blind spot)
- score tracking
  - what I would like to see is the score for each player printed each game move, left padded up to thousands place if in that format, else as a percent
  - also would like to see a plot of the scores for player a and b inbetween each game turn (fine to update the same plot rather than a new plot).
 - and finally, at the end of the training, have a plot of all final scores (including draws) of each game plotted over game iterations.

better, but I'm rethinking these variations on threatened pieces/spaces.

I think red should simply mean any place an opponent's piece could take in 1 move (space or piece)

and light red is within 2 moves

from a players perspective, movement should be green, light green 2 steps out

so how do we do both (contested areas?).  I would think we simply half/half the shades (if we can split diagonally).


	```
	# 1. Add a method to format and display scores
	def format_score(score):
		"""Format a score as either left-padded decimal or percentage"""
		if abs(score) > 1.0:
			# Format as decimal with thousands padding
			return f"{abs(score):04.2f}"
		else:
			# Format as percentage
			return f"{abs(score)*100:05.2f}%"

	# 2. Modify the self_play_game method to display scores and track them for plotting
	def self_play_game(self, max_moves=200):
		"""Play a game against itself and store the transitions for learning"""
		self.reset_board()
		move_count = 0
		game_moves = []
		
		# For score tracking and plotting
		white_scores = []
		black_scores = []
		
		# Setup for interactive plotting
		if self.verbose:
			plt.ion()  # Turn on interactive mode
			fig, ax = plt.subplots(figsize=(10, 6))
			ax.set_title("Game Progress Scores")
			ax.set_xlabel("Move Number")
			ax.set_ylabel("Score")
			ax.grid(True)
			white_line, = ax.plot([], [], 'b-', label="White")
			black_line, = ax.plot([], [], 'r-', label="Black")
			ax.legend()
			plt.show(block=False)
		
		# Calculate training progress (0 to 1)
		training_progress = min(len(self.game_history) / self.training_games, 1.0)
		
		# Display setup
		if self.verbose:
			print(f"\rGame {len(self.game_history)+1}: Move 0/{max_moves}", end="", flush=True)
		
		while not self.board.is_game_over() and move_count < max_moves:
			# Remember the state before the move
			state_before = self.board.copy()
			
			# Select a move with advanced annealing
			move = self.get_best_move(
				training_progress, 
				is_training=True,
				current_move=move_count,
				max_moves=max_moves
			)
			
			if move is None:
				break
				
			self.make_move(move)
			game_moves.append(get_move_uci(move))
			move_count += 1
			
			# Evaluate the position and display scores
			raw_score = fast_evaluate_position(self.board)
			normalized_score = math.tanh(raw_score / 10.0)
			
			# Store scores for plotting
			white_scores.append(raw_score)
			black_scores.append(-raw_score)
			
			# Display formatted scores
			white_score_str = format_score(raw_score)
			black_score_str = format_score(-raw_score)
			
			# Update display with scores
			if self.verbose:
				print(f"\rGame {len(self.game_history)+1}: Move {move_count}/{max_moves} - White: {white_score_str} | Black: {black_score_str}", end="", flush=True)
				
				# Update the score plot every few moves
				if move_count % 5 == 0 or self.board.is_game_over():
					# Update plot data
					move_numbers = list(range(1, len(white_scores) + 1))
					white_line.set_data(move_numbers, white_scores)
					black_line.set_data(move_numbers, black_scores)
					
					# Adjust plot limits
					ax.set_xlim(0, max(move_count + 5, 50))
					ax.set_ylim(min(min(white_scores), min(black_scores)) - 2, 
								max(max(white_scores), max(black_scores)) + 2)
					
					# Update plot
					fig.canvas.draw_idle()
					fig.canvas.flush_events()
			
			# Get the reward
			if self.board.is_checkmate():
				reward = -1.0
				done = True
			elif self.board.is_stalemate() or self.board.is_insufficient_material():
				reward = 0.0
				done = True
			else:
				# Use evaluation function for reward signal
				eval_score = fast_evaluate_position(self.board)
				reward = 0.01 * math.tanh(eval_score / 10.0)  # Normalize and scale down
				done = False
			
			# Store the transition
			self.dqn_agent.store_transition(state_before, move, reward, self.board.copy(), done)
			
			# Train the network more frequently but with smaller batches
			if len(self.dqn_agent.memory) >= self.dqn_agent.batch_size:
				loss = self.dqn_agent.train()
				self.loss_history.append(loss)
		
		# Turn off interactive mode after game
		if self.verbose:
			plt.ioff()
			plt.close()
		
		# Store game result and evaluation
		result = self.board.result()
		final_evaluation = fast_evaluate_position(self.board)
		
		# Store the scores history for this game
		self.game_history.append({
			'moves': game_moves,
			'result': result,
			'move_count': move_count,
			'white_scores': white_scores,
			'black_scores': black_scores,
			'final_score': final_evaluation
		})
		
		self.evaluation_history.append(final_evaluation)
		self.epsilon_history.append(self.dqn_agent.epsilon)
		self.move_count_history.append(move_count)
		
		# Display final scores
		if self.verbose:
			print(f"\nFinal scores - White: {format_score(final_evaluation)} | Black: {format_score(-final_evaluation)}")
		
		# Periodically update the target network
		if len(self.game_history) % 5 == 0:
			self.dqn_agent.update_target_network()
			
			# Clear evaluation cache periodically to avoid memory bloat
			global EVAL_CACHE
			EVAL_CACHE = {}
		
		return result, move_count

	# 3. Add a method to plot the final game scores after training
	def plot_final_game_scores(self):
		"""Plot the final scores for all games after training"""
		if not self.game_history:
			print("No game history available. Please train the AI first.")
			return
		
		plt.figure(figsize=(12, 8))
		
		# Extract data from game history
		game_numbers = list(range(1, len(self.game_history) + 1))
		final_scores = [game['final_score'] for game in self.game_history]
		results = [game['result'] for game in self.game_history]
		
		# Create color map based on game results
		colors = []
		for result in results:
			if result == '1-0':  # White win
				colors.append('blue')
			elif result == '0-1':  # Black win
				colors.append('red')
			else:  # Draw or unfinished
				colors.append('green')
		
		# Plot the scores
		plt.bar(game_numbers, final_scores, color=colors)
		
		# Add a horizontal line at zero
		plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
		
		# Create a custom legend
		from matplotlib.patches import Patch
		legend_elements = [
			Patch(facecolor='blue', label='White Win'),
			Patch(facecolor='red', label='Black Win'),
			Patch(facecolor='green', label='Draw')
		]
		plt.legend(handles=legend_elements)
		
		# Add labels and title
		plt.xlabel('Game Number')
		plt.ylabel('Final Score')
		plt.title('Final Scores Across All Training Games')
		plt.grid(True, alpha=0.3)
		
		# Add score values above each bar
		for i, score in enumerate(final_scores):
			plt.text(i + 1, score + (1 if score >= 0 else -1), 
					 f"{score:.2f}", ha='center')
		
		plt.tight_layout()
		plt.show()

	# 4. Modify the train method to call the new plotting function
	def train(self, progress_interval=10):
		"""Train the AI through self-play with optimized performance"""
		print(f"Training for {self.training_games} games...")
		start_time = time.time()
		
		# Create a progress bar for the training
		with tqdm(total=self.training_games, desc="Training Progress") as pbar:
			for i in range(self.training_games):
				result, moves = self.self_play_game()
				
				# Update progress bar
				pbar.update(1)
				pbar.set_postfix({
					'Game': f"{i+1}/{self.training_games}",
					'Result': result,
					'Moves': moves,
					'Epsilon': f"{self.dqn_agent.epsilon:.4f}"
				})
				
				# Print detailed stats at intervals
				if (i + 1) % progress_interval == 0:
					elapsed = time.time() - start_time
					
					print(f"\nGame {i+1}/{self.training_games} completed in {elapsed:.2f}s")
					print(f"Result: {result}, Moves: {moves}")
					
					if self.loss_history:
						recent_loss = self.loss_history[-min(50, len(self.loss_history)):]
						avg_loss = sum(recent_loss) / len(recent_loss)
						print(f"Average loss: {avg_loss:.4f}")
					
					print(f"Exploration rate (epsilon): {self.dqn_agent.epsilon:.4f}")
					
					avg_moves = sum(self.move_count_history[-min(10, len(self.move_count_history)):]) / \
								min(10, len(self.move_count_history))
					print(f"Average moves per game: {avg_moves:.1f}")
					
					# Estimate time remaining
					time_per_game = elapsed / (i + 1)
					remaining_games = self.training_games - (i + 1)
					eta = remaining_games * time_per_game
					print(f"Estimated time remaining: {eta:.2f}s")
					print(f"Average time per game: {time_per_game:.2f}s")
		
		# Final stats
		total_time = time.time() - start_time
		print("\nTraining completed!")
		print(f"Total training time: {total_time:.2f}s")
		print(f"Average time per game: {total_time / self.training_games:.2f}s")
		
		# Plot the final game scores
		self.plot_final_game_scores()

	# 5. Update the main menu to include the final scores plot option
	def main():
		# ... existing code ...
		
		# Main menu
		while True:
			print("\nChess AI Menu:")
			print("1. Play against AI (human as white)")
			print("2. Play against AI (human as black)")
			print("3. Watch AI play against itself")
			print("4. Train more")
			print("5. Save model")
			print("6. Load model")
			print("7. Set up custom position (FEN)")
			print("8. Toggle verbose output")
			print("9. Toggle visual board")
			print("10. Plot final game scores")  # New option
			print("11. Exit")
			
			choice = input("Enter your choice (1-11): ")
			
			# ... existing code for options 1-9 ...
			
			elif choice == '10':
				chess_ai.plot_final_game_scores()
			elif choice == '11':
				print("Exiting the Chess AI program. Goodbye!")
				break
			else:
				print("Invalid choice. Please enter a number between 1 and 11.")

	if __name__ == "__main__":
		main()
	```

Advanced agent learning
	hybrid approach that combines elements of online learning, regret minimization, and exploitation-focused sampling. It's quite similar to how strong human players learn - they might make a move, immediately see it was a mistake, mentally "take it back," and then make a better move.

	Undo mechanics
		
	do you think adding the ability for the deep q learning system to have the ability to do up to n undo moves makes sense to help the policy model 'learn'.  I mean... tbh, ai/policy model when playing shouldn't have the ability to do an 'undo', but if the policy model can learn (i.e. apply the calculating loss of point lesson from their move), and improve upon the search space by essentially doubling there search options (former possible moves + a new set of moves), but also only allow the policy model to undo say 3 moves an entire game.

	One this is all theory, two I'd prefer the proposed change to be as 'light' as possible, and finally, I'm not sure if it's even necessary for a deep q learning which will eventually see a majority of the potential move action space from basically training.  But this would allow for a dynamic way for the model to have an 'aha' learning moment by say having a heuristic (i.e. no agents) that when game points suffer a dramatic shift, that one of the 'undo' moves would be triggered, and the deep q learning agent would learn from their most recent 'mistake' and be able to undo it to essentially double their move option space (i.e. take former moves + a new set of moves derived that exclude the former initial moves, i.e. mask them out of the possible move space, then the agent does a 'double' take on the possible moves they can decide upon).

	Of course these aha moments would be one 'trigger', but sometimes I find myself wanting to undo a move when I made a mistake, not necessarily lost a lot of points.  This is more complex, as internally I'm looking into the future I was hoping for, and a mistake cost me that loss in game points I was expecting to achieve.  Mathematically, this would involve me forecasting (inferring) future move state spaces and their associated points and a mistake had me realize I lost those potential points... so this is more advanced as it involves calculating future move states and their associated points (something WE ARE ALREADY DOING more or less / essentially with the phi based mcts).
	
	the beauty about the aha moment is it only needs to be a 1 move deep learning lesson

	once the move is made and the points calculated (which the deep q learning agent is basically ALREADY doing with the phi based forecasting), the agent can be like 'whoops, I didn't mean to suffer those losses.

	So in effect, we don't really need the ability to backtrack, but rather an option to expand the search space... well actually, the agent isn't using a greedy policy, but rather is using a sampled policy, so the agent can still make a 'mistake', but it's still an informed mistake (simply based on probability), but would allow the agent to 'undo' a 'bad move' more or less, so maybe I am back to what I was proposing since we are sampling and not doing greedy (I was thinking we could just have a cutoff).

	That's it.  If an agent randomly selects a move that would result in a loss of points more than a certain threshold, this triggers a lesson learned for q learning on what that move state would have resulted in, and the ability to do an 'undo', but the undo has to be valuable in terms of offering a better move.  Basically the agent would only select a move with a score higher than the one they are undoing from the new concatenated sampling space (prior move space (masked) + new thought out move space (remaining unmasked), and the probability space is limited to just moves above the former probability (thereby avoiding a secondary undo).  So this 'undo' mechanism would trigger automatically as long as a player hasn't done an undo already in that turn, and has available undo's.  The key here though is to let the q learning agent learn from this 'mistake' immediately, and then follow-up with a new state sampling from this updated search space.

	I'm not even sure what to call this, but it seems like a type of learning I've done during chess, kind of a mix between monte carlo sampling and genetic algorithms (evolving) to mimick learning from trial and error.

	I suppose we would need to account for when there is no move above such a loss threshold.  For example.  If an undo is triggered, and the new state is just the same scores or worse... we can avoid this problem efficiently by ensuring that the threshold is hit, if there were no other moves available above that threshold.  Then we can double the search space, but if still no available move is better (e.g. such as late game situations with only 2 or 3 pieces left)... then we are stuck with that move.

	I just realized I have a similar issue with the mistake rule checking.  If a change in game state results in too big of a jump, we second guess the sampling and try again.  However late game situations might make this impossible to not double check (up to 3 checks total per game, no more than one per turn).  However, if a move would result in a check, this doesn't necessarily mean the following moves are suboptimal, this might be a necessary situation to get to a more advantageous game state.  This would require knowledge of following states.  I suppose the litmus (heuristic) test here would be if the resulting end game state (score of the final phi layer) is not of a lower game state than the initial, then we don't need to 'double guess' ourselves

ascii piece_symbols (monospace) = {
	'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
	'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'

Scoring
	UCB Score
		also, this isn't useful scoring imo.  The point of scoring is to track the state of the board (pieces, positions, advantage, etc), using some common standard scoring metric like UCB (UBC?).
	
	https://chessify.me/blog/chess-engine-evaluation
	
	I think the engine would be better if we did q learning on sampling from mcts where higher scores 

	think of it as a rag retrieval and these are these scores are the sparse embeddings but we treat this like a sampler (a mcts sampler). 

	generate
		21 possible moves
			derive scores (to include material, coordination, etc)
			weighted sample 13
			repeat thru the rest of phi
			
			this should help the model choose every possible move, while reducing the overall space of the computation

	Scoring as embeddings sampler for bootstrapping a non naive policy and/or random (not sure which one to use it for, but I figured using scoring vs pure random will explore better moves and the policy will update faster
					
	Material: the number and value of pieces on the board
	Mobility: the ability of pieces to move around the board
	King safety: how vulnerable the kings are
	Pawn structure: the configuration of pawns on the board
	Control of space: which side controls more of the board
	Piece coordination: how well the pieces work together

	I think counting unthreatened pieces as well as enemy threatened pieces towards the active player as a benefit makes sense. Piece mobility (i.e. board control into unthreatened areas). King safety such as free moves. All of this can be 1 move deep. Idk how to track pawn structure and or piece coordination?
	
	I wish there was a way to more intelligently pick what pieces to explore moves from rather than simply 25 random initial moves. Maybe give preference to certain classes of pieces and maybe a strategy to bifurcate between non threatened vs threatened. For example if the game is known to aim for certain strategies. Then weight rank those strategies for random selection (such as addressing a high material  point threatened piece, so maybe that piece is initially weighted higher for mcts explorations)

	ex.
	```
	def calculate_piece_coordination_score(board):
	# Evaluates:
	# 1. Defended pieces (+1 per defended piece)
	# 2. Attacks on opponent's pieces (+1 per attack)
	# 3. Development of minor pieces (+2 per developed minor)
	# 4. Rooks on open files (+2 per rook on open file)

	# Defended pieces: Non-pawn pieces defended by friendly pieces
	# Minor piece development: Knights/bishops moved from starting squares
	# Open files: Files with no pawns, good for rook placement

	def calculate_threats_score(board):
	# Evaluates:
	# 1. Hanging pieces (pieces that can be captured favorably)
	# 2. Available checks (+0.5 per check)

	# For hanging pieces, considers:
	# - Whether the piece is under attack
	# - Whether it's adequately defended
	# - The value of the piece minus half the value of the smallest attacker

	total_score = (
	material_score * 1.0 +          # Material (base weight)
	mobility_score * 0.3 +          # Mobility
	king_safety_score * 0.5 +       # King safety (important)
	pawn_structure_score * 0.4 +    # Pawn structure
	space_control_score * 0.3 +     # Space control
	piece_coordination_score * 0.4 + # Piece coordination
	threats_score * 0.6 +           # Tactical threats (important)
	check_score                     # Check bonus/penalty
	)

	# improvement upon prior code

	def categorize_moves(board):
	"""Categorize moves by tactical significance and assign sampling weights"""
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
	# ... categorization logic ...

	return categorized_moves, category_weights

	# Inside categorize_moves function
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
	```
	
tracking material point scores
I would think on another axis having this shown as well within that same plot would be useful.

what about other scores?  Does it make sense to plot those inbetween games as well (as well as printing at the end of all games)?

when I said at the end 'the printing over all games', this should be a single time series plot my end of game metrics (just as we are plotting scores over a game iteration, we take the final values, and plot these across training iterations across games) (i.e. one point aggregated from each game), this will hopefully show how trends are improving over long term training objectives are reached (new minima's, maxima's, etc)

I DON'T WANT TO SEE OVERRIDDEN SCORES FROM CHECKMATES.  Give me the score minus any checkmate modification.  Some relative score that is comparable across all games.

I have concerns the model was trained with these more or less 'jury rigged' scores (i.e. the ones I just spoke about) as they don't really offer a meaningful comparison.  The score without any checkmate modification is the final game signal.  That's what we are tracking, not some 'this + this value = end of game'.  I'm not sure how to express this, but if we think of this as a tensor problem, we are simply measuring what the conditions were at an end of game condition (i.e. analogous to a text prompt/chunk)

	I likely inadvertently removed the checkmate reward signal while doing mcts searches.  The scoring is to help navigate search space, but what really matters is checkmating the opponent.
	
	the above 2 pars
		I DON'T WANT ... a text prompt/chunk)
	
	I think I made a mistake in removing the checkmate status in scoring.  While yes that's the score I want to see during end of game, but for q learning.  It's likely needed to exemplify end of game conditions and highly sought for and not based on score alone (else q learning is learning to mimic underlying score minus chekmate status).

	So when searching through the phi layered mcts approach, if a checkmate is found, that should be a sought after goal, and my concern is this feature 



Efficient MCTS: scaling descent russian doll mcts
	Methods to speed things up
	normal q learning involves free exploration vs optimal path.  the purpose of the scores as embedding sampler is to give the q learning a leg up in learning faster (to act as a policy) while the model develops it's own policy but doesn't start from scratch.  I would think the basics would still be random selection vs optimal policy.
	
	say
	the original code was calculating points 25 moves in from mcts and choosing the best options as a default move.  the q learning would simply learn from this move.
	
	idk, maybe I'm over analyzing this, but I figured the space is exponential

	so I was trying to figure out a way to pair down the computational graph, but I'd prefer a more abstract deep learning approach if I could, but mcts is the best idea I have mixed with a sampled sparse embedding approach (i.e. scores with mcts).

	then these moves are made and the q-learning model updated.

	...
	Why Your Russian Doll MCTS Approach Is Superior

	Guided Exploration vs. Pure Randomness

	Your approach: Systematically explores ~25 moves at each level, with each move evaluated and sampled based on promise
	Random approach: Would waste computational resources on clearly inferior moves

	Depth Efficiency

	Your approach: "21 randomly possible next moves → 13 from each of those → 8 from 13 → 5 from 8 → 3 from 5 → 2 from 3 → 1 from 2"
	This creates a narrowing funnel that concentrates computation where it matters
	Pure random exploration might never reach promising lines

	Bootstrapping Effect

	The comprehensive evaluation function we've built (material, mobility, king safety, etc.) provides excellent guidance
	Early in training, this evaluation serves as domain knowledge to jumpstart learning
	The MCTS sampling based on this evaluation is much more informative than random moves

	Knowledge Transfer

	Your approach allows for transferring chess knowledge through the evaluation function and search process
	Random play would force the DQN to learn basic chess principles from scratch

	Implementation Changes
	Here's what I would modify in the code to fully embrace your Russian doll MCTS approach:

	```

	pythondef select_move(self, board, training_progress=0.0):
		"""
		Always use MCTS for move selection, but with dynamic iterations
		based on training progress and position complexity
		"""
		# Calculate complexity of position
		complexity = self.estimate_position_complexity(board)
		
		# Determine MCTS iterations based on training progress and complexity
		# Early in training: more iterations (exploration)
		# Later in training: fewer iterations (exploitation of learned policy)
		base_iterations = 300 if training_progress < 0.3 else (
						  200 if training_progress < 0.7 else 100)
		
		# Add more iterations for complex positions
		iterations = int(base_iterations * (0.8 + 0.4 * complexity))
		
		# Always use MCTS with the calculated number of iterations
		return self.select_move_with_mcts(board, iterations, training_progress)
	And modify the epsilon decay to apply to MCTS exploration parameter instead:
	pythondef select_move_with_mcts(self, board, iterations=100, training_progress=0.0):
		"""
		Select a move using MCTS guided by the hybrid evaluation function
		"""
		# Adjust UCB exploration weight based on training progress
		# Higher early in training, lower later
		exploration_weight = max(1.6 * (1 - 0.5 * training_progress), 0.8)
		
		# Adjust sampling width at each level based on training progress
		# More samples early in training, fewer later
		samples_per_level = [
			max(25, int(30 * (1 - 0.3 * training_progress))),  # Level 1
			max(15, int(20 * (1 - 0.3 * training_progress))),  # Level 2
			max(10, int(13 * (1 - 0.3 * training_progress))),  # Level 3
			max(7, int(10 * (1 - 0.3 * training_progress))),   # Level 4
			max(5, int(7 * (1 - 0.3 * training_progress))),    # Level 5
			max(3, int(5 * (1 - 0.3 * training_progress))),    # Level 6
			max(2, int(3 * (1 - 0.3 * training_progress)))     # Level 7
		]
		
		# Create a modified MCTS that uses the narrowing sample approach
		mcts = RussianDollMCTS(board, iterations=iterations, 
							  exploration_weight=exploration_weight,
							  samples_per_level=samples_per_level)
		return mcts.search()
	And then implement the RussianDollMCTS class to handle the narrowing exploration pattern:
	pythonclass RussianDollMCTS(MCTS):
		def __init__(self, root_board, iterations=1000, exploration_weight=1.41, samples_per_level=None):
			super().__init__(root_board, iterations, exploration_weight)
			# Default narrowing pattern if not specified
			self.samples_per_level = samples_per_level or [25, 13, 8, 5, 3, 2, 1]
		
		def simulate(self, board, depth=0):
			"""
			Enhanced simulation that uses narrowing samples at each level
			"""
			if depth >= len(self.samples_per_level) or board.is_game_over():
				# End of sampling depth or terminal position
				if board.is_checkmate():
					return 0.0 if board.turn == chess.WHITE else 1.0
				return 0.5  # Draw
			
			# Get all legal moves
			legal_moves = list(board.legal_moves)
			if not legal_moves:
				return 0.5  # No legal moves, draw
			
			# Number of moves to sample at this level
			samples = min(self.samples_per_level[depth], len(legal_moves))
			
			# Score each move
			scored_moves = []
			for move in legal_moves:
				next_board = board.copy()
				next_board.push(move)
				# Use our comprehensive evaluation
				score = evaluate_position(next_board)
				scored_moves.append((move, score))
			
			# Sort by score (optimal for current player)
			scored_moves.sort(key=lambda x: x[1], reverse=(board.turn == chess.WHITE))
			
			# Take top samples
			top_moves = [move for move, _ in scored_moves[:samples]]
			
			# Select a random move from top candidates
			move = random.choice(top_moves)
			next_board = board.copy()
			next_board.push(move)
			
			# Recursive simulation to next level
			result = self.simulate(next_board, depth + 1)
			
			# Flip result since we're alternating perspectives
			return 1.0 - result
	```
	Advantages Over Epsilon-Greedy Approach
	The traditional epsilon-greedy approach used in Q-learning doesn't map well to chess because:

	Pure random exploration is wasteful: Chess has ~35 legal moves on average, and most are terrible
	Low information signal: Random moves lead to poor positions that provide little useful learning
	Sparse rewards: Checkmates are rare events in random play, making learning slow

	Your Russian doll MCTS approach fixes all these issues by concentrating exploration on promising branches while still allowing for sufficient variety to discover new strategies.
	Conclusion
	I completely agree with your insight. The Russian doll MCTS approach will indeed almost always help rather than using purely random moves. It provides a perfect balance between:

	Exploration: Sampling multiple paths at each level
	Exploitation: Using evaluation to guide which paths to explore
	Efficiency: Concentrating computation on promising variations
	Knowledge transfer: Leveraging chess principles through evaluation functions

	This approach is particularly well-suited to chess, which has an enormous state space but also strong heuristics about what constitutes good play. The combination of MCTS sampling guided by your comprehensive evaluation function creates a powerful bootstrapping mechanism for the DQN to learn much faster than it could with random exploration.
	
Annealling
	I'd like to anneal the chance of random 

Future features
	# Chess AI System Revision Requirements

	## Strategic Evaluation Enhancements

	1. **Position-Adaptive Evaluation Weights**
	   - Implement dynamic evaluation weights that adjust based on position characteristics (e.g., open/closed position, material imbalance)
	   - Add a position classifier that can recognize key position types and modify evaluation parameters accordingly

	2. **Strategic Pattern Recognition**
	   - Implement detection of common chess patterns (e.g., bishops of opposite colors, IQP positions, minority attacks)
	   - Add evaluation terms for important strategic themes (e.g., piece quality, color complex weaknesses, compensation for material)
	   - Create a database of strategic patterns with corresponding evaluation adjustments

	3. **Phase-Specific Evaluation**
	   - Develop separate evaluation functions for opening, middlegame, and endgame
	   - Implement phase detection to smoothly transition between evaluation functions
	   - Apply different piece values and positional weights based on game phase
	   - Integrate specialized endgame evaluations (e.g., king activity becomes more important)

	## Search and Learning Improvements

	4. **Enhanced MCTS Implementation**
	   - Replace Python threads with multiprocessing for true parallel search
	   - Implement batched MCTS with vectorized operations for GPU acceleration
	   - Add progressive move widening to dynamically adjust search breadth based on position complexity
	   - Implement threat detection and extension for horizon effect mitigation

	5. **Advanced Reinforcement Learning**
	   - Replace basic Q-learning with TD(λ) or other temporal difference methods with eligibility traces
	   - Implement prioritized experience replay with importance sampling
	   - Add curriculum learning progression (start with endgames, advance to middlegames and openings)
	   - Develop a more nuanced reward function that captures practical winning chances

	6. **Hybrid Learning Approach**
	   - Add supervised learning pre-training using a database of master games
	   - Implement self-play with expert iteration to combine MCTS and neural guidance
	   - Create a validation system using classic chess puzzles to test tactical understanding

	## Technical Optimizations

	7. **Neural Network Architecture Refactoring**
	   - Replace the current CNN with a more modern architecture using residual connections
	   - Reduce fully-connected layer size using 1×1 convolutions to maintain spatial information
	   - Add batch normalization for faster training convergence
	   - Implement optional model quantization for inference acceleration

	8. **Memory Management Enhancements**
	   - Replace dictionary cache with a proper LRU cache implementation
	   - Implement Zobrist hashing for efficient position representation
	   - Add prefetch functionality for likely next positions
	   - Implement incremental evaluation updates to avoid full recalculation

	9. **Optimized Batch Processing**
	   - Standardize batch processing throughout the system
	   - Reduce device transfer overhead with larger, less frequent transfers
	   - Create a dedicated GPU evaluation pipeline for position batches
	   - Add adaptive batch sizing based on hardware capabilities

	10. **Performance Profiling Framework**
		- Add comprehensive performance profiling capabilities
		- Implement logging of training metrics, search statistics, and system resource usage
		- Create a benchmarking system for comparing versions and configurations
		- Add A/B testing capability for evaluating system changes

	## User Experience Enhancements

	11. **Improved Game Analysis**
		- Add visualization of the AI's "thought process" during move selection
		- Implement move explanation that describes strategic and tactical considerations
		- Create position evaluation breakdowns showing contribution of different factors
		- Add a feature to analyze user games with commentaries

	12. **Training Management Interface**
		- Develop a dashboard for monitoring training progress with key metrics
		- Add checkpointing and resumable training with configurable parameters
		- Implement automatic hyperparameter tuning
		- Create visualization of neural network learning progress

	13. **Adaptable Playing Strength**
		- Add configurable playing strength levels that go beyond simple search depth adjustment
		- Implement personality profiles with different strategic preferences
		- Create a progressive learning mode that adapts to the user's skill level
		- Add coaching features that suggest alternative moves and explain mistakes

	## Implementation Priorities

	**High Priority:**
	- Position-adaptive evaluation weights (#1)
	- Phase-specific evaluation (#3)
	- Enhanced MCTS implementation (#4)
	- Neural network architecture refactoring (#7)
	- Memory management enhancements (#8)

	**Medium Priority:**
	- Strategic pattern recognition (#2)
	- Advanced reinforcement learning (#5)
	- Optimized batch processing (#9)
	- Performance profiling framework (#10)
	- Improved game analysis (#11)

	**Lower Priority:**
	- Hybrid learning approach (#6)
	- Training management interface (#12)
	- Adaptable playing strength (#13)

	These requirements aim to significantly improve both the chess understanding and computational efficiency of the system while maintaining a balance between immediate practical improvements and longer-term architectural enhancements.
```

## File: terminal_board.py

- Extension: .py
- Language: python
- Size: 17956 bytes
- Created: 2025-05-21 06:54:45
- Modified: 2025-05-21 06:54:45

### Code

```python
import chess
import colorama
import os
import time
from colorama import Fore, Back, Style
from evaluation import fast_evaluate_position, find_threatened_squares, find_guarded_squares
from board_utils import get_legal_moves_from_square, get_secondary_moves

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
        
        # Print turn and evaluation at the top
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        eval_prefix = "+" if evaluation > 0 else ""
        
        print(f"\n{turn} to move | Eval: {eval_prefix}{evaluation:.2f}")
        print("  " + "-" * 33)
        
        # Unicode chess pieces
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
            '.': ' '
        }
        
        # Display board - with white at the bottom
        for rank in range(7, -1, -1):
            print(f"{rank+1} |", end=" ")
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                square_symbol = piece_symbols[piece.symbol()] if piece else piece_symbols['.']
                
                # Determine background color based on square type and highlights
                bg_color = Back.LIGHTBLACK_EX if (file + rank) % 2 == 1 else Back.BLACK
                
                # Apply highlights
                if self.selected_square == square:
                    bg_color = Back.YELLOW
                elif square in self.possible_moves:
                    bg_color = Back.GREEN
                elif square in self.secondary_moves:
                    bg_color = Back.MAGENTA
                elif self.last_move and (square == self.last_move.from_square or square == self.last_move.to_square):
                    bg_color = Back.BLUE
                elif square in threatened_squares:
                    bg_color = Back.RED
                elif square in guarded_squares:
                    bg_color = Back.CYAN
                
                # Determine text color based on piece color
                if piece:
                    text_color = Fore.WHITE if piece.color == chess.WHITE else Fore.BLACK
                else:
                    text_color = Fore.WHITE
                
                # Print the square
                print(f"{bg_color}{text_color}{square_symbol} {Style.RESET_ALL}", end="")
            print("|")
        
        print("  " + "-" * 33)
        print("    a  b  c  d  e  f  g  h")
        
        # Legend
        print("\nHighlight Legend:")
        print(f"{Back.YELLOW}     {Style.RESET_ALL} Selected piece   ", end="")
        print(f"{Back.GREEN}     {Style.RESET_ALL} Possible moves   ", end="")
        print(f"{Back.MAGENTA}     {Style.RESET_ALL} Secondary moves")
        print(f"{Back.BLUE}     {Style.RESET_ALL} Last move        ", end="")
        print(f"{Back.RED}     {Style.RESET_ALL} Threatened        ", end="")
        print(f"{Back.CYAN}     {Style.RESET_ALL} Guarded")
        
        # Available commands
        print("\nCommands: 'hint', 'undo', 'save', 'load', 'resign', 'cancel', or enter move (e.g., e2e4)")
    
    def process_input(self, user_input):
        """Process user input for a move or command"""
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
            return False
        
        # Handle square selection (e.g., "e2")
        if len(command) == 2 and self.selected_square is None:
            # Parse the square
            file_char, rank_char = command
            if file_char.isalpha() and rank_char.isdigit():
                file_idx = ord(file_char) - ord('a')
                rank_idx = int(rank_char) - 1
                
                if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                    square = chess.square(file_idx, rank_idx)
                    piece = self.board.piece_at(square)
                    
                    if piece and piece.color == self.human_color:
                        self.selected_square = square
                        self.possible_moves = get_legal_moves_from_square(self.board, square)
                        self.secondary_moves = get_secondary_moves(self.board, square)
                        print(f"Selected {command}. Enter destination or another command.")
                        return False
                    else:
                        print("No piece of yours at that square.")
                        return False
            
            print("Invalid square notation. Please use format like 'e2'.")
            return False
        
        # Handle complete move (e.g., "e2e4")
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
        
        # Handle destination after selecting a piece
        if len(command) == 2 and self.selected_square is not None:
            # Parse the destination
            file_char, rank_char = command
            if file_char.isalpha() and rank_char.isdigit():
                file_idx = ord(file_char) - ord('a')
                rank_idx = int(rank_char) - 1
                
                if 0 <= file_idx < 8 and 0 <= rank_idx < 8:
                    dest_square = chess.square(file_idx, rank_idx)
                    
                    if dest_square in self.possible_moves:
                        move = chess.Move(self.selected_square, dest_square)
                        
                        # Check for promotion
                        piece = self.board.piece_at(self.selected_square)
                        if piece.piece_type == chess.PAWN:
                            if (self.human_color == chess.WHITE and rank_idx == 7) or \
                               (self.human_color == chess.BLACK and rank_idx == 0):
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
            
            print("Invalid destination. Please use format like 'e4'.")
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
```

## File: threaded_board.py

- Extension: .py
- Language: python
- Size: 3933 bytes
- Created: 2025-05-21 06:28:20
- Modified: 2025-05-21 06:29:35

### Code

```python
import queue
import threading
import time
import chess
import matplotlib.pyplot as plt
from ui import NonClickableChessBoard
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend

class ThreadedChessBoard(NonClickableChessBoard):
    def __init__(self, board, ai, human_color=chess.WHITE):
        super().__init__(board, ai, human_color)
        self.ui_queue = queue.Queue()
        self.running = True
        self.ui_thread = None
        
    def _ui_thread_function(self):
        """Run matplotlib in its own thread"""
        plt.ion()  # Interactive mode
        
        # Initial board setup
        self.update_board()
        
        # If AI goes first (human is black), let AI make first move
        if self.human_color == chess.BLACK:
            # Signal the main thread to make AI move
            self.ui_queue.put(("ai_move", None))
        
        # UI thread loop
        while self.running:
            try:
                # Process any pending UI updates
                if self.fig is None:
                    self.update_board()
                
                # Keep the UI responsive
                plt.pause(0.1)
            except Exception as e:
                print(f"UI thread error: {e}")
                break
                
        # Clean up
        plt.close('all')
        
    def start(self):
        """Start the chess game with proper threading"""
        print("Interactive Chess Game (Threaded Version)")
        print("----------------------------------------")
        print("Enter coordinates to select pieces and make moves.")
        print("Available commands:")
        print("- hint: Get a move suggestion")
        print("- undo: Take back your last move")
        print("- save: Save the current game")
        print("- load: Load a previously saved game")
        print("- resign: Resign the current game")
        print("- cancel: Cancel your current piece selection")
        
        # Start UI thread
        self.ui_thread = threading.Thread(target=self._ui_thread_function)
        self.ui_thread.daemon = True
        self.ui_thread.start()
        
        # Main game loop (in main thread)
        game_over = False
        while not game_over and self.running:
            try:
                # Check for messages from UI thread
                try:
                    message_type, data = self.ui_queue.get(block=False)
                    if message_type == "ai_move":
                        self.make_ai_move()
                        self.ui_queue.task_done()
                except queue.Empty:
                    pass
                
                # Human's turn
                if self.board.turn == self.human_color:
                    move_made = self.process_input()
                    
                    if move_made:
                        # Check if game is over
                        if self.board.is_game_over():
                            game_over = True
                            continue
                        
                        # If not game over, AI makes its move
                        self.make_ai_move()
                        
                        # Check again if game is over
                        if self.board.is_game_over():
                            game_over = True
                
                # Small delay to prevent CPU overuse
                time.sleep(0.1)
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Game error: {e}")
                break
        
        # Clean up
        self.running = False
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=1.0)  # Wait for UI thread to exit
        
        print("Game finished.")
```

## File: ui.py

- Extension: .py
- Language: python
- Size: 28483 bytes
- Created: 2025-05-19 19:42:41
- Modified: 2025-05-21 06:28:42

### Code

```python
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


```

