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

    
