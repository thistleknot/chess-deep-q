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

class DQNAgent:
    def __init__(self, gamma=0.99, epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.001,
                 batch_size=64, use_aha_learning=False):
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # AHA Learning parameters
        self.use_aha_learning = use_aha_learning
        self.aha_budget_per_game = 3  # Max aha moments per game
        self.aha_budget_remaining = 3
        self.aha_threshold = -1.5  # Trigger when eval drops by this much
        
        if self.use_aha_learning:
            print("AHA Learning enabled - AI can correct mistakes during training")
        
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
            return self.select_move(board, training_progress, is_training=is_training)
        
        # Get current evaluation
        from evaluation import fast_evaluate_position
        current_eval = fast_evaluate_position(board)
        
        # Standard move selection (call the regular MCTS logic directly to avoid recursion)
        # For gameplay (not training), always use MCTS with no exploration
        if not is_training:
            training_progress = 1.0  # Force minimum exploration
            current_move = 200  # Force minimum exploration
        
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
        from mcts import ParallelRussianDollMCTS
        mcts = ParallelRussianDollMCTS(
            board, 
            iterations=base_iterations,
            exploration_weight=exploration_weight,
            samples_per_level=samples_per_level,
            num_workers=self.num_cpu_cores
        )
        
        move = mcts.search(
            neural_net=self.q_network, 
            device=self.device,
            training_progress=training_progress,
            current_move=0,
            max_moves=200
        )
        
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
        print(f"AHA! Detected potential mistake (eval change: {eval_change:.2f})")
        
        # Step 1: Create immediate learning signal (most important part)
        from board_utils import board_to_tensor
        state_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
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
            print("No better alternatives found, keeping original move")
            return move
        
        # Sort by evaluation (best first) and pick the best
        better_moves.sort(key=lambda x: x[1], reverse=True)
        better_move = better_moves[0][0]
        
        # Decrement the budget since we used an AHA moment
        self.aha_budget_remaining -= 1
        print(f"AHA moment used! Corrected {move.uci()} → {better_move.uci()}. Remaining budget: {self.aha_budget_remaining}")
        
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
        Optionally uses AHA learning for mistake correction during training
        """
        # Use AHA learning if enabled, during training, and budget available
        if (hasattr(self, 'use_aha_learning') and self.use_aha_learning and 
            is_training and hasattr(self, 'aha_budget_remaining') and 
            self.aha_budget_remaining > 0 and current_move > 5):  # Don't trigger in opening
            
            return self.select_move_with_aha_learning(
                board, training_progress, is_training, 
                self.aha_budget_remaining, self.aha_threshold
            )
        
        # Regular MCTS move selection (existing logic)
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
