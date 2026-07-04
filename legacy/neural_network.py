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
import chess  # CRITICAL: Required for chess.WHITE in AHA learning
from board_utils import board_to_tensor, EVAL_CACHE  # Need EVAL_CACHE for cache clearing
from annealing import AnnealingSchedule
from constants import (
    GAMMA, LEARNING_RATE, BATCH_SIZE, REPLAY_CAPACITY,
    AHA_BUDGET_PER_GAME, AHA_THRESHOLD,
)


# NOTE (see spec/rl-categorization.spec.md): despite the historical name, this is a state-VALUE
# network V(s) -- a single scalar value head, NOT action-value Q(s,a). There is no per-action head
# and no max_a in training; moves are chosen by MCTS. It is the value-critic half of an
# AlphaZero-lite system (TD-Gammon lineage), not literal DQN. The name is kept for compatibility.
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
    """Value-critic agent (name kept for compatibility). See spec/rl-categorization.spec.md.

    Classification, qualified: OFF-POLICY (replay buffer, uncorrected), ONLINE (self-generated
    self-play, not offline), VALUE-BASED (V(s) critic, no policy gradient, not actor-critic),
    TD(0) bootstrap with a target network, MODEL-FREE learning + known-model MCTS planning.
    It is DQN-*family* in its stability tricks but NOT action-value Q-learning and NOT SARSA/PPO.
    """

    def __init__(self, gamma=GAMMA, epsilon=0.1, epsilon_min=0.1, epsilon_decay=0.995,
                 learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, use_aha_learning=False):
        self.gamma = gamma  # discount factor
        # epsilon is retained only for checkpoint/plot compatibility; the behaviour policy is
        # MCTS (structured exploration), so epsilon never drives move selection.
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Annealing schedule: single source of truth for prior->learned coefficients.
        self.schedule = AnnealingSchedule()
        # Argmax root move from the most recent search, exposed for difficulty regret scoring.
        self.last_root_best = None

        # AHA Learning parameters
        self.use_aha_learning = use_aha_learning
        self.aha_budget_per_game = AHA_BUDGET_PER_GAME  # Max aha moments per game
        self.aha_budget_remaining = AHA_BUDGET_PER_GAME
        self.aha_threshold = AHA_THRESHOLD  # Trigger when eval drops by this much

        if self.use_aha_learning:
            print("AHA Learning enabled - AI can correct mistakes during training")

        # Q-Networks
        self.q_network = ChessQNetwork().to(self.device)
        self.target_q_network = ChessQNetwork().to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=REPLAY_CAPACITY)  # replay buffer
        
        # Number of CPU cores for parallel processing
        self.num_cpu_cores = max(1, multiprocessing.cpu_count() - 1)
        
        # Clear evaluation cache
        global EVAL_CACHE
        EVAL_CACHE = {}

    def select_move_with_aha_learning(self, board, training_progress=0.0, is_training=True, undo_budget=3, eval_threshold=-1.5):
        """Efficient implementation of 'aha moment' learning"""
        if not is_training or undo_budget <= 0:
            # Regular move selection during gameplay or when out of undos
            return self._get_mcts_move(board, training_progress, 0, 200)
        
        # Get current evaluation
        from evaluation import fast_evaluate_position
        current_eval = fast_evaluate_position(board)
        
        # Get the initial move from MCTS
        initial_move = self._get_mcts_move(board, training_progress, 0, 200)
        
        # Quick look-ahead to check if this move is a mistake
        test_board = board.copy()
        test_board.push(initial_move)
        new_eval = fast_evaluate_position(test_board)
        
        # Calculate evaluation change from current player's perspective
        eval_change = (current_eval - new_eval) if board.turn == chess.WHITE else (new_eval - current_eval)
        
        # If not a significant mistake, return the original move
        if eval_change >= eval_threshold:  # eval_threshold is negative (e.g., -1.5)
            return initial_move
        
        # At this point, we've detected a significant mistake
        print(f"AHA! Detected potential mistake (eval drop: {eval_change:.2f})")
                
        # Step 1: Create immediate learning signal
        from board_utils import board_to_tensor
        state_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)

        # Perform direct Q-value update (immediate negative feedback)
        self.optimizer.zero_grad()
        q_value = self.q_network(state_tensor)
        target_q = torch.tensor([[-1.0]], device=self.device)  # FIX: Changed from [-1.0] to [[-1.0]] to match shape [1, 1]
        loss = F.mse_loss(q_value, target_q)
        loss.backward()
        self.optimizer.step()
        
        # Step 2: Find a better alternative move
        better_moves = []
        for alt_move in board.legal_moves:
            if alt_move.uci() == initial_move.uci():
                continue  # Skip the mistake move
                
            # Quick evaluation of alternative
            alt_board = board.copy()
            alt_board.push(alt_move)
            alt_eval = fast_evaluate_position(alt_board)
            
            # Calculate evaluation change from current player's perspective
            alt_eval_change = (current_eval - alt_eval) if board.turn == chess.WHITE else (alt_eval - current_eval)
            
            # If this move is better than our threshold (less negative)
            if alt_eval_change >= eval_threshold:
                better_moves.append((alt_move, alt_eval_change))
        
        # If no better moves found, return original despite the mistake
        if not better_moves:
            print("No better alternatives found, keeping original move")
            return initial_move
        
        # Sort by evaluation change (best first) and pick the best
        better_moves.sort(key=lambda x: x[1], reverse=True)
        best_alternative = better_moves[0][0]
        
        # Decrement the budget since we used an AHA moment
        self.aha_budget_remaining -= 1
        print(f"AHA moment used! Corrected {initial_move.uci()} → {best_alternative.uci()}. Remaining budget: {self.aha_budget_remaining}")
        
        return best_alternative
    
    def _get_mcts_move(self, board, training_progress, current_move, max_moves, temperature=0.0,
                       iterations=None):
        """Run Russian Doll MCTS with schedule-driven annealing; return the selected move.

        All progress-driven parameters come from the AnnealingSchedule (single source of truth).
        temperature controls root selection strength (0 = argmax); the argmax move is cached in
        self.last_root_best for difficulty regret scoring. iterations, when given, overrides the
        progress-scaled search budget (used for fast ELO calibration).
        """
        base_iterations = iterations if iterations is not None else int(50 + 150 * training_progress)
        exploration_weight = self.schedule.mcts_exploration_weight(training_progress)
        samples_per_level = self.schedule.samples_per_level(training_progress)
        learned_leaf_weight = self.schedule.learned_leaf_weight(training_progress)
        prior_bias_temperature = self.schedule.prior_bias_temperature(training_progress)

        # Use parallel MCTS with annealing parameters
        from legacy.mcts import ParallelRussianDollMCTS
        mcts = ParallelRussianDollMCTS(
            board,
            iterations=base_iterations,
            exploration_weight=exploration_weight,
            samples_per_level=samples_per_level,
            num_workers=self.num_cpu_cores,
            learned_leaf_weight=learned_leaf_weight,
            prior_bias_temperature=prior_bias_temperature,
        )

        move = mcts.search(
            neural_net=self.q_network,
            device=self.device,
            training_progress=training_progress,
            current_move=current_move,
            max_moves=max_moves,
            temperature=temperature,
        )
        self.last_root_best = mcts.best_action
        return move

    def update_target_network(self):
        """Copy weights from the Q-network to the target network"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def get_q_value(self, board):
        """State VALUE V(s) for a position (not action-value Q; see spec/rl-categorization)."""
        state = board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state).item()

    # Truthful alias per the categorization spec; get_q_value is retained for compatibility.
    get_value = get_q_value

    def select_move(self, board, training_progress=0.0, is_training=False, current_move=0,
                    max_moves=200, temperature=0.0, iterations=None):
        # At inference the model is mature: use full schedule progress so the leaf blend favours
        # the learned network rather than the prior (training passes its real progress).
        effective_progress = training_progress if is_training else 1.0

        # Use AHA learning if enabled, during training, and budget available
        if (hasattr(self, 'use_aha_learning') and self.use_aha_learning and
            is_training and hasattr(self, 'aha_budget_remaining') and
            self.aha_budget_remaining > 0 and current_move > 5):

            return self.select_move_with_aha_learning(
                board, effective_progress, is_training,
                self.aha_budget_remaining, self.aha_threshold
            )

        # Regular MCTS move selection (temperature drives difficulty at inference)
        return self._get_mcts_move(board, effective_progress, current_move, max_moves,
                                   temperature=temperature, iterations=iterations)

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
