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
