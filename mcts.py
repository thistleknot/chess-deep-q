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

# Optimized Parallel Russian Doll MCTS implementation
class ParallelRussianDollMCTS:
    def __init__(self, board, iterations=20, exploration_weight=1.0, samples_per_level=None, num_workers=None):
        self.board = board.copy()  # Keep one master copy
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
        
        # Pre-compute root state to avoid repeated FEN calls
        self.root_state = self.board.fen()
    
    def _select_action(self, state, training_progress=0.0, current_move=0, max_moves=200):
        """Select the best action from a state using UCB formula with annealing"""
        # Note: Removed board parameter since we don't use it in selection logic
        with self.node_locks[state]:
            actions = self.children.get(state)
            if not actions:
                return None
            
            # Apply annealing to UCB exploration parameter
            alpha = 1 + training_progress  # 1 to 2
            beta = current_move / max_moves if max_moves > 0 else 0.5  # 0 to 1
            
            # Dynamic exploration weight
            effective_exploration = self.exploration_weight * (1.0 - 0.5 * alpha * beta)
            
            # Check for unexplored actions - always prioritize these
            unexplored = [a for a in actions if self.N[state][a] == 0]
            if unexplored:
                return random.choice(unexplored)
            
            # UCB selection with dynamic exploration parameter
            total_visits = sum(self.N[state][a] for a in actions)
            if total_visits == 0:
                return random.choice(actions)
                
            log_total = math.log(total_visits + 1e-10)
            
            ucb_scores = [
                (self.Q[state][a] / (self.N[state][a] + 1e-10)) +
                effective_exploration * math.sqrt(log_total / (self.N[state][a] + 1e-10))
                for a in actions
            ]
            
            return actions[np.argmax(ucb_scores)]
        
    def search(self, neural_net=None, device=None, training_progress=0.0, current_move=0, max_moves=200):
        """Parallel MCTS search with simplified board management"""
        # Initialize root if not seen before
        with self.global_lock:
            if self.root_state not in self.children:
                self._expand_node(self.root_state)
        
        # Parallel simulations - each gets its own board copy
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for _ in range(self.iterations):
                # Each simulation gets a fresh copy
                board_copy = self.board.copy()
                
                futures.append(executor.submit(
                    self._parallel_simulation,
                    board_copy,  # Fresh copy for each simulation
                    neural_net,
                    device,
                    training_progress,
                    current_move,
                    max_moves
                ))
            
            # Wait for all simulations to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"MCTS simulation error: {e}")
        
        # Select best move based on visit counts
        with self.global_lock:
            actions = self.children.get(self.root_state, [])
            if not actions:
                # Fallback: return a random legal move
                legal_moves = list(self.board.legal_moves)
                return random.choice(legal_moves) if legal_moves else None
            
            visit_counts = [self.N[self.root_state][a] for a in actions]
            if all(count == 0 for count in visit_counts):
                # If no actions were visited, return random
                return random.choice(actions)
            
            best_action_idx = np.argmax(visit_counts)
            return actions[best_action_idx]

    # REPLACE the _parallel_simulation method in mcts.py with this SAFER version:

    def _parallel_simulation(self, board, neural_net, device, training_progress=0.0, 
                            current_move=0, max_moves=200):
        """Run a single MCTS simulation with safe board management"""
        states_actions = []  # Store (state, action) pairs for backpropagation
        simulation_board = board.copy()  # Work on a copy to avoid issues
        current_depth = 0
        local_move = current_move
        
        # Selection and expansion phase
        while current_depth < len(self.samples_per_level):
            state = simulation_board.fen()
            
            # Expand node if necessary
            with self.global_lock:
                if state not in self.children:
                    self._expand_node(state)
            
            # Select action with annealing parameters
            action = self._select_action(state, training_progress, local_move, max_moves)
            
            # No actions available or terminal state
            if not action or action not in simulation_board.legal_moves:
                break
            
            # Record state-action pair for backpropagation
            states_actions.append((state, action))
            
            # Execute action
            simulation_board.push(action)
            current_depth += 1
            local_move += 1
            
            # Stop if terminal state or max depth reached
            if simulation_board.is_game_over() or current_depth >= len(self.samples_per_level):
                break
        
        # Evaluate final position
        value = self._evaluate_position(simulation_board, neural_net, device)
        
        # Backpropagation - update Q values (no need to undo moves since we used a copy)
        for state, action in reversed(states_actions):
            with self.node_locks[state]:
                self.N[state][action] += 1
                # Running average update
                old_q = self.Q[state][action]
                self.Q[state][action] = old_q + (value - old_q) / self.N[state][action]
                value = -value  # Flip for opponent's perspective
    
    def _expand_node(self, state):
        """Expand a node in the tree with optimized move generation"""
        if state in self.children:
            return  # Already expanded
            
        # Create board from state only when necessary
        board = chess.Board(state)
        
        # Early termination check
        if board.is_game_over():
            self.children[state] = []
            return
        
        # Use optimized move categorization
        categorized_moves, category_weights = categorize_moves(board)
        total_available_moves = sum(len(moves) for moves in categorized_moves.values())
        
        if total_available_moves == 0:
            self.children[state] = []
            return
        
        # Determine sample size for this level
        level_samples = min(self.samples_per_level[0], total_available_moves)
        
        if level_samples > 0:
            selected_moves = select_weighted_moves(categorized_moves, category_weights, level_samples)
            self.children[state] = selected_moves
        else:
            self.children[state] = []
    
    def _evaluate_position(self, board, neural_net, device):
        """Evaluate a board position with caching optimization"""
        with self._evaluation_lock:
            self.total_positions_evaluated += 1
        
        # Fast terminal position evaluation
        if board.is_checkmate():
            return -1.0 if board.turn == chess.WHITE else 1.0
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
            
        # Check for threefold repetition (expensive, so do it last)
        if board.is_repetition(3):
            return 0.0
        
        # Neural network evaluation
        if neural_net is not None:
            try:
                with torch.no_grad():
                    board_tensor = board_to_tensor(board).unsqueeze(0)
                    if device is not None:
                        board_tensor = board_tensor.to(device)
                    value = neural_net(board_tensor).item()
                    # Clamp to reasonable range
                    return max(-1.0, min(1.0, value))
            except Exception as e:
                print(f"Neural network evaluation error: {e}")
                # Fallback to heuristic evaluation
        
        # Fallback to heuristic evaluation
        try:
            heuristic_value = fast_evaluate_position(board)
            return math.tanh(heuristic_value / 10.0)  # Normalize to [-1, 1]
        except Exception as e:
            print(f"Heuristic evaluation error: {e}")
            return 0.0  # Neutral evaluation as last resort