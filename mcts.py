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
from constants import STRENGTH_TEMP_MIN

# Optimized Parallel Russian Doll MCTS implementation
class ParallelRussianDollMCTS:
    def __init__(self, board, iterations=20, exploration_weight=1.0, samples_per_level=None,
                 num_workers=None, learned_leaf_weight=1.0, prior_bias_temperature=1.0):
        self.board = board.copy()  # Keep one master copy
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.samples_per_level = samples_per_level or [21, 13, 8, 5, 3, 2, 1]
        self.num_workers = num_workers or min(8, multiprocessing.cpu_count())

        # Prior->learned annealing coefficients (see spec/search-mcts.spec.md).
        # learned_leaf_weight: share of a leaf value taken from the network vs the prior.
        # prior_bias_temperature: softens prior move-category weights before sampling (>1 = flatter).
        self.learned_leaf_weight = learned_leaf_weight
        self.prior_bias_temperature = prior_bias_temperature
        # Best (argmax-visit) root action from the last search; exposed for regret scoring.
        self.best_action = None
        
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
        
    def search(self, neural_net=None, device=None, training_progress=0.0, current_move=0,
               max_moves=200, temperature=0.0):
        """Parallel MCTS search with simplified board management.

        temperature controls root move selection: <= STRENGTH_TEMP_MIN picks the most-visited
        move (argmax, strongest); higher values sample p(a) proportional to visits^(1/T),
        giving weaker, more varied play. self.best_action always records the argmax move so
        callers can score regret without a second search.
        """
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
        
        # Select move from the root based on visit counts
        with self.global_lock:
            actions = self.children.get(self.root_state, [])
            if not actions:
                # Fallback: return a random legal move
                legal_moves = list(self.board.legal_moves)
                self.best_action = random.choice(legal_moves) if legal_moves else None
                return self.best_action

            visit_counts = [self.N[self.root_state][a] for a in actions]
            if all(count == 0 for count in visit_counts):
                # If no actions were visited, return random
                self.best_action = random.choice(actions)
                return self.best_action

            # Strongest move = highest MEAN ACTION VALUE (Q), not most-visited. With a modest
            # simulation budget (~200) spread across 20-30 root moves, every move gets only ~10
            # visits, so visit counts are far too flat to separate a clearly winning move (e.g. a
            # free-queen capture, Q~0.93) from a quiet move (Q~0.6) -- the winning move is often NOT
            # the most-visited. Q cleanly reflects the leaf evaluation. Only visited actions are
            # eligible (unvisited Q defaults to 0 and would spuriously win). Visit-count selection is
            # the AlphaZero default but assumes thousands of simulations, which this search lacks.
            q_values = [self.Q[self.root_state][a] if self.N[self.root_state][a] > 0 else -1e9
                        for a in actions]
            self.best_action = actions[int(np.argmax(q_values))]

            # Temperature 0 (or below the strength floor) => play the strongest move.
            if temperature <= STRENGTH_TEMP_MIN:
                return self.best_action

            # Boltzmann sampling over visit counts: p(a) proportional to visits^(1/T).
            counts = np.array(visit_counts, dtype=np.float64)
            weights = np.power(counts, 1.0 / temperature)
            total = weights.sum()
            if total <= 0 or not np.isfinite(total):
                return self.best_action
            probabilities = weights / total
            chosen_idx = int(np.random.choice(len(actions), p=probabilities))
            return actions[chosen_idx]

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
        
        # Evaluate final position (side-to-move-relative value at the leaf).
        value = self._evaluate_position(simulation_board, neural_net, device)

        # Negamax backup: flip to the mover's perspective at each parent BEFORE updating, so
        # Q[state][action] is always the value of `action` for the side to move at `state`. The
        # flip must come first: the leaf value is from the leaf mover's view, and the deepest edge
        # belongs to that mover's opponent.
        for state, action in reversed(states_actions):
            value = -value
            with self.node_locks[state]:
                self.N[state][action] += 1
                # Running average update
                old_q = self.Q[state][action]
                self.Q[state][action] = old_q + (value - old_q) / self.N[state][action]
    
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

        # Anneal the prior's search bias: soften category weights toward uniform as the
        # prior_bias_temperature rises, so late training does not lock the learner into the
        # prior's tactical categories (see spec/prior-evaluator.spec.md).
        if self.prior_bias_temperature != 1.0:
            inv_temp = 1.0 / self.prior_bias_temperature
            category_weights = {k: float(v) ** inv_temp for k, v in category_weights.items()}

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
        
        # Terminal positions, returned from the SIDE-TO-MOVE perspective (matches the negamax
        # backup in _parallel_simulation): the side to move being mated is a loss for the mover.
        if board.is_checkmate():
            return -1.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        # Check for threefold repetition (expensive, so do it last)
        if board.is_repetition(3):
            return 0.0

        # Blend prior and learned value (both White-absolute) by the annealed learned_leaf_weight:
        # the prior guides the search early (weight near 0) and the network takes over late (weight
        # near 1). See spec/search-mcts.spec.md. The prior is only computed when it actually
        # contributes (weight < 1) or when the network is unavailable/fails.
        w = self.learned_leaf_weight
        prior_value = None
        if w < 1.0:
            prior_value = math.tanh(fast_evaluate_position(board) / 10.0)

        white_value = None
        if neural_net is not None and w > 0.0:
            try:
                with torch.no_grad():
                    board_tensor = board_to_tensor(board).unsqueeze(0)
                    if device is not None:
                        board_tensor = board_tensor.to(device)
                    nn_value = max(-1.0, min(1.0, neural_net(board_tensor).item()))
                white_value = nn_value if prior_value is None else (1.0 - w) * prior_value + w * nn_value
            except Exception as e:
                print(f"Neural network evaluation error: {e}")
                # Fall back to the prior alone below.

        # Network absent or failed: use the prior (compute it now if we skipped it above).
        if white_value is None:
            if prior_value is None:
                try:
                    prior_value = math.tanh(fast_evaluate_position(board) / 10.0)
                except Exception as e:
                    print(f"Heuristic evaluation error: {e}")
                    return 0.0
            white_value = prior_value

        # White-absolute -> side-to-move-relative, so the alternating negamax backup is sign-correct
        # for BOTH colors (previously the leaf stayed White-absolute, corrupting Q for Black-to-move
        # nodes and for any simulation whose depth parity put White at the deepest decision).
        return white_value if board.turn == chess.WHITE else -white_value