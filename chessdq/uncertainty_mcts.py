"""Hybrid uncertainty-directed MCTS agent.

Uses K bootstrap ridge heads (the ensemble from ensemble-explore) to detect
positions of high value-uncertainty, then runs bounded PUCT search at those
positions using softmax over the champion eval as the move prior. At low-
uncertainty positions, falls back to 1-ply greedy (fast).

This is a PLAYING agent with a `move_fn(board) -> Move` interface, directly
measurable on the ladder (random, heuristic, SF rungs).

Architecture (all inference-time, no training loop):
  1. Encode board (amap-897)
  2. σ = std across K heads' value predictions
  3. IF σ > threshold: run PUCT(sims) with soft-policy prior from eval
     ELSE: 1-ply greedy over afterstate values (same as champion sans search)
  4. Return move

The value function at leaves is the champion's own linear eval (mean of the
K heads for slightly better accuracy at no extra cost). The policy prior is
softmax(tau * afterstate_values) — the eval itself as a policy, tempered.
"""
import math
import random

import numpy as np
import chess

from chessdq.encoders import get
from chessdq.engine import pst_eval

_ENC_FN, _NIN = get("amap")


# ---------------------------------------------------------------------------
# PUCT tree search with eval-based soft prior
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("board", "turn", "expanded", "terminal", "tv",
                 "moves", "P", "N", "W", "Ntot", "children")

    def __init__(self, board):
        self.board = board
        self.turn = board.turn
        self.expanded = False
        self.terminal = False
        self.tv = 0.0
        self.moves = None
        self.P = self.N = self.W = None
        self.Ntot = 0
        self.children = {}


def _terminal_value(board):
    """White-absolute terminal value."""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


def _make_child(node, i):
    b = node.board.copy(stack=False)
    b.push(node.moves[i])
    ch = _Node(b)
    if b.is_game_over():
        ch.terminal = True
        ch.tv = _terminal_value(b)
        ch.expanded = True
    return ch


def _expand(node, value_fn, tau):
    """Expand node: compute soft-policy prior from afterstate values."""
    moves = list(node.board.legal_moves)
    node.moves = moves
    if not moves:
        node.expanded = True
        node.P = np.zeros(0)
        node.N = np.zeros(0)
        node.W = np.zeros(0)
        return
    # Compute afterstate values for each move -> softmax prior
    vals = np.empty(len(moves), dtype=np.float64)
    sign = 1.0 if node.turn == chess.WHITE else -1.0
    for i, mv in enumerate(moves):
        node.board.push(mv)
        if node.board.is_checkmate():
            vals[i] = 1e6  # mover delivered mate -> always pick
        elif node.board.is_game_over():
            vals[i] = 0.0
        else:
            vals[i] = sign * value_fn(node.board)  # mover-perspective
        node.board.pop()
    # Softmax with temperature
    logits = vals / max(tau, 1e-6)
    logits -= logits.max()
    pri = np.exp(logits)
    pri /= pri.sum()
    node.P = pri
    node.N = np.zeros(len(moves))
    node.W = np.zeros(len(moves))
    node.Ntot = 0
    node.expanded = True


def _select_child(node, c_puct):
    sign = 1.0 if node.turn == chess.WHITE else -1.0
    q = np.where(node.N > 0, node.W / np.maximum(node.N, 1.0), 0.0) * sign
    u = c_puct * node.P * math.sqrt(node.Ntot + 1) / (1.0 + node.N)
    return int(np.argmax(q + u))


def _select_leaf(root, c_puct):
    node, path = root, []
    while node.expanded and not node.terminal:
        if not node.moves:
            break
        i = _select_child(node, c_puct)
        path.append((node, i))
        ch = node.children.get(i)
        if ch is None:
            ch = _make_child(node, i)
            node.children[i] = ch
        node = ch
    return node, path


def _backup(path, v_white):
    for node, i in path:
        node.N[i] += 1
        node.W[i] += v_white
        node.Ntot += 1


def _puct_search(board, value_fn, sims, tau, c_puct):
    """Run PUCT with eval-based soft prior. Returns best move (argmax visits)."""
    root = _Node(board.copy(stack=False))
    _expand(root, value_fn, tau)
    if not root.moves:
        return None
    if len(root.moves) == 1:
        return root.moves[0]
    for _ in range(sims):
        node, path = _select_leaf(root, c_puct)
        if node.terminal:
            _backup(path, node.tv)
        else:
            _expand(node, value_fn, tau)
            # Leaf value from the value function (white-absolute)
            v = value_fn(node.board)
            _backup(path, v)
    return root.moves[int(np.argmax(root.N))]


# ---------------------------------------------------------------------------
# Ensemble uncertainty + hybrid agent
# ---------------------------------------------------------------------------

class UncertaintyMCTSAgent:
    """Hybrid agent: MCTS at uncertain positions, greedy elsewhere.

    Require: W (K, n_features), B (K,) — bootstrap ridge ensemble weights.
    Params:
      sigma_threshold: positions with σ > this get MCTS (default: top ~25% quantile)
      sims: PUCT simulations at uncertain positions
      tau: temperature for the soft policy prior in PUCT
      c_puct: exploration constant for PUCT selection
    """

    def __init__(self, W, B, sigma_threshold=0.010, sims=32, tau=0.15, c_puct=1.5):
        self.W = np.asarray(W, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.sigma_threshold = sigma_threshold
        self.sims = sims
        self.tau = tau
        self.c_puct = c_puct
        # Mean weights as the value function (slightly better than single head)
        self._w_mean = self.W.mean(axis=0)
        self._b_mean = float(self.B.mean())

    def _encode(self, board):
        return _ENC_FN(board).astype(np.float64)

    def _value(self, board):
        """White-absolute value in [-1, 1] (tanh of mean linear score)."""
        x = self._encode(board)
        return float(np.tanh(self._w_mean @ x + self._b_mean))

    def _sigma(self, board):
        """Ensemble disagreement for a single board."""
        x = self._encode(board)
        preds = x @ self.W.T + self.B  # shape (K,)
        return float(preds.std())

    def _greedy_move(self, board):
        """1-ply greedy: pick the afterstate with highest mover-perspective value."""
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        best_mv, best_v = None, -1e9
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                return mv
            v = sign * self._value(board)
            board.pop()
            if v > best_v:
                best_v, best_mv = v, mv
        return best_mv

    def move(self, board):
        """Select a move: MCTS if uncertain, greedy if confident."""
        sigma = self._sigma(board)
        if sigma > self.sigma_threshold:
            mv = _puct_search(board, self._value, self.sims, self.tau, self.c_puct)
            if mv is not None:
                return mv
        return self._greedy_move(board)

    def __call__(self, board):
        return self.move(board)


# ---------------------------------------------------------------------------
# Greedy-only baseline (same value function, no MCTS ever)
# ---------------------------------------------------------------------------

class GreedyBaselineAgent:
    """Same ensemble-mean value function, always greedy — the matched control."""

    def __init__(self, W, B):
        self.W = np.asarray(W, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self._w_mean = self.W.mean(axis=0)
        self._b_mean = float(self.B.mean())

    def _value(self, board):
        x = _ENC_FN(board).astype(np.float64)
        return float(np.tanh(self._w_mean @ x + self._b_mean))

    def __call__(self, board):
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        best_mv, best_v = None, -1e9
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                return mv
            v = sign * self._value(board)
            board.pop()
            if v > best_v:
                best_v, best_mv = v, mv
        return best_mv


# ---------------------------------------------------------------------------
# Measurement harness
# ---------------------------------------------------------------------------

def build_ensemble(cache_path="models/distillA_labels.npz", K=16, ridge=100.0, seed=977):
    """Build the K-head bootstrap ensemble from the cached label corpus."""
    from experiments.distill_linear import featurize, fit_ridge
    z = np.load(cache_path, allow_pickle=True)
    fens, y = list(z["fens"]), z["y"]
    print(f"  building K={K} ensemble from {len(fens)} cached positions...", flush=True)
    X = featurize(fens)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    W = np.empty((K, X.shape[1]), dtype=np.float64)
    B = np.empty(K, dtype=np.float64)
    for k in range(K):
        idx = rng.integers(0, n, size=n)
        w_k, b_k, _ = fit_ridge(X[idx], y[idx], ridge)
        W[k], B[k] = w_k, b_k
    return W, B


def run_ladder(games=20, sims=32, tau=0.15, c_puct=1.5, sigma_threshold=0.010):
    """Build ensemble, create agents, run ladder. Prints results."""
    import time
    from chessdq.measure_ladder import random_mover, heuristic_mover, play, adj_pst, elo_diff

    print("=" * 60)
    print("UNCERTAINTY-MCTS HYBRID — LADDER TEST")
    print("=" * 60)

    t0 = time.time()
    W, B = build_ensemble()
    print(f"  ensemble built in {time.time()-t0:.1f}s", flush=True)

    agent = UncertaintyMCTSAgent(W, B, sigma_threshold=sigma_threshold,
                                  sims=sims, tau=tau, c_puct=c_puct)
    baseline = GreedyBaselineAgent(W, B)

    print(f"\nConfig: sims={sims}, tau={tau}, c_puct={c_puct}, σ_thresh={sigma_threshold}")
    print(f"Games per rung: {games} (alternating colors, cap=100 plies)")

    # --- Hybrid agent vs rungs ---
    print(f"\n--- HYBRID AGENT (MCTS at σ>{sigma_threshold}) ---")
    for name, opp in [("random", random_mover), ("heuristic", heuristic_mover)]:
        t1 = time.time()
        W_r, D_r, L_r = play(agent, opp, games, adj_pst)
        s = (W_r + 0.5 * D_r) / games
        elo = elo_diff(s)
        dt = time.time() - t1
        print(f"  vs {name:10s}: {W_r}W {D_r}D {L_r}L  score={s:.3f}  "
              f"Elo_diff={elo:+.0f}  ({dt:.1f}s)")

    # --- Greedy baseline vs rungs (matched control) ---
    print(f"\n--- GREEDY BASELINE (same value fn, no MCTS) ---")
    for name, opp in [("random", random_mover), ("heuristic", heuristic_mover)]:
        t1 = time.time()
        W_r, D_r, L_r = play(baseline, opp, games, adj_pst)
        s = (W_r + 0.5 * D_r) / games
        elo = elo_diff(s)
        dt = time.time() - t1
        print(f"  vs {name:10s}: {W_r}W {D_r}D {L_r}L  score={s:.3f}  "
              f"Elo_diff={elo:+.0f}  ({dt:.1f}s)")

    total = time.time() - t0
    print(f"\nTotal wall-clock: {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    run_ladder(games=games, sims=sims)
