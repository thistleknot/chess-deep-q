"""Value-only PUCT (spec/bullet-route.spec.md :MCTS-door:) — depth for ANY evaluator.

Standard PUCT over a generic batched value function (White-absolute, [-1,1]-ish), with
a DECLARED hand-crafted prior: P(a) = softmax(child 1-ply values / T_P) — Grill et al.
2020 says the prior is load-bearing at low sims; this is the cheapest non-flat prior
that needs no trained policy head (H2 stays parked). No rollouts at this rung (Q8's
rollout-mix is a declared optional dial, OFF by default).

API: puct_move(board, vf, enc, sims, rng) -> chess.Move
  vf(X: (n,d) float32) -> (n,) White-absolute values; enc(board) -> (d,) float32.
Move choice: softmax over root visit counts at tau_v = 0.02*sims (duel dither).
Failure modes: terminal board -> ValueError; all children mate-losing -> best by Q.
"""
import math

import numpy as np
import chess

C_PUCT = 1.5     # declared (spec :MCTS-door:)
T_PRIOR = 0.2    # declared prior temperature over child 1-ply values


class _Node:
    __slots__ = ("board", "parent", "move", "children", "P", "N", "W", "expanded", "terminal_v")

    def __init__(self, board, parent, move):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.P = 0.0
        self.N = 0
        self.W = 0.0     # total value, WHITE-absolute
        self.expanded = False
        self.terminal_v = None

    def q(self):
        return self.W / self.N if self.N else 0.0


def _terminal_value(board):
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0   # side to move is mated
    return 0.0                                              # stalemate/draw rules


def _expand(node, vf, enc):
    """Create children with softmax-over-child-values prior. Returns the node's
    1-ply BACKED value (White-absolute) — used as the leaf evaluation, saving the
    separate singleton eval call (measured: singleton overhead dominated sim cost)."""
    moves = list(node.board.legal_moves)
    xs, kids = [], []
    for mv in moves:
        b = node.board.copy(stack=False)
        b.push(mv)
        kid = _Node(b, node, mv)
        if b.is_game_over():
            kid.terminal_v = _terminal_value(b)
        kids.append(kid)
        xs.append(enc(b))
    v = vf(np.stack(xs)).astype(np.float64)
    sgn = 1.0 if node.board.turn == chess.WHITE else -1.0
    z = np.array([sgn * (k.terminal_v if k.terminal_v is not None else cv)
                  for k, cv in zip(kids, v)])
    p = np.exp((z - z.max()) / T_PRIOR)
    p /= p.sum()
    for kid, pi in zip(kids, p):
        kid.P = float(pi)
    node.children = kids
    node.expanded = True
    return sgn * float(z.max())                      # backed value, White-absolute


def puct_move(board, vf, enc, sims, rng):
    if board.is_game_over():
        raise ValueError("puct_move on terminal position")
    root = _Node(board.copy(stack=False), None, None)
    root.W = _expand(root, vf, enc)
    root.N = 1
    for _ in range(sims):
        node = root
        # SELECT down to a leaf
        while node.expanded and node.terminal_v is None:
            sgn = 1.0 if node.board.turn == chess.WHITE else -1.0
            best, best_s = None, -1e18
            sqrt_n = math.sqrt(node.N)
            for kid in node.children:
                u = C_PUCT * kid.P * sqrt_n / (1 + kid.N)
                s = sgn * kid.q() + u
                if s > best_s:
                    best, best_s = kid, s
            node = best
        # EVALUATE (backed value from the expansion batch; terminals exact)
        v = node.terminal_v if node.terminal_v is not None else _expand(node, vf, enc)
        # BACKUP (White-absolute all the way — no sign flipping needed)
        while node is not None:
            node.N += 1
            node.W += v
            node = node.parent
    # move choice: visit argmax, REPETITION-AWARE (pinned lesson: dither-by-visit-
    # softmax was nominal — visit gaps >> any sane temperature -> deterministic ->
    # fivefold-repetition cycles, 6/8 games). Walk candidates best-visits-first and
    # take the first that does not create a repeat, unless every move repeats.
    order = sorted(root.children, key=lambda k: -k.N)
    for kid in order:
        board.push(kid.move)
        rep = board.is_repetition(2)
        board.pop()
        if not rep:
            return kid.move
    return order[0].move
