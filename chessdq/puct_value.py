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

Search-arm dials (spec/search-arms.spec.md, all defaults preserve the base engine
bit-for-bit): lambda_backup blends mean and minimax node values (:Lambda-tree-backup:);
leaf_fn swaps the 1-ply backed leaf for an external evaluator, e.g. a shallow native
alpha-beta backed value (:AB-leaf:); selection="ucbv" swaps PUCT for variance-adaptive
UCB-V, Audibert et al. 2009, seeded with the expansion value as one pseudo-sample
(:UCBV-selection: — replaces UCB's forced first-visit sweep, fatal at chess branching).
"""
import math

import numpy as np
import chess

C_PUCT = 1.5     # declared (spec :MCTS-door:)
T_PRIOR = 0.2    # declared prior temperature over child 1-ply values


class _Node:
    __slots__ = ("board", "parent", "move", "children", "P", "N", "W", "W2", "M", "V0",
                 "expanded", "terminal_v")

    def __init__(self, board, parent, move):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.P = 0.0
        self.N = 0
        self.W = 0.0     # total value, WHITE-absolute
        self.W2 = 0.0    # total squared value (sign-invariant; UCB-V variance)
        self.M = 0.0     # minimax subtree value, WHITE-absolute (:Lambda-tree-backup:)
        self.V0 = 0.0    # immutable expansion-batch value (UCB-V pseudo-sample seed)
        self.expanded = False
        self.terminal_v = None

    def q(self):
        return self.W / self.N if self.N else 0.0


def _terminal_value(board):
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0   # side to move is mated
    return 0.0                                              # stalemate/draw rules


def _expand(node, vf, enc, t_prior=T_PRIOR, leaf_fn=None):
    """Create children with softmax-over-child-values prior. Returns the node's
    1-ply BACKED value (White-absolute) — used as the leaf evaluation, saving the
    separate singleton eval call (measured: singleton overhead dominated sim cost).
    With leaf_fn (:AB-leaf:), the returned leaf value comes from leaf_fn(node.board)
    instead (deeper backed estimate); priors still come from the cheap vf batch.
    Every child's M/V0 is seeded from its expansion-batch value, so minimax backup
    and the UCB-V pseudo-sample are total over children from the first visit."""
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
    p = np.exp((z - z.max()) / t_prior)
    p /= p.sum()
    for kid, pi, zi in zip(kids, p, z):
        kid.P = float(pi)
        kid.M = kid.V0 = sgn * float(zi)             # White-absolute init value
    node.children = kids
    node.expanded = True
    backed = sgn * float(z.max())                    # backed value, White-absolute
    if leaf_fn is not None:
        backed = float(leaf_fn(node.board))
    node.M = backed
    return backed


def _minimax_children(node):
    """White-absolute hard minimax over children M (:Lambda-tree-backup: update rule)."""
    ms = [k.M for k in node.children]
    return max(ms) if node.board.turn == chess.WHITE else min(ms)


def _ucbv_score(kid, sgn, ln_np, lb, ucb_c):
    """UCB-V child score (:UCBV-selection:) — mover-perspective mean (pseudo-sample
    seeded, λ-blended) + variance bonus + count bonus. Factored out for gate G4."""
    n_eff = kid.N + 1
    mean = (kid.W + kid.V0) / n_eff
    var = max((kid.W2 + kid.V0 * kid.V0) / n_eff - mean * mean, 0.0)
    q_eff = (1.0 - lb) * mean + lb * kid.M
    return (sgn * q_eff
            + math.sqrt(2.0 * var * ucb_c * ln_np / n_eff)
            + 3.0 * ucb_c * ln_np / n_eff)


def puct_move(board, vf, enc, sims, rng, c_puct=C_PUCT, t_prior=T_PRIOR,
              lambda_backup=0.0, selection="puct", leaf_fn=None, ucb_c=1.0,
              return_root=False):
    if board.is_game_over():
        raise ValueError("puct_move on terminal position")
    lb = float(lambda_backup)
    root = _Node(board.copy(stack=False), None, None)
    root.W = _expand(root, vf, enc, t_prior, leaf_fn)
    root.N = 1
    root.W2 = root.W * root.W
    for _ in range(sims):
        node = root
        # SELECT down to a leaf
        while node.expanded and node.terminal_v is None:
            sgn = 1.0 if node.board.turn == chess.WHITE else -1.0
            best, best_s = None, -1e18
            if selection == "ucbv":
                # UCB-V: mean + sqrt(2*Var*c*lnN/n) + 3*c*lnN/n, each child seeded
                # with its expansion value as one pseudo-sample (n_eff = N+1).
                ln_np = math.log(node.N) if node.N > 1 else 0.0
                for kid in node.children:
                    s = _ucbv_score(kid, sgn, ln_np, lb, ucb_c)
                    if s > best_s:
                        best, best_s = kid, s
            else:
                sqrt_n = math.sqrt(node.N)
                for kid in node.children:
                    u = c_puct * kid.P * sqrt_n / (1 + kid.N)
                    q_eff = kid.q() if lb == 0.0 else (1.0 - lb) * kid.q() + lb * kid.M
                    s = sgn * q_eff + u
                    if s > best_s:
                        best, best_s = kid, s
            node = best
        # EVALUATE (backed value from the expansion batch; terminals exact)
        v = node.terminal_v if node.terminal_v is not None \
            else _expand(node, vf, enc, t_prior, leaf_fn)
        # BACKUP (White-absolute all the way — no sign flipping needed).
        # Minimax M refreshes on ANCESTORS only: the evaluated leaf keeps its own
        # (possibly deeper, leaf_fn-backed) M until sims flow through its children.
        leaf = node
        while node is not None:
            node.N += 1
            node.W += v
            node.W2 += v * v
            if lb > 0.0 and node.expanded and node is not leaf:
                node.M = _minimax_children(node)
            node = node.parent
    # move choice: visit argmax, REPETITION-AWARE (pinned lesson: dither-by-visit-
    # softmax was nominal — visit gaps >> any sane temperature -> deterministic ->
    # fivefold-repetition cycles, 6/8 games). Walk candidates best-visits-first and
    # take the first that does not create a repeat, unless every move repeats.
    order = sorted(root.children, key=lambda k: -k.N)
    chosen = order[0].move
    for kid in order:
        board.push(kid.move)
        rep = board.is_repetition(2)
        board.pop()
        if not rep:
            chosen = kid.move
            break
    return (chosen, root) if return_root else chosen


def _kid_blend(kid, lb):
    """Child blended value, White-absolute: pseudo-sample mean (expansion value
    counts once, robust for unvisited kids) λ-mixed with its minimax M."""
    mean = (kid.W + kid.V0) / (kid.N + 1)
    return (1.0 - lb) * mean + lb * kid.M


def puct_choose(board, vf, enc, sims, rng, tau=0.0, lambda_backup=0.5, leaf_fn=None):
    """TDLeaf-compatible root call (spec/search-arms.spec.md :Lambda-target-training:):
    (move, pv_leaf_board, gv, predicted_reply) — drop-in for search_move's
    return_leaf contract inside qlearn.choose().
    gv honors :Backed-bootstrap: (S&B 015): its minimax component maxes over
    EXPANDED children only — an unvisited kid's 1-ply init never enters the max.
    Require: non-terminal board, sims >= 1. Failure mode: no expanded child
    (cannot happen at sims>=1 — the first sim expands one) -> gv falls back to
    the root mean."""
    lb = float(lambda_backup)
    mv, root = puct_move(board, vf, enc, sims, rng, lambda_backup=lb,
                         leaf_fn=leaf_fn, return_root=True)
    sgn = 1.0 if board.turn == chess.WHITE else -1.0
    if tau > 0.0 and rng is not None:
        # beam-consistent behavior: softmax over mover-perspective child blended
        # values at temperature tau (value units — the tau schedule keeps meaning),
        # with the same repetition refusal as the argmax path
        vals = [sgn * _kid_blend(k, lb) for k in root.children]
        mx = max(vals)
        ws = [math.exp((v - mx) / tau) for v in vals]
        total = sum(ws)
        r = rng.random() * total
        pick, acc = len(ws) - 1, 0.0
        for i, w in enumerate(ws):
            acc += w
            if r <= acc:
                pick = i
                break
        board.push(root.children[pick].move)
        rep = board.is_repetition(2)
        board.pop()
        if rep:
            for i in sorted(range(len(vals)), key=lambda j: -vals[j]):
                board.push(root.children[i].move)
                ok = not board.is_repetition(2)
                board.pop()
                if ok:
                    pick = i
                    break
        mv = root.children[pick].move
    chosen = next(k for k in root.children if k.move == mv)
    # gv: λ-blend of the root's backed-up mean and the EXPANDED-children minimax
    backed = [k.M for k in root.children if k.expanded or k.terminal_v is not None]
    if backed:
        mm = max(backed) if sgn > 0 else min(backed)
        gv = (1.0 - lb) * root.q() + lb * mm
    else:
        gv = root.q()
    # PV leaf: descend the chosen child by max visits while the tree is real
    node = chosen
    while node.expanded and node.children and max(k.N for k in node.children) > 0:
        node = max(node.children, key=lambda k: k.N)
    pred = None
    if chosen.expanded and chosen.children:
        best_reply = max(chosen.children, key=lambda k: k.N)
        if best_reply.N > 0:
            pred = best_reply.move
    return mv, node.board, float(gv), pred
