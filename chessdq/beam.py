"""Net-guided Fibonacci sampled-lookahead beam on the dual-head resnet (spec :Compute-frontier:).

Carries forward mcts_chess.py's Fibonacci beam, but on OUR 18-plane (value, policy) net and with
two fixes:
  * fan-out is POLICY-GUIDED (top-pair_k by the policy head), not uniform random -> better lines
    explored per call (the mcts_chess critique);
  * every layer is scored in ONE batched net forward (serial across layers, batched within) -- so
    the honest time model is ~depth batched calls, while total_calls is the CPU-work measure.

Cost is a deterministic integer: total_calls(widths, pair_k). The schedule is a descending
Fibonacci width ladder derived from the two knobs (depth, total_ops). Values are White-absolute
in [-1, 1] (our resnet convention); terminal leaves override to +-MATE / 0.
"""

import chess
import torch

from chessdq.resnet_model import encode18, move_index

MATE = 2.0   # beyond the tanh range so mates dominate the [-1,1] value


def fib_schedule(depth, width=1, start=1):
    """Descending-Fibonacci width ladder, length `depth` (same shape as mcts_chess.fib_schedule)."""
    assert depth >= 1 and width >= 1 and start >= 1
    need = start + width * (depth - 1)
    fib = [1, 2]
    while len(fib) < need:
        fib.append(fib[-1] + fib[-2])
    return tuple(fib[start - 1 + width * (depth - 1 - i)] for i in range(depth))


def total_calls(widths, pair_k=2):
    """Deterministic leaf-eval count for a schedule: the wide root layer + pair_k fan-out per
    surviving line each subsequent layer. A single int = the compute cost (CPU-work axis)."""
    if not widths:
        return 0
    return int(widths[0] + pair_k * sum(widths[1:]))


def schedule_for(depth, total_ops, pair_k=2):
    """Derive a length-`depth` descending width ladder whose total_calls ~= total_ops.

    Uses the canonical Fibonacci shape (width=1, start=1) scaled to hit the target op budget, so the
    two meaningful knobs (depth, total_ops) map to a concrete schedule; width/start fall out."""
    base = fib_schedule(depth, 1, 1)
    base_calls = total_calls(base, pair_k)
    scale = max(1.0, total_ops / base_calls)
    return tuple(max(1, round(w * scale)) for w in base)


def _terminal_white(board):
    """White-absolute value if `board` is terminal, else None (side to move being mated = loss)."""
    if board.is_checkmate():
        return -MATE if board.turn == chess.WHITE else MATE
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    return None


class _Line:
    __slots__ = ("board", "root_move", "value", "policy", "terminal")

    def __init__(self, board, root_move, value, policy, terminal):
        self.board = board          # leaf board of this line
        self.root_move = root_move  # the root move it commits to
        self.value = value          # ROOT-mover-perspective value (higher = better for root)
        self.policy = policy        # {move: prior} at `board`, for the next fan-out
        self.terminal = terminal


class NetBeam:
    """A policy-guided, batched Fibonacci beam. `select(board)` returns the root move.

    widths: the schedule (descending). pair_k: continuations fanned per line per layer.
    tau: prune temperature over root-perspective value. eps: heuristic blend weight
    (eps*tanh(pst/400) + (1-eps)*net_value); eps=0 is pure net, eps=1 net-free (needs pst).
    `self.calls` accumulates net leaf evaluations = the measured compute cost.
    """

    def __init__(self, net, device, widths, pair_k=2, tau=0.25, eps=0.0):
        self.net, self.device = net, device
        self.widths = tuple(widths)
        self.pair_k, self.tau, self.eps = pair_k, tau, eps
        self.calls = 0
        self.last_margin = 0.0   # :Move-margin: = max - mean(finalists); decisiveness (>= 0)
        self.last_value = 0.0    # root-perspective backed-up value acted on (for cross-move :Delta:)

    @torch.no_grad()
    def _eval_batch(self, boards):
        """ONE net forward for a list of boards. Returns (white_values, priors) lists; terminal
        boards override the value and get empty priors. Counts one call per live board."""
        n = len(boards)
        wv, pr = [None] * n, [None] * n
        live = []
        for i, b in enumerate(boards):
            t = _terminal_white(b)
            if t is None:
                live.append(i)
            else:
                wv[i], pr[i] = t, {}
        if live:
            self.calls += len(live)
            x = torch.stack([encode18(boards[i]) for i in live]).to(self.device)
            val, pol = self.net(x)
            val = val.squeeze(1).tolist()
            for j, i in enumerate(live):
                v = max(-1.0, min(1.0, val[j]))
                if self.eps > 0.0:
                    from chessdq.engine import pst_eval
                    import math
                    h = math.tanh(pst_eval(boards[i]) / 400.0)
                    v = self.eps * h + (1.0 - self.eps) * v
                wv[i] = v
                moves = list(boards[i].legal_moves)
                logits = pol[j][[move_index(m) for m in moves]]
                pr[i] = dict(zip(moves, torch.softmax(logits, 0).tolist()))
        return wv, pr

    @staticmethod
    def _root_persp(white_value, root_turn):
        """Root-mover view of a White-absolute value (higher = better for the root mover)."""
        return white_value if root_turn == chess.WHITE else -white_value

    def _prune(self, lines, w):
        """Temperature-sample `w` survivors without replacement by root-perspective value."""
        if len(lines) <= w:
            return lines
        pool, picked = list(lines), []
        while len(picked) < w:
            logits = torch.tensor([l.value / self.tau for l in pool])
            i = int(torch.multinomial(torch.softmax(logits, 0), 1).item())
            picked.append(pool.pop(i))
        return picked

    def _fan_out(self, lines, root_turn):
        """Extend each non-terminal line by its negamax-best of the top-pair_k POLICY moves."""
        cand, owner = [], []
        for li, l in enumerate(lines):
            if l.terminal:
                continue
            moves = (sorted(l.policy, key=l.policy.get, reverse=True)[:self.pair_k]
                     if l.policy else list(l.board.legal_moves)[:self.pair_k])
            for m in moves:
                b = l.board.copy(stack=False)
                b.push(m)
                cand.append(b)
                owner.append(li)
        if not cand:
            return
        wv, pr = self._eval_batch(cand)
        best = {}   # line -> (mover_value, board, white_value, priors)
        for b, li, v, p in zip(cand, owner, wv, pr):
            mover_sign = 1 if lines[li].board.turn == chess.WHITE else -1
            mover_val = v * mover_sign
            if li not in best or mover_val > best[li][0]:
                best[li] = (mover_val, b, v, p)
        for li, (_, b, v, p) in best.items():
            l = lines[li]
            l.board, l.policy = b, p
            l.value = self._root_persp(v, root_turn)
            l.terminal = _terminal_white(b) is not None

    @torch.no_grad()
    def select(self, root):
        """Return the beam's root move (legal), or None on a terminal board."""
        legal = list(root.legal_moves)
        if not legal:
            return None
        rt = root.turn
        if len(legal) == 1:
            # Forced move: stamp its HONEST value (one eval), not a stale/absent stamp, so a
            # cross-move :Delta: consumer isn't fed a phantom estimate. Margin 0 (no alternatives).
            b = root.copy(stack=False); b.push(legal[0])
            self.last_value = self._root_persp(self._eval_batch([b])[0][0], rt)
            self.last_margin = 0.0
            return legal[0]

        # Root policy -> keep the top widths[0] root moves BY POLICY (not random).
        _, root_pr = self._eval_batch([root])
        pr0 = root_pr[0] or {m: 0.0 for m in legal}
        top = sorted(legal, key=lambda m: pr0.get(m, 0.0), reverse=True)[:self.widths[0]]
        boards = []
        for m in top:
            b = root.copy(stack=False)
            b.push(m)
            boards.append(b)
        wv, pr = self._eval_batch(boards)
        lines = [_Line(b, m, self._root_persp(v, rt), p, _terminal_white(b) is not None)
                 for m, b, v, p in zip(top, boards, wv, pr)]

        finalists = lines
        for i, w in enumerate(self.widths[1:]):
            last = (i == len(self.widths) - 2)
            if last and w == 1:
                break                       # :Root-commit: argmax supersedes the dead final collapse
            lines = self._prune(lines, w)
            if len(lines) > 1:
                finalists = lines          # last top_k (>1) survivor set, for :Move-margin:
            if not last:
                self._fan_out(lines, rt)
        # :Root-commit: — argmax over root backed-up values (not a temperature pick); exploration is
        # upstream. :Move-margin: over the finalists; last_value feeds cross-move :Delta:.
        best = max(finalists, key=lambda l: l.value)
        self.last_margin = best.value - sum(l.value for l in finalists) / len(finalists)
        self.last_value = best.value
        return best.root_move
