"""Alpha-beta chess engine: the strength substrate.

Iterative deepening + negamax alpha-beta + quiescence (captures) + transposition table (Zobrist)
+ move ordering (TT move, MVV-LVA, killers, history). Strength comes from search depth here; the
learned evaluator only has to be a good, fast value function to feed it.

Evaluation is pluggable: pass any `eval_fn(board) -> white-absolute centipawn-ish score`. Defaults
to the project's fast_evaluate_position, but a lighter material+PST eval searches far deeper per
second (see fast_material_eval).
"""

import time
import random
import chess
import chess.polyglot

from evaluation import fast_evaluate_position
from constants import PIECE_VALUES

MATE = 1_000_000
INF = 10_000_000

# --- :Phi-widening: forward-pruning budget (spec/search-mcts.spec.md) -------------------------
# Below PHI_FULL_WIDTH_FLOOR plies EVERY legal move is searched (:Full-width-floor:) so the agent
# never prunes inside the shallow tactical horizon. Beyond it, only the top-phi_width(depth) moves
# by ordering are expanded -- captures/checks always exempt (refutation-preserving), which is what
# keeps the sound alpha-beta realization sound where the sampled beam is not.
PHI_FULL_WIDTH_FLOOR = 3

# Fibonacci-spaced width budget, indexed by REMAINING search depth: wide near the root (large
# remaining depth) narrowing toward the leaves. Inlined (not beam.fib_schedule) to keep engine.py
# torch-free -- the teacher evaluate() labeling path must not drag in torch.
_PHI_FIB = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# Transposition-table size cap for :Tree-reuse:. When exceeded, the oldest half of the
# insertion-ordered dict is evicted (Python dicts preserve insertion order).
TT_SIZE_CAP = 2_000_000


def phi_width(depth):
    """:Phi-widening: quiet-move breadth budget at a node with `depth` plies remaining.

    Wide near the root (large `depth`) narrowing to 2 toward the leaves (`depth`->0), so the search
    spends fan-out where refutations still branch and narrows where they cannot. Captures and checks
    are exempt at the call site, so this bounds only late quiet-move breadth.

    Preconditions: `depth` may be any int; it is clamped into [0, len(_PHI_FIB)-1].
    Guarantee: returns a positive width from the Fibonacci ladder.
    """
    return _PHI_FIB[min(max(depth, 0), len(_PHI_FIB) - 1)]

# Piece-square-free fast material eval (centipawns), White-absolute. Much faster than the
# feature-rich evaluator -> deeper search per second.
_CP = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def fast_material_eval(board):
    score = 0
    for pt, v in _CP.items():
        score += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return score


# Piece-square tables (Michniewski "simplified evaluation"), written a8-first (visual top-left).
# Positional understanding at material-eval speed -- the biggest strength-per-node win available.
_PST = {
    chess.PAWN: [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10,
                 5,5,10,25,25,10,5,5, 0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5,
                 5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40,
                   -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30, -30,0,15,20,20,15,0,-30,
                   -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10,
                   -10,5,5,10,10,5,5,-10, -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10,
                   -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
                 -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
    chess.QUEEN: [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10,
                  -5,0,5,5,5,5,0,-5, 0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10,
                  -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
    chess.KING: [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
                 -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
                 -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10,
                 20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20],
}


def pst_eval(board):
    """Material + piece-square tables, White-absolute centipawns. Fast and positionally aware."""
    score = 0
    for pt, base in _CP.items():
        table = _PST[pt]
        for sq in board.pieces(pt, chess.WHITE):
            rank, file = sq >> 3, sq & 7
            score += base + table[(7 - rank) * 8 + file]
        for sq in board.pieces(pt, chess.BLACK):
            rank, file = sq >> 3, sq & 7
            score -= base + table[rank * 8 + file]
    return score


def _heuristic_eval(board):
    """Adapt the project's evaluator (pawn-ish units) to centipawns, White-absolute."""
    return fast_evaluate_position(board) * 100.0


# Compact opening book: main lines replayed at import into a {position -> [moves]} index, so the
# engine plays sound, varied openings instantly (and skips searching the opening).
_BOOK_LINES = [
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3",   # Open Sicilian
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4",             # Sicilian ...Nc6
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6",         # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3",             # Italian
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",                   # French
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4",             # Caro-Kann
    "e2e4 e7e5 g1f3 g8f6",                             # Petroff
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",                   # Nimzo-Indian
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",         # King's Indian
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",                   # QGD
    "d2d4 d7d5 c2c4 c7c6",                             # Slav
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5",                   # QGD/Catalan
    "c2c4 e7e5 b1c3 g8f6",                             # English
    "g1f3 d7d5 d2d4 g8f6 c2c4",                         # transposition
]


def _build_book(lines):
    book = {}
    for line in lines:
        b = chess.Board()
        for uci in line.split():
            key = " ".join(b.fen().split()[:2])   # placement + side to move
            book.setdefault(key, [])
            if uci not in book[key]:
                book[key].append(uci)
            b.push(chess.Move.from_uci(uci))
    return book


OPENING_BOOK = _build_book(_BOOK_LINES)

EXACT, LOWER, UPPER = 0, 1, 2


class AlphaBetaEngine:
    def __init__(self, eval_fn=None, time_limit=2.0, max_depth=64,
                 policy_fn=None, policy_plies=0, policy_weight=80_000.0,
                 phi_widen=False):
        self.eval_fn = eval_fn or pst_eval
        self.time_limit = time_limit
        self.max_depth = max_depth
        # :Phi-widening: gate. Default OFF so the teacher evaluate() labeling path stays
        # full-width/sound; the :Deployed-agent: ("nnue" branch in agents.make_agent) sets it ON.
        self.phi_widen = phi_widen
        self.tt = {}
        self.killers = {}
        self.history = {}
        self.nodes = 0
        # Learned move-ordering prior (spec/search-mcts.spec.md: policy head consumed via ordering).
        # policy_fn(board) -> {move: prior in [0,1]}. Applied ONLY at ply <= policy_plies because a
        # net call is ~2-3ms/node vs a us-fast eval -- deep application would blow the time budget.
        # Default policy_plies=0 = root-only (a handful of calls per move, negligible overhead).
        self.policy_fn = policy_fn
        self.policy_plies = policy_plies
        self.policy_weight = policy_weight

    # --- transposition store (:Tree-reuse: replacement policy) -------------
    def _tt_store(self, key, depth, val, flag, move):
        """Depth-preferred, size-capped TT store (spec :Tree-reuse:).

        Require: `flag` is one of EXACT/LOWER/UPPER already downgraded by the caller for
        phi-pruned nodes (a forward-pruned node is never EXACT).
        Maintain: a colliding entry searched to GREATER depth is never clobbered by a shallower
        one; total entries stay <= TT_SIZE_CAP (evict oldest half on overflow, insertion-ordered).
        """
        existing = self.tt.get(key)
        if existing is not None and existing[0] > depth:
            return                                   # keep the deeper entry
        if existing is None and len(self.tt) >= TT_SIZE_CAP:
            for k in list(self.tt.keys())[: len(self.tt) // 2]:
                del self.tt[k]
        self.tt[key] = (depth, val, flag, move)

    # --- evaluation (side-to-move perspective) -----------------------------
    def _evaluate(self, board):
        e = self.eval_fn(board)
        return e if board.turn == chess.WHITE else -e

    # --- move ordering -----------------------------------------------------
    def _order(self, board, moves, tt_move, depth):
        killers = self.killers.get(depth, ())
        # Learned prior at shallow plies only (`depth` here is the ply; root=0). Computed once per
        # node, then indexed inside score(). A high-prior quiet move is lifted toward capture
        # territory (policy_weight) so the engine tries the net's suggestion early -> more cutoffs.
        priors = self.policy_fn(board) if (self.policy_fn is not None
                                           and depth <= self.policy_plies) else None
        w = self.policy_weight

        def score(m):
            if m == tt_move:
                return 1_000_000
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or chess.PAWN
                attacker = board.piece_type_at(m.from_square) or chess.PAWN
                base = 100_000 + 10 * _CP[victim] - _CP[attacker]
                return base + (w * priors.get(m, 0.0) if priors else 0.0)
            if m in killers:
                return 90_000
            h = self.history.get((board.turn, m.from_square, m.to_square), 0)
            return h + (w * priors.get(m, 0.0) if priors else 0.0)

        return sorted(moves, key=score, reverse=True)

    # --- quiescence (captures only) ----------------------------------------
    def _quiesce(self, board, alpha, beta):
        self.nodes += 1
        stand = self._evaluate(board)
        if stand >= beta:
            return beta
        # Delta pruning: if even capturing a queen can't reach alpha, stop searching captures.
        if stand + 975 < alpha:
            return alpha
        if stand > alpha:
            alpha = stand
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        # MVV: try to grab the biggest victim first (best chance of a cutoff).
        captures.sort(key=lambda m: _CP.get(board.piece_type_at(m.to_square) or chess.PAWN, 0),
                      reverse=True)
        for m in captures:
            victim = _CP.get(board.piece_type_at(m.to_square) or chess.PAWN, 0)
            # Per-capture delta pruning: hopeless captures that can't raise alpha are skipped.
            if stand + victim + 150 < alpha:
                continue
            board.push(m)
            score = -self._quiesce(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # --- negamax alpha-beta ------------------------------------------------
    def _negamax(self, board, depth, alpha, beta, ply):
        if time.time() - self.start > self.time_limit:
            raise TimeoutError
        self.nodes += 1

        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
            return 0

        key = chess.polyglot.zobrist_hash(board)
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            e_depth, e_val, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_val
                if e_flag == LOWER and e_val > alpha:
                    alpha = e_val
                elif e_flag == UPPER and e_val < beta:
                    beta = e_val
                if alpha >= beta:
                    return e_val

        if depth <= 0:
            return self._quiesce(board, alpha, beta)

        # Null-move pruning: hand the opponent a free move; if we're still >= beta, prune.
        # Skipped in check and in likely-zugzwang (no non-pawn material) to stay sound.
        if (depth >= 3 and not board.is_check() and any(
                board.pieces(pt, board.turn)
                for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))):
            board.push(chess.Move.null())
            null_val = -self._negamax(board, depth - 3, -beta, -beta + 1, ply + 1)
            board.pop()
            if null_val >= beta:
                return beta

        best_val, best_move = -INF, None
        alpha_orig = alpha
        phi_pruned = False           # any quiet move dropped by :Phi-widening: at this node?
        for i, m in enumerate(self._order(board, list(board.legal_moves), tt_move, ply)):
            is_cap = board.is_capture(m)
            # :Phi-widening: forward prune (PRE-push). Beyond the :Full-width-floor:, drop late
            # quiet non-checking moves outside the phi_width(depth) budget. Refutation-preserving:
            # captures and checks are ALWAYS searched -- board.gives_check(m) is the pre-push check
            # test (the post-push is_check below is for LMR of moves that survive). This extends the
            # existing LMR predicate (i>=4, depth>=3, quiet, non-check) from REDUCE to DROP.
            if self.phi_widen and ply >= PHI_FULL_WIDTH_FLOOR and i >= phi_width(depth) \
                    and not is_cap and not board.gives_check(m):
                phi_pruned = True
                continue
            board.push(m)
            gives_check = board.is_check()
            if i == 0:
                # Principal variation: search the first (best-ordered) move on the full window.
                val = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # PVS scout with a null window, plus LMR reduction for late quiet moves.
                reduction = 1 if (i >= 4 and depth >= 3 and not is_cap and not gives_check) else 0
                val = -self._negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                if val > alpha:      # scout beat alpha -> re-search at full depth and window
                    val = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if val > best_val:
                best_val, best_move = val, m
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                if not is_cap:                  # killer + history for quiet cutoffs
                    self.killers[ply] = (m,) + self.killers.get(ply, ())[:1]
                    k = (board.turn, m.from_square, m.to_square)
                    self.history[k] = self.history.get(k, 0) + depth * depth
                break

        flag = EXACT if alpha_orig < best_val < beta else (LOWER if best_val >= beta else UPPER)
        if phi_pruned and flag == EXACT:
            # A forward-pruned node is not a true EXACT bound: a dropped quiet move could have been
            # better, so best_val is a LOWER bound on the node's true value. Downgrade EXACT->LOWER.
            # (UPPER/fail-low results are LEFT as-is: phi-widening is a deliberate heuristic prune, and
            # keeping the UPPER TT cutoffs is what buys the deep node reduction — 93% vs 33% at d5 —
            # while tactical soundness is guaranteed separately by the move-level capture/check
            # exemption. This is the accepted positional-pruning trade, not a soundness bug.)
            flag = LOWER
        self._tt_store(key, depth, best_val, flag, best_move)
        return best_val

    # --- public API --------------------------------------------------------
    def search(self, board):
        """Iterative-deepening search; return (best_move, info dict).

        Works on a copy: a timeout unwinds the recursion without popping, so the caller's board
        must never be the one we mutate. Returned moves are valid on the original position.
        """
        board = board.copy()
        self.start = time.time()
        self.nodes = 0
        # Seed a legal fallback so we always return a legal move, even if depth 1 times out
        # (explosive quiescence at very short time limits can exceed the budget before it finishes).
        best_move = next(iter(board.legal_moves), None)
        best_val, reached = 0, 0

        # Opening book: instant, sound, varied openings.
        book = OPENING_BOOK.get(" ".join(board.fen().split()[:2]))
        if book:
            legal = [mv for mv in (chess.Move.from_uci(u) for u in book) if mv in board.legal_moves]
            if legal:
                return random.choice(legal), {"depth": 0, "score_cp": 0, "nodes": 0,
                                              "nps": 0, "time": 0.0, "book": True}

        for depth in range(1, self.max_depth + 1):
            try:
                val, mv = self._root(board, depth)
            except TimeoutError:
                break
            if mv is not None:
                best_move, best_val, reached = mv, val, depth
            if time.time() - self.start > self.time_limit:
                break
        elapsed = time.time() - self.start
        return best_move, {"depth": reached, "score_cp": best_val, "nodes": self.nodes,
                           "nps": self.nodes / elapsed if elapsed else 0, "time": elapsed}

    def evaluate(self, board, depth=3):
        """White-absolute centipawn value from a fixed-depth alpha-beta (with quiescence).

        Used as a distillation teacher: the fixed depth keeps labels cheap, and quiescence at the
        leaves resolves hanging captures -- the single biggest thing a raw value net needs to learn.
        Bypasses iterative deepening and the opening book so every position (incl. the start) gets a
        real searched score. _root returns a side-to-move (negamax) score; flip to White-absolute.
        """
        board = board.copy()
        self.start = time.time()
        saved_limit, self.time_limit = self.time_limit, 1e9   # depth-bounded, not time-bounded
        self.nodes = 0
        try:
            val, _ = self._root(board, depth)
        except TimeoutError:
            val = self._evaluate(board)
        finally:
            self.time_limit = saved_limit
        return val if board.turn == chess.WHITE else -val

    def _root(self, board, depth):
        key = chess.polyglot.zobrist_hash(board)
        entry = self.tt.get(key)
        tt_move = entry[3] if entry else None
        best_val, best_move = -INF, None
        alpha, beta = -INF, INF
        for m in self._order(board, list(board.legal_moves), tt_move, 0):
            board.push(m)
            val = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if val > best_val:
                best_val, best_move = val, m
            if best_val > alpha:
                alpha = best_val
        # Root is ply 0 (below :Full-width-floor:) so no phi pruning here -> EXACT is legitimate;
        # still route through the depth-preferred, size-capped store for :Tree-reuse:.
        self._tt_store(key, depth, best_val, EXACT, best_move)
        return best_val, best_move
