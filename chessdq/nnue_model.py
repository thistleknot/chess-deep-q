"""NNUE-style fast, hole-free leaf eval (spec/nnue-eval.spec.md), v1 — king-bucketed HalfKP-lite.

:Feature-set: each non-king piece fires one sparse index = king_bucket*(64*10) + piece_sq*10 +
piece_type_colour, so a piece's value is conditioned on the White king's zone (the hole-free
property). :NNUE-net: EmbeddingBag(sum) = the :Accumulator: → clipped-ReLU → tiny head → White-
absolute centipawns, so it drops into AlphaBetaEngine(eval_fn=...) exactly like pst_eval.

v1 is recompute-per-leaf (no incremental accumulator yet). Device-agnostic — the :NNUE-kill-check:
judges speed at equal time, not the device.
"""
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NNUE_KING_BUCKETS = 4          # White king quadrant: (rank>=4)*2 + (file>=4)
NNUE_PIECE_TYPES = 10          # {P,N,B,R,Q} x {white, black}; kings define the bucket
NNUE_ACC_DIM = 128             # accumulator width
NNUE_HIDDEN = 32
NUM_FEATURES = NNUE_KING_BUCKETS * 64 * NNUE_PIECE_TYPES   # 2560

# (piece_type, color) -> 0..9
_PC = {(pt, col): (pt - 1) + (0 if col == chess.WHITE else 5)
       for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
       for col in (chess.WHITE, chess.BLACK)}


def _king_bucket(board):
    ksq = board.king(chess.WHITE)
    if ksq is None:
        return 0
    return (1 if (ksq >> 3) >= 4 else 0) * 2 + (1 if (ksq & 7) >= 4 else 0)


def features(board):
    """Sparse active feature indices for a position (≤ 30). King-bucketed by the White king."""
    kb = _king_bucket(board) * (64 * NNUE_PIECE_TYPES)
    idx = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        idx.append(kb + sq * NNUE_PIECE_TYPES + _PC[(piece.piece_type, piece.color)])
    return idx


class NNUENet(nn.Module):
    """EmbeddingBag accumulator + a tiny head → White-absolute centipawns."""

    def __init__(self, acc_dim=NNUE_ACC_DIM, hidden=NNUE_HIDDEN):
        super().__init__()
        self.acc = nn.EmbeddingBag(NUM_FEATURES, acc_dim, mode="sum")
        self.h1 = nn.Linear(acc_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feats, offsets):
        a = torch.clamp(self.acc(feats, offsets), 0.0, 1.0)   # clipped-ReLU accumulator (NNUE)
        h = F.relu(self.h1(a))
        return self.out(h).squeeze(1)                          # White-absolute centipawns


def load_nnue(path="models/nnue.pt", device="cpu"):
    ck = torch.load(path, map_location=device)
    net = NNUENet(ck.get("acc_dim", NNUE_ACC_DIM), ck.get("hidden", NNUE_HIDDEN)).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def make_nnue_eval(net, device="cpu"):
    """Return an eval_fn(board) -> White-absolute centipawns (recompute-per-leaf)."""
    @torch.no_grad()
    def nnue_eval(board):
        idx = features(board)
        if not idx:
            return 0.0
        feats = torch.tensor(idx, dtype=torch.long, device=device)
        offsets = torch.zeros(1, dtype=torch.long, device=device)
        return float(net(feats, offsets).item())
    return nnue_eval


# ---------------------------------------------------------------------------
# :Accumulator: — the incremental first-layer sum (spec/nnue-eval.spec.md, Stage 3).
#
# The recompute path (make_nnue_eval) re-sums ~30 embedding rows AND pays torch's
# per-call kernel-launch/dispatch overhead at EVERY leaf (~281-500 us/call). That
# overhead caps search depth at a fixed time budget — the binding strength constraint.
#
# The fix: hold A = sum(W[:, f]) for the active features f, and on each move UPDATE it
# in O(changed) — subtract the moved/captured piece's OLD feature rows, add its NEW
# rows — instead of re-summing all rows. A White-king move that changes the king bucket
# shifts EVERY feature index, so that side is fully recomputed (standard NNUE refresh).
# The tiny head (clipped-ReLU -> Linear -> Linear) then runs on the maintained A, in
# pure numpy, with no torch dispatch. Correctness INVARIANT: the incrementally
# maintained A equals the full recompute at every position (test_accumulator.py).
# ---------------------------------------------------------------------------

_FEAT_STRIDE = 64 * NNUE_PIECE_TYPES   # index contribution of one king-bucket step


def _bucket_of_sq(sq):
    """White-king bucket for a king on `sq` — matches _king_bucket's quadrant scheme."""
    return (1 if (sq >> 3) >= 4 else 0) * 2 + (1 if (sq & 7) >= 4 else 0)


def extract_nnue_arrays(net):
    """Pull the net's weights into contiguous float32 numpy arrays for launch-free CPU eval.

    Require: `net` is an NNUENet (EmbeddingBag accumulator + h1 + out).
    Guarantee: returns (W, h1_w, h1_b, out_w, out_b) as float32 ndarrays, where
      A = W[active_features].sum(0) reproduces the EmbeddingBag(mode="sum") accumulator.
    """
    with torch.no_grad():
        W = net.acc.weight.detach().cpu().numpy().astype(np.float32)      # (NUM_FEATURES, acc_dim)
        h1_w = net.h1.weight.detach().cpu().numpy().astype(np.float32)    # (hidden, acc_dim)
        h1_b = net.h1.bias.detach().cpu().numpy().astype(np.float32)      # (hidden,)
        out_w = net.out.weight.detach().cpu().numpy().astype(np.float32)  # (1, hidden)
        out_b = net.out.bias.detach().cpu().numpy().astype(np.float32)    # (1,)
    return W, h1_w, h1_b, out_w, out_b


class IncrementalNNUE:
    """Stateful :NNUE-eval: that maintains the :Accumulator: across make/unmake.

    Keeps its OWN 64-square mailbox (`mb`) plus the first-layer sum `acc`, kept in lockstep
    by applying/reverting move deltas — it never mutates a python-chess Board in the hot
    path (that push/pop cost is what a per-node eval cannot afford). On each __call__(board)
    it syncs to the target board's move_stack: unmake to the common prefix, then apply each
    new move's O(changed) feature delta, then run the tiny head on the maintained `acc` in
    pure numpy (no torch dispatch).

    Drop-in for AlphaBetaEngine.eval_fn: __call__(board) -> White-absolute centipawns,
    same value frame as make_nnue_eval (the recompute reference).

    Require: successive calls come from a search on ONE board object pushed/popped in the
             alpha-beta pattern; reset(root_board) anchors the mailbox to the search root
             (the engine calls it; the invariant test calls it explicitly).
    Guarantee: the returned value equals make_nnue_eval(net)(board) at every position
               (fp tolerance) — the :Accumulator: invariant.
    Maintain: mb/acc/kb/nfeat/wksq are a consistent snapshot of board.move_stack[:base_len]
              + self.applied; len(self.undo) == len(self.applied).
    """

    def __init__(self, net):
        W, h1_w, h1_b, out_w, out_b = extract_nnue_arrays(net)
        self.W = np.ascontiguousarray(W)                 # (NUM_FEATURES, acc_dim), row-gather friendly
        self.acc_dim = W.shape[1]
        hidden = h1_w.shape[0]
        # Head-1 with the bias FOLDED into an extra input column (augmented-1 trick): one matvec
        # over [clip(acc); 1] computes h1_w @ a + h1_b with no separate add call.
        self._h1_wb = np.ascontiguousarray(
            np.concatenate([h1_w, h1_b[:, None]], axis=1))   # (hidden, acc_dim + 1)
        self._out_wf = np.ascontiguousarray(out_w.reshape(-1))   # (hidden,)
        self._out_bf = float(out_b[0])                            # scalar, folded in Python
        # Preallocated head buffers (reused every call -> no per-eval allocation).
        self._aug = np.ones(self.acc_dim + 1, dtype=np.float32)   # last element stays 1.0
        self._hbuf = np.empty(hidden, dtype=np.float32)
        # Signed-gather sign vectors, cached by (n_removed, n_added): delta = signs @ W[idxs].
        self._sign_cache = {}
        self.reset(chess.Board())

    def _signs(self, nr, na):
        s = self._sign_cache.get((nr, na))
        if s is None:
            s = np.array([-1.0] * nr + [1.0] * na, dtype=np.float32)
            self._sign_cache[(nr, na)] = s
        return s

    # --- full recompute of the accumulator from the mailbox ----------------
    def _acc_from_mb(self):
        base = self.kb * _FEAT_STRIDE
        mb = self.mb
        idx = [base + sq * NNUE_PIECE_TYPES + _PC[p]
               for sq, p in enumerate(mb) if p is not None and p[0] != chess.KING]
        if not idx:
            return np.zeros(self.acc_dim, dtype=np.float32), 0
        return self.W[idx].sum(0).astype(np.float32, copy=False), len(idx)

    def reset(self, board):
        """Re-anchor the mailbox/accumulator to `board` as the search root (full refresh).

        Engine hook — called at the top of AlphaBetaEngine.search()/evaluate() so the
        move_stack diffing in __call__ is relative to the searched root.
        """
        self.base_len = len(board.move_stack)
        self.applied = []                # moves applied since anchor (mirror of move_stack[base_len:])
        self.undo = []                   # parallel undo records (None == null move)
        self.mb = [None] * 64            # mailbox: sq -> (piece_type, color) or None
        for sq, piece in board.piece_map().items():
            self.mb[sq] = (piece.piece_type, piece.color)
        self.wksq = board.king(chess.WHITE)
        self.kb = _king_bucket(board)
        self.acc, self.nfeat = self._acc_from_mb()

    # --- head (launch-free numpy forward over the maintained accumulator) --
    def _value(self):
        # Reference (make_nnue_eval) returns 0.0 when there are no non-king pieces.
        if self.nfeat == 0:
            return 0.0
        # 4 numpy calls total: clip -> matvec(folded bias) -> relu -> matvec. No allocation.
        np.clip(self.acc, 0.0, 1.0, out=self._aug[:self.acc_dim])   # clipped-ReLU into aug[:d]
        h = np.dot(self._h1_wb, self._aug, out=self._hbuf)           # h1_w@a + h1_b (bias folded)
        np.maximum(h, 0.0, out=h)                                    # F.relu
        return float(np.dot(self._out_wf, h)) + self._out_bf         # White-absolute centipawns

    # --- incremental make: apply one move's feature delta ------------------
    def _apply(self, m):
        """Advance the mailbox/accumulator by `m` (O(changed) delta, or full-refresh on
        a White-king bucket change). Precondition: state is the position BEFORE `m`."""
        if not m:                                    # null move: features unchanged
            self.undo.append(None)
            self.applied.append(m)
            return

        mb = self.mb
        from_sq, to_sq, promo = m.from_square, m.to_square, m.promotion
        mpt, mcol = mb[from_sq]
        moving_is_wk = (from_sq == self.wksq)
        kb_before = self.kb
        kb_after = _bucket_of_sq(to_sq) if moving_is_wk else kb_before
        refresh = (kb_after != kb_before)
        base = kb_before * _FEAT_STRIDE

        changes = []                                 # (sq, old_value) to revert mb on unmake
        removed = []
        added = []
        n_captured = 0

        # Capture (en passant victim sits on a different square than to_sq).
        if mpt == chess.PAWN and (from_sq & 7) != (to_sq & 7) and mb[to_sq] is None:
            cap_sq = to_sq + (-8 if mcol == chess.WHITE else 8)
            victim = mb[cap_sq]
            changes.append((cap_sq, victim))
            if not refresh:
                removed.append(base + cap_sq * NNUE_PIECE_TYPES + _PC[victim])
            mb[cap_sq] = None
            n_captured = 1
        elif mb[to_sq] is not None:                  # ordinary capture (victim on to_sq)
            if not refresh:
                removed.append(base + to_sq * NNUE_PIECE_TYPES + _PC[mb[to_sq]])
            n_captured = 1

        # Move the piece (record old contents of both squares for unmake).
        changes.append((from_sq, mb[from_sq]))
        changes.append((to_sq, mb[to_sq]))
        new_pt = promo if promo else mpt
        mb[from_sq] = None
        mb[to_sq] = (new_pt, mcol)

        if mpt != chess.KING:
            if not refresh:
                removed.append(base + from_sq * NNUE_PIECE_TYPES + _PC[(mpt, mcol)])
                added.append(base + to_sq * NNUE_PIECE_TYPES + _PC[(new_pt, mcol)])
        elif abs((to_sq & 7) - (from_sq & 7)) == 2:  # castling: the rook relocates too
            rank = from_sq >> 3
            if (to_sq & 7) > (from_sq & 7):          # kingside: rook h -> f
                rf, rt = chess.square(7, rank), chess.square(5, rank)
            else:                                     # queenside: rook a -> d
                rf, rt = chess.square(0, rank), chess.square(3, rank)
            rook = mb[rf]
            changes.append((rf, mb[rf]))
            changes.append((rt, mb[rt]))
            mb[rf] = None
            mb[rt] = rook
            if not refresh:
                removed.append(base + rf * NNUE_PIECE_TYPES + _PC[(chess.ROOK, mcol)])
                added.append(base + rt * NNUE_PIECE_TYPES + _PC[(chess.ROOK, mcol)])

        # Snapshot the pre-move scalars, then commit the new state.
        nfeat_prev, wksq_prev = self.nfeat, self.wksq
        self.applied.append(m)
        if moving_is_wk:
            self.wksq = to_sq
        if refresh:
            # King bucket changed -> every index shifted -> full recompute; keep the old acc for
            # exact unmake (reference restore, no fp drift).
            acc_prev = self.acc
            self.kb = kb_after
            self.acc, self.nfeat = self._acc_from_mb()
            self.undo.append((True, acc_prev, kb_before, nfeat_prev, wksq_prev, changes))
        else:
            # One fused signed gather (delta = signs @ W[idxs]) applied IN PLACE -> no acc.copy().
            idxs = removed + added
            if idxs:
                delta = np.dot(self._signs(len(removed), len(added)), self.W[idxs])
                self.acc += delta
            else:
                delta = None                          # e.g. a quiet same-bucket king move
            self.nfeat = nfeat_prev - n_captured
            self.undo.append((False, delta, kb_before, nfeat_prev, wksq_prev, changes))

    def _unapply(self):
        """Revert the last applied move (O(changed))."""
        rec = self.undo.pop()
        self.applied.pop()
        if rec is None:                              # null move
            return
        is_refresh, payload, kb_prev, nfeat_prev, wksq_prev, changes = rec
        if is_refresh:
            self.acc = payload                       # restore the pre-refresh accumulator array
        elif payload is not None:
            self.acc -= payload                      # reverse the in-place delta
        self.kb, self.nfeat, self.wksq = kb_prev, nfeat_prev, wksq_prev
        mb = self.mb
        for sq, old in changes:
            mb[sq] = old

    # --- sync the mailbox/accumulator to an arbitrary target board ---------
    def _sync(self, board):
        tgt = board.move_stack
        if len(tgt) < self.base_len:                 # unwound below the anchor -> re-anchor
            self.reset(board)
            return
        suffix = tgt[self.base_len:]
        cur = self.applied
        n = 0
        lim = min(len(cur), len(suffix))
        while n < lim and cur[n] == suffix[n]:
            n += 1
        while len(self.applied) > n:
            self._unapply()
        for i in range(n, len(suffix)):
            self._apply(suffix[i])

    def __call__(self, board):
        self._sync(board)
        return self._value()


def make_incremental_nnue_eval(net, device="cpu"):
    """Return a callable incremental eval_fn(board) -> White-absolute centipawns.

    Same signature/value frame as make_nnue_eval, but maintains the :Accumulator: across
    make/unmake instead of recomputing per leaf. `device` is accepted for API symmetry;
    inference is CPU/numpy (launch-free), the fast path for per-node alpha-beta.
    """
    return IncrementalNNUE(net)
