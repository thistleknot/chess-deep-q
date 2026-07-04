"""NNUE-style fast, hole-free leaf eval (spec/nnue-eval.spec.md), v1 — king-bucketed HalfKP-lite.

:Feature-set: each non-king piece fires one sparse index = king_bucket*(64*10) + piece_sq*10 +
piece_type_colour, so a piece's value is conditioned on the White king's zone (the hole-free
property). :NNUE-net: EmbeddingBag(sum) = the :Accumulator: → clipped-ReLU → tiny head → White-
absolute centipawns, so it drops into AlphaBetaEngine(eval_fn=...) exactly like pst_eval.

v1 is recompute-per-leaf (no incremental accumulator yet). Device-agnostic — the :NNUE-kill-check:
judges speed at equal time, not the device.
"""
import chess
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
