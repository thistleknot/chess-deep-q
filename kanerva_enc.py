"""Kanerva coding over the paper's 5k king-relative features (operator's feature pipeline:
5k -> Kanerva -> ZCA).

Layer 1 — :Features-5k:: HalfKP-lite with EIGHT king buckets (the paper-scale variant of
nnue_model.features): bucket = (rank>=4)*4 + file//2, x 64 squares x 10 piece types = 5120
binary features, + 1 side-to-move bit = 5121. Hand-DEFINED (provenance class: feature
definitions); no trained weights touched.

Layer 2 — :Kanerva:: sparse distributed coding. K_OUT random prototypes (RNG seed 11 —
declared), each a random subset of the 5121 dims at DENSITY; output_j = |active(x) ∩
prototype_j| (integer overlap counts, computed sparsely from the ≤31 active indices via a
prototype-membership table). This is the reduction: 5121 -> K_OUT.

Layer 3 — ZCA over the Kanerva outputs (build_kanerva_zca.py, own/random-play corpus) is
applied by qlearn's existing QLEARN_ZCA wrap; this module stops at layer 2.

Failure modes: none silent — dims and density are module constants (declared infrastructure).
"""
import chess
import numpy as np

K_IN = 8 * 64 * 10 + 1        # 5121: 8 king buckets x 64 sq x 10 piece types + side-to-move
K_OUT = 512                    # Kanerva prototypes (declared)
DENSITY = 1.0 / 32.0           # fraction of input dims each prototype samples (~160 bits)
_STM = K_IN - 1

_PC = {(pt, col): (pt - 1) + (0 if col == chess.WHITE else 5)
       for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
       for col in (chess.WHITE, chess.BLACK)}

# prototype membership: for each input dim, the list of prototypes containing it.
_rng = np.random.default_rng(11)
_mask = _rng.random((K_OUT, K_IN)) < DENSITY          # K_OUT x K_IN boolean (declared RNG)
_members = [np.flatnonzero(_mask[:, d]).astype(np.int32) for d in range(K_IN)]


def active5k(board):
    """Sparse active indices in the 5121-dim space (<= 31)."""
    ksq = board.king(chess.WHITE)
    kb = 0 if ksq is None else ((1 if (ksq >> 3) >= 4 else 0) * 4 + ((ksq & 7) >> 1))
    base = kb * 640
    idx = [base + sq * 10 + _PC[(p.piece_type, p.color)]
           for sq, p in board.piece_map().items() if p.piece_type != chess.KING]
    if board.turn == chess.WHITE:
        idx.append(_STM)
    return idx


def encode_kanerva(board):
    """K_OUT-dim overlap counts — the Kanerva code of the position (float32)."""
    out = np.zeros(K_OUT, dtype=np.float32)
    for d in active5k(board):
        out[_members[d]] += 1.0
    return out


# --- :Kanerva-809: (bake set #5) — EXPANSION over the proven 809 donor features -----------
# More prototypes than inputs: each output is a sparse random projection (soft conjunction)
# of the 809 — nonlinearity without hidden layers (classic SDM use). Declared constants.
K8_OUT = 2048
_rng8 = np.random.default_rng(13)
_M8 = (_rng8.random((K8_OUT, 809)) < (1.0 / 32.0)).astype(np.float32)


def encode_kanerva809(x809):
    """809 raw features -> 2048 overlap/projection counts."""
    return _M8 @ x809
