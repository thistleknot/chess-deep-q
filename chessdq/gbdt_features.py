"""Dense, fast hand-features for the GBDT eval (shared by train + inference).

Raw piece-square planes make a tree compute material via 64 splits; explicit material counts + PST
sums give it material/position for free and are cheap (O(pieces)). ~29 features. The eval trains on
raw centipawns (not the squashed tanh target) so material is linearly scaled.
"""
import chess
import numpy as np

from chessdq.engine import _PST

PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def features(board):
    f = []
    # Material counts per piece type per colour (12).
    for pt in PT:
        f.append(len(board.pieces(pt, chess.WHITE)))
        f.append(len(board.pieces(pt, chess.BLACK)))
    # PST sums per piece type, White and Black separately (12), matching engine.pst_eval convention
    # (_PST is a8-first: White square sq -> index (7-rank)*8+file; Black square -> rank*8+file).
    for pt in PT:
        table = _PST[pt]
        w = sum(table[(7 - (sq >> 3)) * 8 + (sq & 7)] for sq in board.pieces(pt, chess.WHITE))
        b = sum(table[(sq >> 3) * 8 + (sq & 7)] for sq in board.pieces(pt, chess.BLACK))
        f.append(w); f.append(b)
    # Side to move + castling rights (5).
    f.append(1.0 if board.turn == chess.WHITE else 0.0)
    f.append(float(board.has_kingside_castling_rights(chess.WHITE)))
    f.append(float(board.has_queenside_castling_rights(chess.WHITE)))
    f.append(float(board.has_kingside_castling_rights(chess.BLACK)))
    f.append(float(board.has_queenside_castling_rights(chess.BLACK)))
    return np.array(f, dtype=np.float32)


def cp_from_tanh(v, clip_v=0.995, clip_cp=1500.0):
    """Recover White-absolute centipawns from a stored tanh(cp/400) label."""
    import math
    v = min(max(float(v), -clip_v), clip_v)
    return max(-clip_cp, min(clip_cp, 400.0 * math.atanh(v)))
