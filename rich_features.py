"""Richer linear feature vector φ (spec/linear-value-rl.spec.md :Feature-set-design:).

FIRST 29 dims == `gbdt_features.features` (material + PST + stm + castling) so `models/linear.npz`
partially warm-starts. Then a SEARCH-COMPLEMENTARY positional block (mobility, king safety, pawn
structure incl. passed pawns, space, coordination, bishop pair) + §9.5.1 interaction products — all as
SIGNED white−black terms (baking in the eval's antisymmetry: eval(mirror) = −eval, half the params).
Tactical features (threats/hangs/captures/checks) are DELIBERATELY EXCLUDED — the search computes those.
"""
import chess
import numpy as np

from gbdt_features import features as _gbdt

CASTLED_W = {chess.G1, chess.C1}
CASTLED_B = {chess.G8, chess.C8}
CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
CENTER_SQ = [chess.D4, chess.D5, chess.E4, chess.E5]
EXT_CENTER = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D6,
              chess.E3, chess.E6, chess.F3, chess.F4, chess.F5, chess.F6]


def _passed_count(board, color):
    """Pawns with no enemy pawn on the same or adjacent file ahead of them (a long-game asset)."""
    pawns = board.pieces(chess.PAWN, color)
    enemy = list(board.pieces(chess.PAWN, not color))
    n = 0
    for p in pawns:
        f, r = chess.square_file(p), chess.square_rank(p)
        blocked = any(abs(chess.square_file(e) - f) <= 1 and
                      (chess.square_rank(e) > r if color == chess.WHITE else chess.square_rank(e) < r)
                      for e in enemy)
        if not blocked:
            n += 1
    return n


def _positional(board):
    """15 SIGNED (white−black) search-complementary positional features + interactions."""
    W, B = chess.WHITE, chess.BLACK
    wk, bk = board.king(W), board.king(B)
    wk_att = 1.0 if (wk is not None and board.is_attacked_by(B, wk)) else 0.0
    bk_att = 1.0 if (bk is not None and board.is_attacked_by(W, bk)) else 0.0
    wk_cast = 1.0 if wk in CASTLED_W else 0.0
    bk_cast = 1.0 if bk in CASTLED_B else 0.0
    wk_cen = 1.0 if wk in CENTER else 0.0
    bk_cen = 1.0 if bk in CENTER else 0.0

    wp = list(board.pieces(chess.PAWN, W)); bp = list(board.pieces(chess.PAWN, B))
    wf = [chess.square_file(s) for s in wp]; bf = [chess.square_file(s) for s in bp]
    w_dbl = sum(wf.count(f) - 1 for f in set(wf)); b_dbl = sum(bf.count(f) - 1 for f in set(bf))
    w_iso = sum(1 for f in wf if f - 1 not in wf and f + 1 not in wf)
    b_iso = sum(1 for f in bf if f - 1 not in bf and f + 1 not in bf)
    w_pass = _passed_count(board, W); b_pass = _passed_count(board, B)

    w_cc = sum(1 for sq in CENTER_SQ if board.is_attacked_by(W, sq))
    b_cc = sum(1 for sq in CENTER_SQ if board.is_attacked_by(B, sq))
    w_ec = sum(1 for sq in EXT_CENTER if board.is_attacked_by(W, sq))
    b_ec = sum(1 for sq in EXT_CENTER if board.is_attacked_by(B, sq))

    w_def = sum(1 for sq, p in board.piece_map().items() if p.color == W and board.is_attacked_by(W, sq))
    b_def = sum(1 for sq, p in board.piece_map().items() if p.color == B and board.is_attacked_by(B, sq))

    w_bp = 1.0 if len(board.pieces(chess.BISHOP, W)) >= 2 else 0.0
    b_bp = 1.0 if len(board.pieces(chess.BISHOP, B)) >= 2 else 0.0

    # interactions (§9.5.1 products; already signed as white-advantage)
    wq = 1.0 if board.pieces(chess.QUEEN, W) else 0.0
    bq = 1.0 if board.pieces(chess.QUEEN, B) else 0.0
    w_exp = 1.0 if (wk_cen or wk_att) else 0.0
    b_exp = 1.0 if (bk_cen or bk_att) else 0.0
    openness = (16 - (len(wp) + len(bp))) / 16.0
    nonpawn = sum(len(board.pieces(pt, c)) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                  for c in (W, B))
    phase = max(0.0, 1.0 - nonpawn / 14.0)         # endgame_phase: 0 opening → 1 bare
    i_king = b_exp * wq - w_exp * bq               # white gains if black king exposed & white has queen
    i_bishop = (w_bp - b_bp) * openness            # bishop pair worth more in open positions
    i_passed = (w_pass - b_pass) * phase           # passers worth more in the endgame

    return np.array([
        wk_att - bk_att, wk_cast - bk_cast, wk_cen - bk_cen,       # king safety
        w_dbl - b_dbl, w_iso - b_iso, w_pass - b_pass,             # pawn structure
        w_cc - b_cc, w_ec - b_ec,                                  # space
        w_def - b_def,                                             # coordination
        w_bp - b_bp,                                               # bishop pair
        i_king, i_bishop, i_passed,                                # interactions
    ], dtype=np.float32)


def features(board):
    """φ = [gbdt 29-dim block] ++ [13 signed positional/interaction dims] = 42 dims, White-absolute.

    Search-complementary only: king safety, pawn structure (incl. passed), space, coordination,
    bishop pair, + §9.5.1 interactions. Mobility / whole-board control dropped (expensive AND closest
    to what search already sees); tactics excluded by design (:Feature-set-design:)."""
    return np.concatenate([_gbdt(board), _positional(board)]).astype(np.float32)
