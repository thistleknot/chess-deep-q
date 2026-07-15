"""Encoder registry — ONE home for input manifolds (spec/bakeoff.spec.md), shared by
the trainer (qlearn.py) and the duel ruler (head2head.py `enc:` specs).

Every encoder is a pure function board -> float32 vector plus its dimension. Hand
feature DEFINITIONS only (purity law): no trained weights, no hand piece values in the
Merge-21 SME planes — the objective prices the conjunctions. Kanerva-family imports are
lazy (native/aux deps). ZCA/whitening wraps are applied by the CONSUMER (qlearn), not
here.

Failure modes: unknown name -> ValueError; Kanerva names raise ImportError if
kanerva_enc is unavailable.
"""
import numpy as np
import chess

from cem_loop import encode, encode_features, NIN, NFEAT


def _tpst(board, _e=encode, _n=NIN):
    # Merge 21 arm A :Threat-planes:: pst-769 base ⊕ piece × square × THREATENED
    # ⊕ × GUARDED (operator's highlight-engine semantics as interactions)
    x = np.zeros(_n + 2 * 768, dtype=np.float32)
    x[:_n] = _e(board)
    for sq, piece in board.piece_map().items():
        idx = (0 if piece.color == chess.WHITE else 384) + 64 * (piece.piece_type - 1) + sq
        if board.is_attacked_by(not piece.color, sq):
            x[_n + idx] = 1.0
        if board.is_attacked_by(piece.color, sq):
            x[_n + 768 + idx] = 1.0
    return x


def _hpst(board, _e=encode, _n=NIN):
    # Merge 21 arm B :Hanging-planes:: piece × square × (THREATENED AND NOT GUARDED)
    x = np.zeros(_n + 768, dtype=np.float32)
    x[:_n] = _e(board)
    for sq, piece in board.piece_map().items():
        if (board.is_attacked_by(not piece.color, sq)
                and not board.is_attacked_by(piece.color, sq)):
            x[_n + (0 if piece.color == chess.WHITE else 384)
              + 64 * (piece.piece_type - 1) + sq] = 1.0
    return x


def _amap(board, _e=encode, _n=NIN):
    # Merge 21 arm C :Attack-maps:: per-square any-attack binary per side (operator's
    # mobility/space/center features as LEARNABLE maps — no hand square-weights)
    x = np.zeros(_n + 128, dtype=np.float32)
    x[:_n] = _e(board)
    for sq in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, sq):
            x[_n + sq] = 1.0
        if board.is_attacked_by(chess.BLACK, sq):
            x[_n + 64 + sq] = 1.0
    return x


def _amaph(board, _a=_amap, _n=NIN):
    # Merge 21 :Composition-arm: (fires on the amap CONFIRMED verdict, 2026-07-15):
    # amap (897d, the confirmed winner) ⊕ hanging planes (768d, the leaning-tie arm B) —
    # tests whether hanging-piece conjunctions add on TOP of coverage maps
    x = np.zeros(_n + 128 + 768, dtype=np.float32)
    x[:_n + 128] = _a(board)
    off = _n + 128
    for sq, piece in board.piece_map().items():
        if (board.is_attacked_by(not piece.color, sq)
                and not board.is_attacked_by(piece.color, sq)):
            x[off + (0 if piece.color == chess.WHITE else 384)
              + 64 * (piece.piece_type - 1) + sq] = 1.0
    return x


def _kpst(board, _e=encode, _n=NIN):
    # Merge 20 :Kpst:: pst-769 planes per WHITE-king quadrant (file-half × rank-half)
    x = _e(board)
    k = board.king(chess.WHITE)
    b = 2 * (chess.square_file(k) >= 4) + (chess.square_rank(k) >= 4)
    out = np.zeros(4 * _n, dtype=np.float32)
    out[b * _n:(b + 1) * _n] = x
    return out


def get(name):
    """(enc_fn, nin) for an encoder name."""
    if name == "pst":
        return encode, NIN
    if name == "kc":
        return encode_features, NFEAT
    if name == "tpst":
        return _tpst, NIN + 2 * 768
    if name == "hpst":
        return _hpst, NIN + 768
    if name == "amap":
        return _amap, NIN + 128
    if name == "amaph":
        return _amaph, NIN + 128 + 768
    if name == "kpst":
        return _kpst, 4 * NIN
    if name == "hk":
        from kanerva_enc import active5k, K_IN

        def enc_fn(board, _a=active5k, _n=K_IN):   # _n bound at DEF time (hk lesson:
            x = np.zeros(_n, dtype=np.float32)     # ZCA wraps rebind consumer globals)
            x[_a(board)] = 1.0
            return x
        return enc_fn, K_IN
    if name == "kx":
        from kanerva_enc import encode_kanerva809 as _kx8

        def enc_fn(board, _k=_kx8, _e=encode_features):
            x = _e(board)
            return np.concatenate([x, _k(x)])
        return enc_fn, 809 + 2048
    if name == "k8":
        from kanerva_enc import encode_kanerva809 as _k8, K8_OUT
        return (lambda board, _k=_k8, _e=encode_features: _k(_e(board))), K8_OUT
    if name == "nk":
        from kanerva_enc import encode_kanerva, K_OUT
        return encode_kanerva, K_OUT
    raise ValueError(f"unknown encoder '{name}' "
                     "(pst | kc | tpst | hpst | amap | amaph | kpst | hk | kx | k8 | nk)")
