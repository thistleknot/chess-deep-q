"""Label-augmentation transforms (Stage-1 :Label-augmentation:, spec teacher-distillation).

Chess is NOT 8-fold symmetric: pawns have a direction and castling breaks left-right, so
only two symmetries yield valid relabelings of a distillation row. These are pure functions
on a label dict {"fen", "value", "wdl", "policy"}; Stage-2 training calls them at load time
so the on-disk corpus stays compact.

  mirror(row)      HORIZONTAL mirror (files a<->h). value and wdl UNCHANGED, policy move
                   coordinates file-mirrored. An involution: mirror(mirror(row)) == row for
                   positions without castling rights (a horizontal mirror moves the king off
                   the e-file, so standard-FEN castling rights are not preserved — this is
                   exactly why "castling breaks left-right").
  color_flip(row)  COLOR-FLIP (swap piece colors + vertical flip + toggle side to move).
                   value and cp SIGN-NEGATED, wdl reversed (w<->l), policy move coordinates
                   vertically mirrored. A true symmetry (castling rights preserved).

Both keep any extra keys (e.g. "non_quiet") intact.
"""

import chess

AUGMENT_ENABLED = True


def _flip_file(square):
    """Horizontal mirror of a single square (file a<->h, rank unchanged)."""
    return chess.square(7 - chess.square_file(square), chess.square_rank(square))


def _flip_rank(square):
    """Vertical mirror of a single square (rank 1<->8, file unchanged)."""
    return chess.square_mirror(square)


def _transform_uci(uci, sq_fn):
    """Apply a per-square transform to a UCI move's endpoints, preserving promotion."""
    move = chess.Move.from_uci(uci)
    if move.from_square == move.to_square:  # null move guard
        return uci
    return chess.Move(sq_fn(move.from_square), sq_fn(move.to_square), promotion=move.promotion).uci()


def _transform_policy(policy, sq_fn):
    """Transform every move UCI in a policy list, leaving probabilities untouched."""
    if not policy:
        return policy
    return [[_transform_uci(uci, sq_fn), p] for uci, p in policy]


def mirror(row):
    """Horizontal-mirror a label row (files a<->h). value/wdl unchanged; policy file-mirrored.

    Returns None when the position has castling rights: a horizontal flip moves the king off
    the e-file, so python-chess drops the rights and the mirrored FEN is NO LONGER value-
    equivalent to the original (castling is now impossible). Mirror is a valid relabeling only
    for no-castling positions; the Stage-2 loader skips a None. color_flip has no such limit.
    """
    board = chess.Board(row["fen"])
    if board.castling_rights:
        return None
    board = board.transform(chess.flip_horizontal)
    out = dict(row)
    out["fen"] = board.fen()
    out["policy"] = _transform_policy(row.get("policy"), _flip_file)
    return out


def color_flip(row):
    """Color-flip a label row: swap colors + vertical flip + toggle stm.

    value/cp sign-negated, wdl reversed (w<->l), policy move coordinates vertically mirrored.
    """
    board = chess.Board(row["fen"]).mirror()
    out = dict(row)
    out["fen"] = board.fen()
    out["value"] = -row["value"]
    wdl = row.get("wdl")
    out["wdl"] = None if wdl is None else [wdl[2], wdl[1], wdl[0]]
    out["policy"] = _transform_policy(row.get("policy"), _flip_rank)
    return out
