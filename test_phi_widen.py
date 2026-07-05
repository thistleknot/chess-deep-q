"""Tests for the :Deployed-agent: :Phi-widening: search (spec/search-mcts.spec.md).

Covers the three claims the spec pins on the sound alpha-beta realization:

  (a) :Phi-widening: SOUNDNESS -- on tactical/refutation positions, phi_widen=True returns the SAME
      best move as full-width (phi_widen=False) at a fixed depth. Captures and checks are exempt
      from the width budget, so no refutation is pruned; this is what distinguishes the sound
      alpha-beta realization from the sampled beam.
  (b) :Tree-reuse: -- one persistent engine reused across successive search() calls accumulates a
      non-empty, growing Zobrist-keyed transposition table (prior computation carried forward).
  (c) :Phi-widening: FIRES -- on a rich midgame position at equal fixed depth, phi_widen=True
      searches strictly fewer nodes than full-width (the forward pruning actually prunes).

Fail-fast: no try/except swallowing. Run: `python test_phi_widen.py`.
"""

import time

import chess

from engine import AlphaBetaEngine, PHI_FULL_WIDTH_FLOOR, phi_width, _PHI_FIB, INF


# Free queen en prise (the historical failing position): White exd5 wins the queen -- a CAPTURE,
# hence exempt from the phi budget, so phi and full-width must agree. UNIQUE decisive best move
# (+900 margin), so argmax has no tie to break arbitrarily.
FREE_QUEEN_W = "rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
# Mirror: Black to move, White queen hanging on d4, e5xd4 wins it (:Negamax-backup-convention: for
# the other color, still a capture-decided position).
FREE_QUEEN_B = "rnb1kbnr/pppp1ppp/8/4p3/3Q4/8/PPPP1PPP/RNB1KBNR b KQkq - 0 1"
# Knight forks nothing -- it simply wins a hanging queen: Nc3xd5 (+900), a CAPTURE, unique best.
NXQ = "4k3/8/8/3q4/8/2N5/8/4K3 w - - 0 1"
# Trap shape (poisoned pawn): White Qxb7?? is refuted by ...Bxb7 (the c8 bishop recaptures -- a
# CAPTURE, exempt from the phi budget), so the queen is lost for a pawn. The move's eval right after
# the capture is +1 pawn, but its backed-up value is ~ -800: the refutation is never pruned, so
# BOTH phi and full-width must back it up to a loss. (Move choice here ties among quiet moves and is
# NOT asserted -- only the poisoned move's backed-up value.)
TRAP_QXB7 = "2bqk3/1p6/8/8/8/1Q6/8/4K3 w - - 0 1"
TRAP_MOVE = chess.Move.from_uci("b3b7")   # Qxb7

# Rich midgame with many legal moves, so the phi budget bites at ply >= FLOOR.
MIDGAME = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"


def _fixed_depth_search(eng, board, depth):
    """Deterministic depth-bounded root search returning (best_move, nodes).

    Bypasses iterative deepening / the opening book / the time budget the way evaluate() does, so
    phi_widen=True and phi_widen=False are compared at the SAME nominal depth. Uses a fresh engine
    per call site (empty TT) so node counts are not perturbed by carried entries.
    """
    b = board.copy()
    eng.start = time.time()
    saved, eng.time_limit = eng.time_limit, 1e9
    eng.nodes = 0
    try:
        _, mv = eng._root(b, depth)
    finally:
        eng.time_limit = saved
    return mv, eng.nodes


def _move_value_white(eng, board, move, depth):
    """White-absolute backed-up value of a specific root `move` at fixed depth.

    Pushes `move`, then negamaxes the resulting (opponent-to-move) position and flips to White's
    frame. Used to prove a poisoned capture backs up to a loss because the exempt recapture is
    always searched (:Phi-widening: refutation preservation).
    """
    b = board.copy()
    eng.start = time.time()
    saved, eng.time_limit = eng.time_limit, 1e9
    eng.nodes = 0
    try:
        b.push(move)
        # _negamax returns side-to-move-relative; after the push it is the opponent's frame, so
        # negate to get the mover-of-`move`'s value, which is White here (board.turn was White).
        v = -eng._negamax(b, depth - 1, -INF, INF, 1)
    finally:
        eng.time_limit = saved
    return v


def test_phi_width_helper():
    """phi_width is wide near the root (large remaining depth) narrowing toward the leaves."""
    assert _PHI_FIB[0] == 2 and phi_width(0) == 2, "leaf budget floors at 2"
    assert phi_width(1) == 3 and phi_width(2) == 5, "Fibonacci spacing"
    # Monotone non-decreasing in remaining depth (wide near root).
    widths = [phi_width(d) for d in range(0, 12)]
    assert widths == sorted(widths), f"phi_width must not narrow as depth grows: {widths}"
    assert phi_width(999) == _PHI_FIB[-1], "clamps to table length"
    print("  [ok] phi_width: wide near root -> 2 deep, clamped, monotone")


def test_soundness_same_best_move():
    """(a) On decisive tactical positions, phi_widen=True == phi_widen=False best move.

    Each fixture has a UNIQUE winning capture (large margin), so the refutation-exempt capture is
    always searched and the argmax is unambiguous -- no tie for phi/full to break differently.
    """
    depth = 5
    for fen in (FREE_QUEEN_W, FREE_QUEEN_B, NXQ):
        board = chess.Board(fen)
        full = AlphaBetaEngine(phi_widen=False, max_depth=depth)
        phi = AlphaBetaEngine(phi_widen=True, max_depth=depth)
        mv_full, _ = _fixed_depth_search(full, board, depth)
        mv_phi, _ = _fixed_depth_search(phi, board, depth)
        assert mv_full is not None and mv_phi is not None, f"no move on {fen}"
        assert mv_full == mv_phi, (
            f"SOUNDNESS VIOLATION on {fen}: full-width picked {mv_full}, "
            f"phi-widen picked {mv_phi} (a refutation was pruned)")
        assert board.is_capture(mv_phi), f"expected the winning capture on {fen}, got {mv_phi}"
    assert chess.Board(FREE_QUEEN_W).is_capture(chess.Move.from_uci("e4d5"))
    print(f"  [ok] soundness: phi == full best move on all 3 decisive positions (depth {depth})")


def test_trap_refutation_preserved():
    """(a') The trap shape: a poisoned capture backs up to a LOSS under BOTH phi and full-width.

    Qxb7 grabs a pawn (eval ~ +1 right after) but ...Bxb7 recaptures the queen. The recapture is a
    CAPTURE, exempt from the phi budget, so phi-widen cannot prune the refutation: its backed-up
    White-absolute value is deeply negative, identical in sign to full-width.
    """
    depth = 5
    board = chess.Board(TRAP_QXB7)
    assert board.is_capture(TRAP_MOVE), "Qxb7 must be a capture in the fixture"
    v_full = _move_value_white(AlphaBetaEngine(phi_widen=False), board, TRAP_MOVE, depth)
    v_phi = _move_value_white(AlphaBetaEngine(phi_widen=True), board, TRAP_MOVE, depth)
    assert v_full < -500, f"fixture broken: full-width does not see Qxb7 losing (v={v_full})"
    assert v_phi < -500, (
        f"SOUNDNESS VIOLATION: phi-widen missed the ...Bxb7 refutation of Qxb7 "
        f"(phi v={v_phi} vs full v={v_full}) -- a capture was pruned")
    print(f"  [ok] trap: Qxb7 backs up to a loss under both (full={v_full:.0f}, phi={v_phi:.0f})")


def test_tree_reuse_tt_grows():
    """(b) A persistent engine reused across moves accumulates a non-empty, growing TT.

    Uses a fixed search depth (deterministic) rather than a wall-clock budget, so the assertion does
    not depend on machine speed: the ONE persistent engine's Zobrist-keyed table carries every
    scored interior node forward and gains new keys as fresh positions are searched (:Tree-reuse:).
    """
    depth = 4
    eng = AlphaBetaEngine(phi_widen=True)   # ONE persistent engine across both moves
    board = chess.Board(MIDGAME)            # not an opening-book position -> real search fills the TT

    mv1, _ = _fixed_depth_search(eng, board, depth)
    size1 = len(eng.tt)
    assert size1 > 100, f"TT barely populated ({size1}) -- no computation carried forward"
    assert mv1 in board.legal_moves

    board.push(mv1)
    reply = next(iter(board.legal_moves))
    board.push(reply)                       # opponent reply -> new position, same persistent engine
    mv2, _ = _fixed_depth_search(eng, board, depth)
    size2 = len(eng.tt)
    assert mv2 in board.legal_moves
    assert size2 > size1, (
        f"TT did not grow across moves ({size1} -> {size2}); :Tree-reuse: not accumulating")
    print(f"  [ok] tree-reuse: persistent TT grew {size1} -> {size2} across two searches")


def test_phi_widening_prunes():
    """(c) At equal fixed depth, phi_widen=True searches strictly fewer nodes than full-width."""
    depth = 5
    board = chess.Board(MIDGAME)
    full = AlphaBetaEngine(phi_widen=False, max_depth=depth)
    phi = AlphaBetaEngine(phi_widen=True, max_depth=depth)
    _, nodes_full = _fixed_depth_search(full, board, depth)
    _, nodes_phi = _fixed_depth_search(phi, board, depth)
    assert nodes_phi < nodes_full, (
        f"phi-widen did not prune: {nodes_phi} nodes vs full-width {nodes_full}")
    reduction = 100.0 * (nodes_full - nodes_phi) / nodes_full
    print(f"  [ok] pruning fires: full={nodes_full} nodes, phi={nodes_phi} nodes "
          f"({reduction:.1f}% fewer) at depth {depth}")


def test_floor_constant():
    """The :Full-width-floor: is 3 (guaranteed full-width base below the tactical horizon)."""
    assert PHI_FULL_WIDTH_FLOOR == 3
    print("  [ok] PHI_FULL_WIDTH_FLOOR == 3")


if __name__ == "__main__":
    test_floor_constant()
    test_phi_width_helper()
    test_soundness_same_best_move()
    test_trap_refutation_preserved()
    test_tree_reuse_tt_grows()
    test_phi_widening_prunes()
    print("\nALL PHI-WIDEN TESTS PASSED")
