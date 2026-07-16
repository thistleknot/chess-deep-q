"""Batched shallow minimax using the learned net as the leaf evaluator.

The net eval is good (it wins a free queen) but slow per single call (~2-3ms); batched it is 20-120k
positions/s. Alpha-beta evaluates nodes one at a time, so it can't exploit batching. Instead we run
a fixed-depth full-width / beam minimax and batch-evaluate every node's children in ONE forward
pass. This gives the net real lookahead (it stops hanging pieces to 2-3 move tactics) at ~0.3-0.8s
per move -- far stronger than the ~200-iteration sampled MCTS, and the natural policy-model move
selector to drive both play and self-play.

Convention: the net outputs a White-absolute value in [-1,1]. Minimax maximizes it for White and
minimizes it for Black -- no negamax sign juggling. Mates use +/-MATE beyond the tanh range so they
dominate; draws are 0.
"""

import chess
import torch

from chessdq.board_utils import board_to_tensor

MATE = 30.0

# Piece values for MVV-LVA capture ordering in quiescence.
_QCP = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def _terminal_white_value(board):
    """White-absolute value if `board` is terminal, else None."""
    if board.is_checkmate():
        return -MATE if board.turn == chess.WHITE else MATE   # side to move is mated
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    return None


def _resolve_quiet(board, max_plies=8):
    """Cheaply resolve a leaf to a quiet position for batched net evaluation: greedily play
    winning/equal captures (MVV-LVA proxy: victim value >= attacker value) until none remain, so
    the net only ever scores quiet positions (spec :Quiescence-search:). Pure-Python material logic,
    NO net calls — this keeps :Batched-leaf-evaluation: intact (the batching/quiescence tension).
    Approximation: MVV-LVA, not full SEE, so a defended-equal capture may be over-played; adequate
    to kill the dominant horizon errors (hanging pieces, missed recaptures)."""
    b = board
    copied = False
    for _ in range(max_plies):
        if b.is_game_over():
            break
        caps = [m for m in b.legal_moves if b.is_capture(m)]
        if not caps:
            break

        def gain(m):
            v = _QCP.get(b.piece_type_at(m.to_square) or chess.PAWN, 0)
            a = _QCP.get(b.piece_type_at(m.from_square) or chess.PAWN, 0)
            return v - a
        best = max(caps, key=gain)
        if gain(best) < 0:      # no winning/equal capture -> quiet enough
            break
        if not copied:
            b = board.copy(); copied = True
        b.push(best)
    return b


def policy_priors(board, net, device):
    """Softmax policy priors over the legal moves at `board`, from the net's policy head.

    Returns {move: prior in [0,1]} from ONE forward pass (encode18). This is the move-ordering
    surface the policy head was trained for but never consumed (search used the value head only).
    Expects a dual-head net returning (value, policy); indexes logits by resnet_model.move_index
    (from*64+to, so underpromotions collapse to the queening move -- fine for ordering).
    """
    from chessdq.resnet_model import encode18, move_index
    moves = list(board.legal_moves)
    if not moves:
        return {}
    with torch.no_grad():
        out = net(encode18(board).unsqueeze(0).to(device))
        logits = (out[1] if isinstance(out, tuple) else out).squeeze(0)
        sel = logits[[move_index(m) for m in moves]]
        probs = torch.softmax(sel, dim=0).tolist()
    return dict(zip(moves, probs))


def _white_values(boards, net, device, encode_fn=None, qdepth=0, board_eval=None):
    """White-absolute values for a list of boards, with terminal overrides.

    encode_fn converts a board to an input tensor (defaults to the legacy 12-plane encoding).
    Nets returning (value, policy) tuples are handled by taking the value head. qdepth>0 resolves
    each leaf through quiescence (per-board, sequential); qdepth=0 is the batched net eval.
    board_eval, if given, overrides the net entirely: a board -> White-absolute [-1,1] function
    (used for the eval↔search swap experiment, e.g. the hand heuristic in this same search).
    """
    encode = encode_fn or board_to_tensor
    # Quiescence: resolve each leaf to a quiet position (cheap, no net calls) THEN batch-eval.
    eval_boards = [_resolve_quiet(b, max_plies=qdepth) for b in boards] if qdepth > 0 else boards
    term = [_terminal_white_value(b) for b in eval_boards]
    live_idx = [i for i, t in enumerate(term) if t is None]
    out = [t if t is not None else 0.0 for t in term]
    if board_eval is not None:
        for i in live_idx:
            out[i] = max(-1.0, min(1.0, board_eval(eval_boards[i])))
        return out
    if live_idx:
        batch = torch.stack([encode(eval_boards[i]) for i in live_idx]).to(device)
        with torch.no_grad():
            pred = net(batch)
            if isinstance(pred, tuple):
                pred = pred[0]
            vals = pred.squeeze(1).tolist()
        for j, i in enumerate(live_idx):
            out[i] = max(-1.0, min(1.0, vals[j]))
    return out


def _value(board, net, device, depth, beam, encode_fn=None, q=0, board_eval=None):
    """White-absolute minimax value of `board` searched to `depth` with per-node top-`beam` pruning.
    q>0 applies quiescence at the frontier leaves (not at ordering nodes)."""
    t = _terminal_white_value(board)
    if t is not None:
        return t
    moves = list(board.legal_moves)
    if not moves:
        return 0.0
    children = []
    for m in moves:
        c = board.copy(); c.push(m); children.append(c)
    white_to_move = board.turn == chess.WHITE
    if depth <= 1:
        vals = _white_values(children, net, device, encode_fn, qdepth=q, board_eval=board_eval)
        return max(vals) if white_to_move else min(vals)
    # Beam: order by cheap (non-quiescence) values, keep top-`beam`, recurse deeper.
    vals = _white_values(children, net, device, encode_fn, qdepth=0, board_eval=board_eval)
    order = sorted(range(len(moves)), key=lambda i: vals[i], reverse=white_to_move)[:beam]
    deeper = [_value(children[i], net, device, depth - 1, beam, encode_fn, q, board_eval) for i in order]
    return max(deeper) if white_to_move else min(deeper)


def best_move(board, net, device, depth=2, beam=8, encode_fn=None, q=0, board_eval=None):
    """Return the net-minimax best move for the side to move (White-absolute minimax).
    q>0 enables quiescence at frontier leaves (spec :Quiescence-search:). board_eval overrides the
    net with a board->[-1,1] function (eval↔search swap experiment)."""
    moves = list(board.legal_moves)
    if not moves:
        return None
    children = []
    for m in moves:
        c = board.copy(); c.push(m); children.append(c)
    white_to_move = board.turn == chess.WHITE
    # Ordering uses cheap batched values; at depth 1 the children ARE the frontier -> quiesce them.
    one_ply = _white_values(children, net, device, encode_fn, qdepth=(q if depth <= 1 else 0),
                            board_eval=board_eval)
    if depth <= 1:
        best_i = max(range(len(moves)), key=lambda i: one_ply[i]) if white_to_move \
            else min(range(len(moves)), key=lambda i: one_ply[i])
        return moves[best_i]
    # Root beam: only search the most promising root moves deeper (keep it wider than inner beam).
    root_beam = max(beam, 12)
    order = sorted(range(len(moves)), key=lambda i: one_ply[i], reverse=white_to_move)[:root_beam]
    scored = [(_value(children[i], net, device, depth - 1, beam, encode_fn, q, board_eval), i) for i in order]
    best_i = (max(scored) if white_to_move else min(scored))[1]
    return moves[best_i]
