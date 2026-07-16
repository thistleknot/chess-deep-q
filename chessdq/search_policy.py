"""Merge 4 — depth-2 search policy over a learned evaluator (spec/decision-time-planning.spec.md).

`move_values` returns White-absolute values per legal move via width-limited depth-2 negamax with
BATCHED leaf evaluation; `search_move` samples the root softmax (behavior) or argmax (measurement).
value_fn: callable (n,769) float32 -> (n,) White-absolute values in [-1,1] (any critic).
Failure modes: none critical — a position with no legal replies is scored terminally.
"""
import numpy as np
import chess

from chessdq.cem_loop import encode, NIN


def _probe_terminal(board):
    """CHEAP terminal probe: None if any legal move exists (short-circuits — no full outcome() with
    its per-leaf repetition scans). Mate = mover in check with no moves; stalemate = 0. Leaf-level
    repetition/insufficient draws are deliberately left to the critic (≈0 there anyway)."""
    if any(board.legal_moves):
        return None
    if board.is_check():
        return -1.0 if board.turn == chess.WHITE else 1.0   # side to move is mated
    return 0.0


def _allows_mate_in_1(board):
    """True if the side to move can deliver checkmate immediately."""
    for mv in board.legal_moves:
        board.push(mv)
        mated = board.is_checkmate()
        board.pop()
        if mated:
            return True
    return False


def move_values(board, value_fn, width=8, qext=False, want_leaves=False, encode_fn=encode):
    """(moves, values[, leaves]): White-absolute depth-2 value per legal move. Every root move is
    pushed once (exact mate/stalemate detection); only the top-`width` by 1-ply value get reply
    expansion. qext: quiescence-lite — when the deepest reply is a capture, extend by one
    recapture-only ply with standing-pat (side to move takes the better of static and best
    recapture). want_leaves (spec :PV-leaf:, TDLeaf): also return per-move the 769-dim encoding of
    the PV LEAF — the position whose value survived the min/max propagation into vals[i]
    (unexpanded/terminal moves: the root afterstate itself)."""
    moves = list(board.legal_moves)
    n = len(moves)
    encs = [None] * n
    term = [None] * n
    for i, mv in enumerate(moves):
        board.push(mv)
        encs[i] = encode_fn(board)
        term[i] = _probe_terminal(board)
        board.pop()
    X1 = np.stack(encs) if n else np.empty((0, NIN), dtype=np.float32)
    vals = np.asarray(value_fn(X1), dtype=np.float32).copy()   # 1-ply values (fallback + ranking)
    for i, t in enumerate(term):
        if t is not None:
            vals[i] = t
    leaf = encs if want_leaves else None
    pred = [None] * n if want_leaves else None                 # :Ramp-filter:: predicted reply

    sigma = 1.0 if board.turn == chess.WHITE else -1.0
    order = np.argsort(-sigma * vals)                          # best-for-mover first
    expand = [i for i in order[:width] if term[i] is None]

    # batch ALL replies of all expanded roots through the evaluator (one static batch, and with
    # qext one extra recapture batch); entries carry (value, leaf_enc|None, reply_move) so the
    # PV leaf AND the predicted reply (:Ramp-filter:) ride the same min/max propagation;
    # leaves[lid] = [root_idx, static_val, [(rec_val, enc)], own_enc, reply_move]
    reply_vals = {i: [] for i in expand}
    leaves = []
    leaf_X, leaf_meta = [], []                                 # meta: leaf id
    q_X, q_meta = [], []                                       # recapture leaves: meta: leaf id
    for i in expand:
        board.push(moves[i])
        for r in board.legal_moves:
            cap_sq = r.to_square if (qext and board.is_capture(r)) else None
            board.push(r)
            t = _probe_terminal(board)
            if t is not None:
                reply_vals[i].append((t, encode_fn(board) if want_leaves else None, r))
            else:
                lid = len(leaves)
                enc = encode_fn(board)
                leaves.append([i, 0.0, [], enc, r])
                leaf_X.append(enc)
                leaf_meta.append(lid)
                if cap_sq is not None:                         # recapture-only extension
                    for c in board.legal_moves:
                        if c.to_square == cap_sq and board.is_capture(c):
                            board.push(c)
                            tc = _probe_terminal(board)
                            if tc is not None:
                                leaves[lid][2].append((tc, encode_fn(board) if want_leaves else None))
                            else:
                                q_X.append(encode_fn(board))
                                q_meta.append(lid)
                            board.pop()
            board.pop()
        board.pop()
    if leaf_X:
        lv = np.asarray(value_fn(np.stack(leaf_X)), dtype=np.float32)
        for v, lid in zip(lv, leaf_meta):
            leaves[lid][1] = float(v)
    if q_X:
        qv = np.asarray(value_fn(np.stack(q_X)), dtype=np.float32)
        for v, enc, lid in zip(qv, q_X, q_meta):
            leaves[lid][2].append((float(v), enc if want_leaves else None))
    for i_, static, recs, enc, rmove in leaves:                # standing-pat: mover's best of
        cand = [(static, enc)] + recs                          # static and any recapture
        best = (max(cand, key=lambda p: p[0]) if sigma > 0 else
                min(cand, key=lambda p: p[0])) if recs else cand[0]
        reply_vals[i_].append((best[0], best[1], rmove))       # predicted reply = the depth-2
    for i in expand:                                           # move r, even when a recapture
        if reply_vals[i]:                                      # extended it deeper
            pick = (min(reply_vals[i], key=lambda p: p[0]) if sigma > 0 else
                    max(reply_vals[i], key=lambda p: p[0]))    # opponent picks THEIR best reply
            vals[i] = pick[0]
            if want_leaves:
                leaf[i] = pick[1]
                pred[i] = pick[2]
    if want_leaves:
        # :Backed-bootstrap:: True iff vals[i] is exact (terminal) or depth-backed — unexpanded
        # moves keep optimistic 1-ply values and must never feed a max/bootstrap (S&B 015)
        backed = [term[i] is not None or (i in reply_vals and bool(reply_vals[i]))
                  for i in range(n)]
        return moves, vals, leaf, backed, pred
    return moves, vals


def move_values_d3(board, value_fn, beam=12, w2=8, qext=False, want_leaves=False,
                   encode_fn=encode):
    """(moves, values[, leaves]): White-absolute DEPTH-3 value per legal move. 1-ply values rank all
    moves; the top-`beam` non-terminal moves are re-scored by running the existing depth-2 search
    (`move_values`, width=w2) from the position after the move — the opponent then picks their
    best depth-2-backed reply. Everything below reuses move_values' batched eval path.
    want_leaves (spec :PV-leaf:): a beamed move's leaf is the leaf of the depth-2 reply that
    survived the opponent's min/max; unbeamed/terminal moves keep the root afterstate."""
    moves = list(board.legal_moves)
    n = len(moves)
    encs = [None] * n
    term = [None] * n
    for i, mv in enumerate(moves):
        board.push(mv)
        encs[i] = encode_fn(board)
        term[i] = _probe_terminal(board)
        board.pop()
    X1 = np.stack(encs) if n else np.empty((0, NIN), dtype=np.float32)
    vals = np.asarray(value_fn(X1), dtype=np.float32).copy()
    for i, t in enumerate(term):
        if t is not None:
            vals[i] = t
    leaf = encs if want_leaves else None
    pred = [None] * n if want_leaves else None                 # :Ramp-filter:: predicted reply

    sigma = 1.0 if board.turn == chess.WHITE else -1.0
    order = np.argsort(-sigma * vals)
    backed = [t is not None for t in term] if want_leaves else None
    for i in order[:beam]:
        if term[i] is not None:
            continue
        board.push(moves[i])
        rleaves = None
        if want_leaves:
            rmoves, rvals, rleaves, _rb, _rp = move_values(board, value_fn, width=w2, qext=qext,
                                                           want_leaves=True, encode_fn=encode_fn)
        else:
            _, rvals = move_values(board, value_fn, width=w2, qext=qext,
                                   encode_fn=encode_fn)     # depth-2 under the reply
        if len(rvals):                                         # opponent picks THEIR best reply
            j = int(rvals.argmin() if sigma > 0 else rvals.argmax())
            vals[i] = rvals[j]
            if want_leaves:
                leaf[i] = rleaves[j]
                backed[i] = True                               # :Backed-bootstrap:
                pred[i] = rmoves[j]                            # the reply that won the min/max
        board.pop()
    if want_leaves:
        return moves, vals, leaf, backed, pred
    return moves, vals


def search_move(board, value_fn, tau=0.0, rng=None, width=8, depth=2, beam=12, qext=False,
                guard=False, return_leaf=False, encode_fn=encode, soft_backup_tau=0.0):
    """Root policy: argmax when tau<=0 (measurement), softmax sample otherwise.
    depth=2 -> move_values(width); depth=3 -> move_values_d3(beam, w2=width).
    guard: mate-in-1 guard on the argmax path — walk candidates best-first, take an immediate
    mate if we have one, and skip any move that hangs an opponent mate-in-1 (moves outside the
    expanded width/beam keep optimistic 1-ply values and can otherwise hang mate). Lazy: probes
    only until the first safe candidate; if EVERY move allows mate, falls back to best-valued.
    return_leaf (spec :PV-leaf:, TDLeaf): return (move, leaf_encoding_of_chosen_move,
    greedy_search_value, predicted_reply) — gv is the White-absolute best BACKED root value;
    soft_backup_tau (spec :Backup-temperature:, operator's eligibility-trace analogy): >0
    replaces the max-backup gv with MELLOWMAX over the backed root values (Asadi & Littman
    2017) — tau_b->0 = minimax max, tau_b->inf = MC-style mean; move CHOICE is unchanged;
    predicted_reply (:Ramp-filter:) is the opponent reply the search expects to the CHOSEN
    move (None when the chosen move was unexpanded)."""
    if depth >= 3:
        out = move_values_d3(board, value_fn, beam=beam, w2=width, qext=qext,
                             want_leaves=return_leaf, encode_fn=encode_fn)
    else:
        out = move_values(board, value_fn, width, qext=qext, want_leaves=return_leaf,
                          encode_fn=encode_fn)
    moves, vals = out[0], out[1]
    leaves = out[2] if return_leaf else None
    sigma = 1.0 if board.turn == chess.WHITE else -1.0
    if return_leaf:
        # :Backed-bootstrap: (S&B 015): the bootstrap max runs over depth-backed values ONLY —
        # optimistic unexpanded 1-ply fallbacks inflate a max and poison TDLeaf targets
        bidx = [i for i, bk in enumerate(out[3]) if bk]
        greedy_j = (bidx[int(np.argmax(sigma * vals[bidx]))] if bidx
                    else int(np.argmax(sigma * vals)))
    else:
        greedy_j = int(np.argmax(sigma * vals))
    if tau <= 0 or rng is None:
        j = greedy_j
        if guard:
            order = np.argsort(-sigma * vals)
            j = int(order[0])                               # all-moves-lose fallback
            for k in order:
                board.push(moves[k])
                mate = board.is_checkmate()                 # we mate: take it
                hangs = False if mate else _allows_mate_in_1(board)
                board.pop()
                if mate or not hangs:
                    j = int(k)
                    break
    else:
        logits = sigma * vals / tau
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        j = int(rng.choices(range(len(moves)), weights=p)[0])
    if return_leaf:
        gv = float(vals[greedy_j])
        if soft_backup_tau > 0:
            bv = sigma * (vals[bidx] if bidx else vals)     # mover-perspective BACKED values
            m = float(bv.max())
            gv = sigma * (m + soft_backup_tau
                          * float(np.log(np.mean(np.exp((bv - m) / soft_backup_tau)))))
        return moves[j], leaves[j], gv, out[4][j]
    return moves[j]
