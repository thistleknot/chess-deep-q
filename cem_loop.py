"""Merge 1 — the self-improvement loop (spec/self-improvement-loop.spec.md). CEM-style policy iteration.

The loop you described: generate games, KEEP ONLY the ones that checkmate within the turn cap into a buffer,
fit a value on those (clean ±1 labels — no draw mush), play greedily by that value, regenerate, repeat.
Watch the checkmate-rate rise across iterations — that rising trend is the falsifiable proof it learns.

Imports Merge 0's ChessEnv UNCHANGED. Value is over the raw board state only (piece-square one-hot = a
learnable piece-square table); no hand-crafted features. Family: Cross-Entropy Method / generalized policy
iteration / self-imitation.

Usage: python cem_loop.py [iterations] [games_per_iter] [turn_cap]
"""
import sys
import random

import numpy as np
import chess

from chess_rl import ChessEnv

NIN = 769   # 12 piece types * 64 squares + 1 side-to-move


def encode(board):
    """Raw board state: piece-square one-hot (12x64) + side-to-move. The least-engineered encoding.
    Bitboard-vectorized (identical output to the original piece_map loop): encode is the hottest path
    under Merge-4 search (~270 calls/move), and unpackbits is ~10-20x the python loop."""
    occ_w, occ_b = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    buf = b"".join((bb & occ).to_bytes(8, "little")
                   for occ in (occ_w, occ_b)
                   for bb in (board.pawns, board.knights, board.bishops,
                              board.rooks, board.queens, board.kings))
    x = np.empty(NIN, dtype=np.float32)
    x[:768] = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    x[768] = 1.0 if board.turn == chess.WHITE else 0.0
    return x


# --- Merge 6 (spec/eval-features.spec.md :Phase-B:) — KnightCap-donor feature encoding ---------
NFEAT = 809   # 769 raw planes + 40 donor features (tridge/KnightCap eval.h, 577-coeff vector)

_ADJ_FILES = [(chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
              for f in range(8)]
_BB_ALL = chess.BB_ALL


def _front_span(color, sq):
    """Squares a pawn on `sq` must clear to be PASSED: same+adjacent files, ranks strictly ahead."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    files = chess.BB_FILES[f] | _ADJ_FILES[f]
    ahead = ((_BB_ALL << (8 * (r + 1))) & _BB_ALL) if color == chess.WHITE else ((1 << (8 * r)) - 1)
    return files & ahead


_PASSED = {chess.WHITE: [_front_span(chess.WHITE, s) for s in range(64)],
           chess.BLACK: [_front_span(chess.BLACK, s) for s in range(64)]}
_MINOR_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)   # mobility-tracked types
_TYPE_VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
             chess.ROOK: 5.0, chess.QUEEN: 9.0}


def encode_features(board):
    """KnightCap-donor features over the raw planes (spec/eval-features.spec.md :Phase-B:).
    x[0:769] = encode(board); the +40: bishop pair, mobility and SAFE mobility per N/B/R/Q,
    hung pieces (count + value-sum), king-ring attack, castling rights, pawn structure
    (doubled/isolated/passed), rook file states, connected rooks. Within each group: White
    block then Black block. All features normalized to O(1). Pure function of the position;
    budget <=5x encode() (measured at adoption)."""
    x = np.zeros(NFEAT, dtype=np.float32)
    x[:NIN] = encode(board)
    occ = (board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK])
    pawns, kings, occ_all = board.pawns, board.kings, board.occupied

    # attack pass straight off python-chess's precomputed tables (Board.attacks_mask does these
    # same lookups behind per-call dispatch — measured 3x slower): per-side unions + (type,sq,mask)
    atk = [0, 0]
    pieces = ([], [])
    _DIAG_A, _DIAG_M = chess.BB_DIAG_ATTACKS, chess.BB_DIAG_MASKS
    _RANK_A, _RANK_M = chess.BB_RANK_ATTACKS, chess.BB_RANK_MASKS
    _FILE_A, _FILE_M = chess.BB_FILE_ATTACKS, chess.BB_FILE_MASKS
    for si, color in ((0, chess.WHITE), (1, chess.BLACK)):
        o, add = occ[si], pieces[si].append
        a = 0
        for sq in chess.scan_forward(pawns & o):
            am = chess.BB_PAWN_ATTACKS[color][sq]
            a |= am
            add((sq, chess.PAWN, am))
        for sq in chess.scan_forward(board.knights & o):
            am = chess.BB_KNIGHT_ATTACKS[sq]
            a |= am
            add((sq, chess.KNIGHT, am))
        for sq in chess.scan_forward(board.bishops & o):
            am = _DIAG_A[sq][_DIAG_M[sq] & occ_all]
            a |= am
            add((sq, chess.BISHOP, am))
        for sq in chess.scan_forward(board.rooks & o):
            am = _RANK_A[sq][_RANK_M[sq] & occ_all] | _FILE_A[sq][_FILE_M[sq] & occ_all]
            a |= am
            add((sq, chess.ROOK, am))
        for sq in chess.scan_forward(board.queens & o):
            am = (_DIAG_A[sq][_DIAG_M[sq] & occ_all]
                  | _RANK_A[sq][_RANK_M[sq] & occ_all] | _FILE_A[sq][_FILE_M[sq] & occ_all])
            a |= am
            add((sq, chess.QUEEN, am))
        k = kings & o
        if k:
            a |= chess.BB_KING_ATTACKS[k.bit_length() - 1]
        atk[si] = a

    cr = board.castling_rights                 # raw rights bitboard (rook home squares)
    for si in (0, 1):
        color, opp = (chess.WHITE, chess.BLACK)[si], 1 - si
        own, safe = occ[si], ~atk[opp]
        not_own = ~own
        own_p, opp_p = pawns & own, pawns & occ[opp]
        own_r = board.rooks & own
        # [769:771] bishop pair
        x[769 + si] = 1.0 if (board.bishops & own).bit_count() >= 2 else 0.0
        # [787:791] hung: own non-king pieces attacked and NOT defended (count W,B then value W,B)
        hung = own & atk[opp] & ~atk[si] & ~kings
        x[787 + si] = hung.bit_count() / 5.0
        # ONE pass over the side's pieces: [771:779] mobility + [779:787] SAFE mobility (N/B/R/Q,
        # W block then B block), hung value-sum, and the rook file/connectivity features
        mob = [0, 0, 0, 0]
        smob = [0, 0, 0, 0]
        hval = 0.0
        ropen = rhalf = 0
        connected = 0.0
        for _sq, t, am in pieces[si]:
            if t != chess.PAWN:
                free = am & not_own
                mob[t - 2] += free.bit_count()
                smob[t - 2] += (free & safe).bit_count()
                if t == chess.ROOK:
                    fmask = chess.BB_FILES[chess.square_file(_sq)]
                    if not (pawns & fmask):
                        ropen += 1
                    elif not (own_p & fmask):
                        rhalf += 1
                    if am & own_r & ~chess.BB_SQUARES[_sq]:
                        connected = 1.0
            if hung and (chess.BB_SQUARES[_sq] & hung):
                hval += _TYPE_VAL[t]
        for ti in range(4):
            x[771 + 4 * si + ti] = mob[ti] / 14.0
            x[779 + 4 * si + ti] = smob[ti] / 14.0
        x[789 + si] = hval / 9.0
        x[803 + si] = ropen / 2.0
        x[805 + si] = rhalf / 2.0
        x[807 + si] = connected
        # [791:793] king-ring attack: enemy-attacked squares around own king
        ksq = (kings & own).bit_length() - 1
        if ksq >= 0:
            x[791 + si] = (chess.BB_KING_ATTACKS[ksq] & atk[opp]).bit_count() / 8.0
        # [793:797] castling rights K,Q per side (WK, WQ, BK, BQ) — raw FEN rights, no revalidation
        back = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
        x[793 + 2 * si] = 1.0 if (cr & back & chess.BB_FILE_H) else 0.0
        x[794 + 2 * si] = 1.0 if (cr & back & chess.BB_FILE_A) else 0.0
        # [797:803] pawn structure: doubled, isolated, passed (W,B per subgroup)
        doubled = isolated = 0
        for f in range(8):
            cnt = (own_p & chess.BB_FILES[f]).bit_count()
            if cnt > 1:
                doubled += cnt - 1
            if cnt and not (own_p & _ADJ_FILES[f]):
                isolated += cnt
        passed = 0
        pm = _PASSED[color]
        for s in chess.scan_forward(own_p):
            if not (pm[s] & opp_p):
                passed += 1
        x[797 + si] = doubled / 8.0
        x[799 + si] = isolated / 8.0
        x[801 + si] = passed / 8.0
    return x


class ValuePolicy:
    """V(board) = tanh(w . encode(board)); epsilon-greedy over afterstate value (white max, black min)."""

    def __init__(self):
        self.w = np.zeros(NIN, dtype=np.float32)

    def value_x(self, x):
        return float(np.tanh(self.w @ x))

    def choose(self, board, epsilon, rng):
        moves = list(board.legal_moves)
        if rng.random() < epsilon:
            return moves[rng.randrange(len(moves))]
        want_max = (board.turn == chess.WHITE)
        best, best_v = None, None
        for mv in moves:
            board.push(mv)
            v = self.value_x(encode(board))
            board.pop()
            if best is None or (v > best_v if want_max else v < best_v):
                best, best_v = mv, v
        return best

    def fit(self, X, y, epochs=8, lr=0.2, batch=256):
        """Regress tanh(w.x) toward the ±1 buffer labels. Minibatch SGD."""
        n = len(y)
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                xb, yb = X[bi], y[bi]
                pred = np.tanh(xb @ self.w)
                grad = (pred - yb) * (1.0 - pred ** 2)     # d/dpred MSE * d tanh
                self.w -= lr * (xb.T @ grad) / len(bi)
        pred = np.tanh(X @ self.w)
        return float(np.mean((pred - y) ** 2))


def play_game(env, pol, epsilon, rng, turn_cap):
    """One game under epsilon-greedy policy. Return (positions_encoded, z, decisive, plies)."""
    board = env.reset()
    xs = [encode(board)]
    done = False
    reward = 0.0
    ply_cap = 2 * turn_cap
    env.ply_cap = ply_cap
    while not done:
        mv = pol.choose(board, epsilon, rng)
        board, reward, done = env.step(mv)
        xs.append(encode(board))
    decisive = board.is_checkmate() and board.fullmove_number <= turn_cap
    return xs, reward, decisive, env.plies


def winrate_vs_random(pol, rng, games, turn_cap):
    """Cheap strength proxy: greedy policy (eps=0) vs uniform-random, agent alternating colors."""
    env = ChessEnv(ply_cap=2 * turn_cap)
    wins = draws = 0
    for g in range(games):
        agent_white = (g % 2 == 0)
        board = env.reset()
        done = False
        reward = 0.0
        while not done:
            if (board.turn == chess.WHITE) == agent_white:
                mv = pol.choose(board, 0.0, rng)
            else:
                ms = list(board.legal_moves)
                mv = ms[rng.randrange(len(ms))]
            board, reward, done = env.step(mv)
        z = reward if agent_white else -reward
        if z > 0:
            wins += 1
        elif z == 0:
            draws += 1
    return wins / games, draws / games


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    G = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    turn_cap = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    rng = random.Random(0)
    np.random.seed(0)

    env = ChessEnv()
    pol = ValuePolicy()
    buf_X, buf_y = [], []

    print(f"Merge 1 CEM loop: {iters} iterations, {G} games/iter, turn_cap={turn_cap}\n", flush=True)
    for it in range(iters):
        epsilon = max(0.2, 0.6 ** it)     # iter0 ~1.0 pure random, decays toward a 0.2 floor
        decisive = 0
        new_pos = 0
        for _ in range(G):
            xs, z, dec, plies = play_game(env, pol, epsilon, rng, turn_cap)
            if dec:
                decisive += 1
                for x in xs:
                    buf_X.append(x); buf_y.append(z)
                new_pos += len(xs)
        rate = decisive / G

        X = np.array(buf_X, dtype=np.float32)
        y = np.array(buf_y, dtype=np.float32)
        loss = pol.fit(X, y) if len(y) else float("nan")
        wr, dr = winrate_vs_random(pol, rng, 30, turn_cap)

        print(f"iter {it} | eps {epsilon:.2f} | checkmate-rate {rate:6.1%} "
              f"({decisive}/{G}) | buffer {len(buf_y):6d} pos (+{new_pos}) | "
              f"loss {loss:.4f} | vs-random W{wr:.0%} D{dr:.0%}", flush=True)

    print("\nvalue sanity (learned piece-square value):")
    for name, fen in [("start", chess.STARTING_FEN),
                      ("white up a queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                      ("black up a rook", "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w KQkq - 0 1")]:
        print(f"  {name:20s}: V = {pol.value_x(encode(chess.Board(fen))):+.3f}")
    print("\nread: does checkmate-rate rise across iters? that is the self-improvement signal.")


if __name__ == "__main__":
    main()
