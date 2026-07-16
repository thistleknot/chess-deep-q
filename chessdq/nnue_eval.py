"""Python evaluator for bullet-trained (768 -> H)x2 SCReLU nets (goal-1600 arm 1).

Loads a bullet checkpoint's raw.bin (f32, SavedFormat order: l0w [768 cols x H], l0b [H],
l1w [2H], l1b [1]) and evaluates chess.Board positions. Output is WHITE-absolute in
"sigmoid-input" units (cp = 400*out per the stock recipe's eval_scale); search only
needs monotonicity, so `value()` returns `out` sign-corrected to White's perspective.

Convention gate (run as script): the net is dual-perspective STM-relative and the
feature map/flip conventions come from bullet's Chess768 + bulletformat — instead of
trusting a re-derivation, `gate()` correlates value() against the training corpus's own
white-relative cp labels, SPLIT BY SIDE-TO-MOVE. Correct conventions -> strong positive
r in both halves; a perspective/sign error -> negative r on the black-to-move half.

Failure modes: wrong file size -> assert; gate r < 0.5 in either half -> SystemExit.
"""
import sys

import numpy as np
import chess

HIDDEN = 128


class NNUEEval:
    def __init__(self, raw_bin_path):
        raw = np.fromfile(raw_bin_path, dtype=np.float32)
        n = 768 * HIDDEN + HIDDEN + 2 * HIDDEN + 1
        assert raw.size == n, f"raw.bin has {raw.size} f32, expected {n}"
        o = 0
        self.l0w = raw[o:o + 768 * HIDDEN].reshape(768, HIDDEN); o += 768 * HIDDEN
        self.l0b = raw[o:o + HIDDEN]; o += HIDDEN
        self.l1w = raw[o:o + 2 * HIDDEN]; o += 2 * HIDDEN
        self.l1b = float(raw[o])

    def features(self, board):
        """(stm_idx list, ntm_idx list) per bullet Chess768 + bulletformat stm-relative
        storage: when Black is to move the board is color-swapped and vertically
        flipped before the standard map."""
        stm_white = board.turn == chess.WHITE
        f_stm, f_ntm = [], []
        for sq, piece in board.piece_map().items():
            pt = piece.piece_type - 1                      # 0..5 pawn..king
            if stm_white:
                col, s = (piece.color == chess.BLACK), sq
            else:
                col, s = (piece.color == chess.WHITE), sq ^ 56
            f_stm.append((384 if col else 0) + 64 * pt + s)
            f_ntm.append((0 if col else 384) + 64 * pt + (s ^ 56))
        return f_stm, f_ntm

    def value(self, board):
        """White-absolute scalar (sigmoid-input units; cp = 400*this)."""
        f_stm, f_ntm = self.features(board)
        acc_s = self.l0b + self.l0w[f_stm].sum(axis=0)
        acc_n = self.l0b + self.l0w[f_ntm].sum(axis=0)
        h = np.concatenate([np.clip(acc_s, 0, 1) ** 2, np.clip(acc_n, 0, 1) ** 2])
        out = float(h @ self.l1w + self.l1b)
        return out if board.turn == chess.WHITE else -out


def gate(net, corpus_path, sample=3000):
    import itertools
    rows_w, rows_b = [], []
    with open(corpus_path, encoding="ascii") as fh:
        for ln in itertools.islice(fh, 0, sample * 40, 40):   # stride for variety
            fen, cp, _res = [s.strip() for s in ln.split("|")]
            b = chess.Board(fen)
            (rows_w if b.turn == chess.WHITE else rows_b).append((net.value(b), int(cp)))
    out = {}
    for name, rows in (("white-to-move", rows_w), ("black-to-move", rows_b)):
        v, c = np.array([r[0] for r in rows]), np.array([r[1] for r in rows], float)
        r = float(np.corrcoef(v, c)[0, 1])
        out[name] = (r, len(rows))
        print(f"gate {name}: r={r:+.3f} over {len(rows)} positions")
    if any(r < 0.5 for r, _n in out.values()):
        raise SystemExit("convention gate FAILED — perspective/feature map wrong")
    print("convention gate PASS")


if __name__ == "__main__":
    net = NNUEEval(sys.argv[1] if len(sys.argv) > 1
                   else r"C:\Users\user\Documents\dev\bullet\checkpoints\vol1600-40\raw.bin")
    gate(net, "data/corpus_vol_f.txt")
    for lbl, fen in (("startpos", chess.Board().fen()),
                     ("white Q up", "3q3k/8/8/8/8/8/8/QQ5K w - - 0 1"),
                     ("black Q up", "qq5k/8/8/8/8/8/8/3Q3K b - - 0 1")):
        print(f"{lbl:12s} value {net.value(chess.Board(fen)):+.3f}")
