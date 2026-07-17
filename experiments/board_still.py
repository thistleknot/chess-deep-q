"""Render a mid-game still of the terminal chess interface to a PNG.

Plays a seeded random legal-move game to "the thick of it", selects the mover's
most mobile piece (so the selected/possible/threat/guard highlights are all
live), captures the REAL TerminalChessBoard.display_board() ANSI output, and
rasterizes it with Windows Terminal (Campbell) colors. Used to produce the
README gameplay screenshot without hand-cropping a terminal window.

Usage (from repo root):
    python experiments/board_still.py [seed] [out.png]

Preconditions: Pillow installed; Cascadia Mono + Segoe UI Symbol fonts present
(standard on Windows 10+). Fails fast if fonts or output dir are missing.
"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import chess
from PIL import Image, ImageDraw, ImageFont

from chessdq.terminal_board import TerminalChessBoard
from chessdq.board_utils import get_legal_moves_from_square, get_secondary_moves

# Windows Terminal "Campbell" palette.
_ANSI16 = {
    0: (12, 12, 12), 1: (197, 15, 31), 2: (19, 161, 14), 3: (193, 156, 0),
    4: (0, 55, 218), 5: (136, 23, 152), 6: (58, 150, 221), 7: (204, 204, 204),
    8: (118, 118, 118), 9: (231, 72, 86), 10: (22, 198, 12), 11: (249, 241, 165),
    12: (59, 120, 255), 13: (180, 0, 158), 14: (97, 214, 214), 15: (242, 242, 242),
}
FG_DEFAULT = _ANSI16[7]
BG_DEFAULT = _ANSI16[0]
FONT_MONO = r"C:\Windows\Fonts\CascadiaMono.ttf"
FONT_CHESS = r"C:\Windows\Fonts\seguisym.ttf"   # Cascadia Mono lacks U+2654-265F
CHESS_GLYPHS = {chr(c) for c in range(0x2654, 0x2660)}


def _xterm256(n):
    """256-color index -> RGB (16 base, 6x6x6 cube, gray ramp)."""
    if n < 16:
        return _ANSI16[n]
    if n < 232:
        n -= 16
        steps = (0, 95, 135, 175, 215, 255)
        return (steps[n // 36], steps[(n // 6) % 6], steps[n % 6])
    g = 8 + (n - 232) * 10
    return (g, g, g)


def parse_ansi(text):
    """ANSI SGR text -> list of lines, each a list of (char, fg, bg) cells."""
    fg, bg = FG_DEFAULT, BG_DEFAULT
    lines, cells = [], []
    pos = 0
    for m in re.finditer(r"\x1b\[([0-9;]*)m", text):
        for ch in text[pos:m.start()]:
            if ch == "\n":
                lines.append(cells)
                cells = []
            elif ch != "\r":
                cells.append((ch, fg, bg))
        pos = m.end()
        codes = [int(c) for c in m.group(1).split(";") if c] or [0]
        i = 0
        while i < len(codes):
            c = codes[i]
            if c == 0:
                fg, bg = FG_DEFAULT, BG_DEFAULT
            elif 30 <= c <= 37:
                fg = _ANSI16[c - 30]
            elif 90 <= c <= 97:
                fg = _ANSI16[c - 90 + 8]
            elif 40 <= c <= 47:
                bg = _ANSI16[c - 40]
            elif 100 <= c <= 107:
                bg = _ANSI16[c - 100 + 8]
            elif c in (38, 48) and i + 2 < len(codes) and codes[i + 1] == 5:
                color = _xterm256(codes[i + 2])
                if c == 38:
                    fg = color
                else:
                    bg = color
                i += 2
            i += 1
    for ch in text[pos:]:
        if ch == "\n":
            lines.append(cells)
            cells = []
        elif ch != "\r":
            cells.append((ch, fg, bg))
    if cells:
        lines.append(cells)
    return lines


def render_png(lines, out_path, font_px=22, margin=24):
    """Rasterize parsed cells as a terminal-styled image. Guarantees the PNG on disk."""
    mono = ImageFont.truetype(FONT_MONO, font_px)
    glyph = ImageFont.truetype(FONT_CHESS, font_px)
    ascent, descent = mono.getmetrics()
    cell_w = int(mono.getlength("M"))
    cell_h = ascent + descent
    n_cols = max((len(l) for l in lines), default=1)
    img = Image.new("RGB", (n_cols * cell_w + 2 * margin,
                            len(lines) * cell_h + 2 * margin), BG_DEFAULT)
    draw = ImageDraw.Draw(img)
    for row, cells in enumerate(lines):
        y = margin + row * cell_h
        for col, (ch, fg, bg) in enumerate(cells):
            x = margin + col * cell_w
            if bg != BG_DEFAULT:
                draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], fill=bg)
            if ch != " ":
                f = glyph if ch in CHESS_GLYPHS else mono
                w = draw.textlength(ch, font=f)
                draw.text((x + (cell_w - w) / 2, y), ch, font=f, fill=fg)
    img.save(out_path)
    return img.size


def midgame_position(seed, plies=26):
    """Seeded random game biased toward captures; returns (board, last_move) mid-game."""
    rng = random.Random(seed)
    board = chess.Board()
    move = None
    for _ in range(plies):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        captures = [m for m in legal if board.is_capture(m)]
        pool = captures if captures and rng.random() < 0.5 else legal
        move = rng.choice(pool)
        board.push(move)
    return board, move


def pick_selection(board):
    """The mover's most mobile piece (never the king) -> (square, moves) or (None, set())."""
    best, best_moves = None, set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == board.turn and piece.piece_type != chess.KING:
            moves = get_legal_moves_from_square(board, square)
            if len(moves) > len(best_moves):
                best, best_moves = square, moves
    return best, best_moves


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("images", "chess-v1.5-midgame.png")

    board, last_move = midgame_position(seed)
    selected, possible = pick_selection(board)

    class _StubAI:  # display_board only touches difficulty extras, all absent here
        pass

    ui = TerminalChessBoard(board, _StubAI())
    ui.last_move = last_move
    ui.selected_square = selected
    ui.possible_moves = possible
    ui.secondary_moves = get_secondary_moves(board, selected) if selected is not None else set()

    captured = io.StringIO()
    stdout = sys.stdout
    sys.stdout = captured
    try:
        ui.display_board()
        print("\nEnter move (e.g., e2e4) or command: ", end="")
    finally:
        sys.stdout = stdout

    size = render_png(parse_ansi(captured.getvalue()), out)
    sel = chess.square_name(selected) if selected is not None else "-"
    print(f"seed={seed} fen={board.fen()}")
    print(f"selected={sel} moves={len(possible)} -> {out} {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
