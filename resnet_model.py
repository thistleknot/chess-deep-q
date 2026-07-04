"""Dual-head residual tower per spec/learned-model.spec.md.

18 input planes (12 piece placement + side-to-move + 4 castling + en passant), a small residual
trunk (RES_BLOCKS x RES_FILTERS), a White-absolute tanh value head, and a policy head over a flat
from*64+to move index (promotions collapse to the queening move; underpromotion is rare enough to
ignore at this strength). Sized to respect the batched-throughput budget, not to be Leela.
"""

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

RES_BLOCKS = 5
RES_FILTERS = 64
PLANES = 18
POLICY_SIZE = 64 * 64


def encode18(board):
    """18-plane tensor. Side-to-move plane is mandatory once a policy head exists
    (12 placement planes cannot identify the mover); value stays White-absolute."""
    t = torch.zeros(PLANES, 8, 8)
    for sq, piece in board.piece_map().items():
        ch = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        t[ch, sq >> 3, sq & 7] = 1.0
    if board.turn == chess.WHITE:
        t[12].fill_(1.0)
    if board.has_kingside_castling_rights(chess.WHITE):
        t[13].fill_(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        t[14].fill_(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        t[15].fill_(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        t[16].fill_(1.0)
    if board.ep_square is not None:
        t[17, board.ep_square >> 3, board.ep_square & 7] = 1.0
    return t


def move_index(move):
    """Flat policy index: from*64 + to."""
    return move.from_square * 64 + move.to_square


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class ChessResNet(nn.Module):
    """Trunk + value head (tanh scalar, White-absolute) + policy head (POLICY_SIZE logits)."""

    def __init__(self, blocks=RES_BLOCKS, filters=RES_FILTERS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(PLANES, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        self.vh = nn.Sequential(
            nn.Conv2d(filters, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(8 * 64, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())
        self.ph = nn.Sequential(
            nn.Conv2d(filters, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(16 * 64, POLICY_SIZE))

    def forward(self, x):
        h = self.trunk(self.stem(x))
        return self.vh(h), self.ph(h)
