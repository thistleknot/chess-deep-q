"""Merge 0 — the chess MDP scaffold (spec/environment.spec.md). Board + random legal moves, nothing more.

This is the reviewable base every RL rung builds on: an Environment with reset/step and a white-absolute
terminal reward, plus a uniform-random rollout to establish the baseline that learning must beat. NO value,
NO policy, NO learning, NO evaluator/search — reward comes only from the game result.

Interface mirrors reference/all-rl-algorithms/01_simple_rl.ipynb GridEnvironmentSimple.

  State  s : chess.Board                          (env is agnostic to how a rung encodes it)
  Action a : a legal move
  Reward r : 0 every step; terminal white-absolute z: +1 white mate, -1 black mate, 0 draw / ply-cap
  Policy   : uniform random over legal moves (ε = 1 — the exploration floor every rung starts from)

Usage: python chess_rl.py [episodes] [ply_cap]
"""
import sys
import random

import chess

PLY_CAP = 200          # random play wanders; cap -> draw so episodes always return


class ChessEnv:
    """Owner of state transitions and reward. reset/step only — no policy, no value, no learning."""

    def __init__(self, ply_cap=PLY_CAP):
        self.ply_cap = ply_cap
        self.board = None
        self.plies = 0

    def reset(self, board=None):
        """Standard start, or an EXPLORING START (spec :Curriculum-starts:) when a legal board is
        given. Reward semantics unchanged: 0 until terminal, then white-absolute z."""
        self.board = board if board is not None else chess.Board()
        self.plies = 0
        return self.board

    def step(self, move):
        """Apply move. Return (board, reward, done). Reward 0 until terminal, then white-absolute z."""
        self.board.push(move)
        self.plies += 1
        done = self.board.is_game_over() or self.plies >= self.ply_cap
        reward = self._terminal_z() if done else 0.0
        return self.board, reward, done

    def _terminal_z(self):
        if self.board.is_checkmate():
            # side to move is mated -> the OTHER side delivered mate
            return -1.0 if self.board.turn == chess.WHITE else 1.0
        return 0.0   # stalemate / insufficient / repetition / 75-move / ply cap


def random_policy(board, rng):
    """ε = 1: uniform choice over legal moves. Merge 0's entire behavior policy."""
    moves = list(board.legal_moves)
    return moves[rng.randrange(len(moves))]


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    ply_cap = int(sys.argv[2]) if len(sys.argv) > 2 else PLY_CAP
    rng = random.Random(0)
    env = ChessEnv(ply_cap=ply_cap)

    w = l = d = 0
    total_len = 0
    print(f"Merge 0 baseline: {episodes} uniform-random games, ply_cap={ply_cap}\n", flush=True)
    for ep in range(1, episodes + 1):
        board = env.reset()
        done = False
        reward = 0.0
        while not done:
            board, reward, done = env.step(random_policy(board, rng))
        total_len += env.plies
        if reward > 0:
            w += 1
        elif reward < 0:
            l += 1
        else:
            d += 1
        if ep % max(1, episodes // 10) == 0:
            n = ep
            print(f"ep {ep:5d} | W {w} L {l} D {d} | white-win {100*w/n:.0f}% "
                  f"decisive {100*(w+l)/n:.0f}% | avg_len {total_len/n:.0f}", flush=True)

    n = episodes
    print(f"\nrandom baseline over {n} games: "
          f"W {w} ({100*w/n:.1f}%)  L {l} ({100*l/n:.1f}%)  D {d} ({100*d/n:.1f}%)  "
          f"| decisive {100*(w+l)/n:.1f}%  avg_len {total_len/n:.0f}")
    print("this is the floor every learning rung must beat (spec/environment.spec.md).")


if __name__ == "__main__":
    main()
