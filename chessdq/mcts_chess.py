"""
### RLChessA2C, for training a Leela-style policy+value chess agent via
    advantage actor-critic, with exploration supplied by a Fibonacci-annealed
    sampled-lookahead beam (flat fetch -> temperature survival -> negamax
    backups) that anneals off with epsilon.

==================================== SPEC =====================================

--- Requirements ---

The agent MUST learn from self-play only; no game databases.
The behavior policy MUST be a mixture: with probability epsilon, the move
    comes from the Fibonacci beam sampler (the off-policy search arm);
    otherwise the move is sampled from the actor's masked softmax.
When episode index reaches anneal_episodes: epsilon MUST equal epsilon_end,
    leaving pure on-policy self-play (the Leela regime).
Beam leaf evaluation MUST blend eps*Phi + (1-eps)*V_net, so the heuristic
    anneals out of the search on the same schedule and the search stays
    guided by the learned value head as training progresses.
While epsilon > 0: policy gradients are biased (no importance correction) —
    the exact mixture probability of a beam-chosen move marginalizes over
    the stochastic pruning rounds and is intractable. This is the
    Q-learning stance: learn from search-chosen actions unweighted; the
    bias SHALL vanish as epsilon anneals to 0.
The critic target MUST be the n-step negamax return
    G_t = sum_{k<n} (-gamma)^k r_{t+k} + (-gamma)^n V(s_{t+n}),
    bootstrapping without waiting for episode end (values are always from the
    side-to-move perspective; the sign alternation is the two-player flip).
The advantage MUST be A_t = G_t - V(s_t)  (the R - V(S) form).
Reward shaping MUST be potential-based, F = gamma*(-Phi(s')) - Phi(s) with
    Phi = tanh-squashed material+PST score, so the optimal policy is invariant
    to the shaping and the heuristic cannot corrupt the final policy.
If a game exceeds max_plies or is drawn: terminal reward MUST be 0.
The beam's played move MUST be the argmax over root backed-up values
    (:Root-commit:); exploration lives upstream in fetch and survival
    sampling. select() MUST stamp last_value (acted-on estimate, the prior
    in the cross-move :Delta:) and last_margin (:Move-margin:, max - mean
    of the final top_k) on every exit path, including forced-move and
    proven-mate short-circuits.
Checkpoints SHOULD load-if-exists; metrics SHALL persist to sqlite.
Elo estimation MUST anchor to an opponent of known rating: Stockfish under
    UCI_LimitStrength (UCI_Elo floor 1320, calibrated at long TC / CCRL
    40/4). Below that floor no calibrated anchor exists — lower ladder rungs
    (random, 1-ply heuristic) report win rates and Elo *differences* only.
An all-win or all-loss match MUST be reported as a bound, not an estimate
    (the logistic saturates; D = 400*log10(s/(1-s)) diverges at s in {0,1}).
Underpromotions are collapsed to queen promotion (MAY be lifted later; the
    policy head is 64x64 from-to and does not distinguish promotion piece).
Beam search MUST NOT claim draws by repetition (boards are copied without
    move stacks for speed; threefold detection is unavailable inside search).

--- Structural ---

Config: DEVICE — torch device string, varies by host hardware
Constant: PIECE_VALUES — centipawn material weights, standard chess valuation
Constant: PST — piece-square tables, classical positional priors
Constant: POLICY_SIZE — 4096, the 64x64 from-to move index space
Config: phi_depth/phi_width/phi_start — beam schedule knobs, vary by the
    score-vs-compute sweet spot found via the sweep mode

class HeuristicScorer [service]: stateless board evaluation, the off-policy
    knowledge source inside the beam's leaf blend
    phi(board) -> float: tanh-squashed score in (-1,1) from the side-to-move
        perspective; fails closed to 0.0 on terminal boards

class ActorCritic [entity]: the learned policy+value network (Leela-mini)
    state: conv trunk + policy head + value head parameters, mutated by
        optimizer steps only
    forward(x) -> (logits[4096], value in (-1,1)): joint heads over a shared
        trunk; garbage-in on malformed plane tensors
    masked_policy(logits, mask) -> probs: renormalized softmax over legal
        moves only
        (Guarantee: probs sum to 1 over legal indices; illegal indices are 0)

class Line [value]: one surviving lookahead trajectory in the beam —
    (board at leaf, root move, root-perspective value, terminal flag)

class FibonacciBeamSampler [service]: annealed sampled lookahead
    state: tau, pair_k, mmr_lambda — selection knobs, set at construction
    state: last_value — root-perspective value acted on, stamped by select
    state: last_margin — max - mean(final top_k), stamped by select
    select(root, net, eps) -> Move: run the beam, play the argmax over
        root backed-up values; proven root mate short-circuits (max backup
        with certainty); fails fast (assert) on terminal root boards
        (Require: root has legal moves;
         Guarantee: returned move is legal at root AND is the argmax over
         the final survivor set AND last_value/last_margin are stamped)

class BehaviorPolicy [service]: the epsilon mixture over (beam, actor)
    state: epsilon — float, mutated per-episode by anneal()
    select(board, net) -> (move, action_idx, legal_mask, used_beam):
        beam arm with prob epsilon, masked-softmax sample otherwise;
        fails on terminal boards
    anneal(episode) -> None: linear decay epsilon_start -> epsilon_end over
        anneal_episodes
        (Maintain: epsilon_end <= epsilon <= epsilon_start)

class ReplayEpisode [value]: one self-play game's aligned transition arrays
    (states, action indices, shaped rewards, legal masks, beam flags)

class Trainer [service]: owns the self-play -> update loop and persistence
    state: net, optimizer, behavior, episode counter — mutated by train()
    play_episode() -> ReplayEpisode: one full self-play game; fails by
        truncation-to-draw at max_plies
    nstep_returns(ep) -> tensor: negamax n-step targets with bootstrapped
        tails; no-grad value evaluation
    update(ep) -> dict: one A2C gradient step on the episode batch
        (Require: len(ep) > 0; Guarantee: gradient clipped at max_grad_norm)
    train(episodes) -> None: full loop with sqlite metrics + checkpointing
    evaluate(games) -> float: greedy-policy win+draw rate vs a random mover
    agent_move(board, use_beam) -> Move: eval-time move, argmax pi or the
        eps=0 net-guided beam; fails on terminal boards
    elo_match(opponent, games, use_beam) -> (score, diff, lo, hi):
        color-alternating match vs one opponent; truncations adjudicated
        as draws (bias noted in output)
    elo_ladder(games, use_beam, stockfish_path, ...) -> None: run the rung
        sequence random -> heuristic -> SF skill 0 -> SF UCI_Elo 1320,
        printing anchored Elo only for calibrated rungs
    sweep(depths, widths, starts, games) -> None: grid the schedule knobs
        vs heuristic-1ply at eps=1.0 (net excluded), reporting score, Elo
        diff, and measured sec/move — the score-vs-compute frontier

class RandomMover [service]: uniform-legal baseline, the ladder's floor
class HeuristicMover [service]: 1-ply greedy over -Phi(child) with mate
    preference — the material/PST rung between random and Stockfish
class StockfishMover [service]: UCI engine wrapper; Skill Level or
    UCI_LimitStrength+UCI_Elo; anchor attribute set only when UCI_Elo is
    used (the only calibrated setting); fails fast if the binary is missing

def fib_schedule(depth, width, phi_start) -> tuple: pure schedule
    generation — descending Fibonacci, `width` phi-steps per level,
    terminating at position `phi_start`; belongs outside any class
def elo_from_score(score, games) -> (diff, lo, hi): logistic Elo difference
    with a normal-approx 95% CI, Laplace-clamped so 0/n and n/n report
    bounds — pure math, belongs outside any class
def board_to_tensor(board) -> tensor[13,8,8]: pure encoding from the
    side-to-move perspective (board is flipped for black so one net plays
    both colors) — no state, belongs outside any class
def move_to_index(move, turn) -> int: pure from*64+to mapping — stateless
def index_legal_mask(board) -> tensor[4096]: pure legality mask — stateless

--- Behavioral (FibonacciBeamSampler.select) ---

Input: root — position to move from, chess.Board
Input: net — ActorCritic for leaf value blending
Input: eps — heuristic blend weight in [0,1], float
Parameters: tau in R+ (survival temperature; values live in [-1,1], so
            tau ~ 0.25 discriminates while tau = 1 is nearly flat),
            pair_k in Z+ (continuations fetched per line per round),
            mmr_lambda >= 0 (root-move duplication penalty)
Uses: Phi — heuristic leaf score, Board -> (-1,1)
Uses: net.forward — batched value evaluation, boards -> (-1,1) each
Initialize: widths <- fib_schedule(phi_depth, phi_width, phi_start)
Initialize: lines <- one Line per uniformly fetched root move,     # global
            at most widths[0], leaf-valued via the blend           # to call

Loop over width w in widths[1:]:                    # depth-1 rounds
    Loop over each non-terminal line:
        fetch <- up to pair_k uniform continuations         # transient
        batch-evaluate fetched leaves via the blend
        commit the continuation maximal for that ply's mover
            ("pick max each pair" — the negamax backup)
        line.value <- root-perspective value of the committed leaf
            (Maintain: mate lines freeze at +-1 and are never extended)
    lines <- temperature-sample w survivors without replacement,
             score = value/tau - mmr_lambda * picked-root-move count
finalists <- the last survivor set with more than one line
played <- argmax over finalists by backed-up value    # :Root-commit:
last_value <- played line's value; last_margin <- max - mean(finalists)
Assert: the played root move is legal at root and maximal over finalists

Given a root with one legal move, When select runs, Then that move MUST be
    returned with last_value stamped from the child's evaluation.
Given a mating root move exists among the fetched candidates, When select
    runs, Then it MUST be played with last_value = 1.0 stamped.
Given tau -> 0, When survivors are sampled, Then survival MUST approach
    greedy top-w by backed-up value; the terminal commit is argmax always.

--- Behavioral (Trainer.train) ---

Input: episodes — total self-play games to run, int
Parameters: n_step in Z+, gamma in (0,1], lr, entropy_coef, value_coef,
            shaping_beta, max_plies, checkpoint_every
Uses: sqlite — metrics table append, (episode, ...) -> commit
Uses: torch.save/load — checkpoint round-trip, path -> state dict
Initialize: net <- checkpoint if exists else fresh        # global to run
Initialize: episode_counter <- restored or 0              # global to run

Loop over episode e in [start, episodes):
    behavior.anneal(e)                                    # epsilon decays
    Initialize ep <- play_episode()                       # loop-scoped
        Loop while not game over and ply < max_plies:
            (move, idx, mask, used_beam) <- behavior.select(board, net)
            r <- shaping(board, board') [+ terminal on mate]
            push transition; push board'
    G <- nstep_returns(ep)                                # negamax targets
    A <- G - V(states)                                    # advantage
    loss <- -(A.detach()*logpi).mean()
            + value_coef*mse(V, G) - entropy_coef*H(pi)
    step optimizer                                        (Maintain: grad
                                                          norm <= clip)
    When e % checkpoint_every == 0: save checkpoint + sqlite metrics
Assert: on normal completion, checkpoint on disk reflects the final episode

Given a fresh run, When train(N) completes, Then a subsequent train(M) MUST
    resume from episode N via the checkpoint, not restart from 0.
Given epsilon == 0, When select() is called, Then the beam arm MUST never
    fire and training is pure on-policy A2C.

================================================================================
Usage:
    python rl_chess_a2c.py                    # defaults to train
    python rl_chess_a2c.py train --episodes 500
    python rl_chess_a2c.py eval  --games 20
    python rl_chess_a2c.py elo   --games 40 --stockfish-path stockfish.exe
    python rl_chess_a2c.py elo   --games 40 --use-beam   # rating with search
    python rl_chess_a2c.py sweep --games 20 --sweep-depths 4,6,8
    python rl_chess_a2c.py play              # human (UCI moves) vs agent
Note: metrics schema changed from the pre-beam version (mean_rho ->
beam_frac); delete an old rl_chess_metrics.sqlite. Network checkpoints
remain compatible.
"""

import argparse
import math
import os
import random
import sqlite3
import time
from collections import Counter

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POLICY_SIZE = 64 * 64
CKPT_PATH = "rl_chess_a2c.pt"
DB_PATH = "rl_chess_metrics.sqlite"

# ----------------------------------------------------------------------------
# Heuristic scorer: material + piece-square tables. Phi is potential-based-
# shaping-compatible: squashed to (-1,1), zero at terminal states, always
# side-to-move perspective.
# ----------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

# White-perspective PSTs (a1=index 0). Mirrored for black at lookup.
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0],
    chess.KNIGHT: [
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [
         0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0],
    chess.QUEEN: [
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  5,  0,  0,  0,  0,-10,
       -10,  5,  5,  5,  5,  5,  0,-10,
         0,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5,
       -10,  0,  5,  5,  5,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20],
    chess.KING: [
        20, 30, 10,  0,  0, 10, 30, 20,
        20, 20,  0,  0,  0,  0, 20, 20,
       -10,-20,-20,-20,-20,-20,-20,-10,
       -20,-30,-30,-40,-40,-30,-30,-20,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30],
}


class HeuristicScorer:
    """Stateless board evaluation; the heuristic side of the beam's leaf
    blend and the shaping potential.

    phi: side-to-move-perspective score squashed to (-1,1); 0.0 on terminal
    boards so potential-based shaping telescopes cleanly to zero at episode
    end. Failure mode: none — pure arithmetic over python-chess piece maps.
    Note: uses is_game_over() without claim_draw so it stays valid on
    stack-less board copies inside the beam.
    """

    SQUASH = 600.0  # centipawns at which tanh reaches ~0.76

    def phi(self, board: chess.Board) -> float:
        if board.is_game_over():
            return 0.0
        score = 0
        for sq, piece in board.piece_map().items():
            v = PIECE_VALUES[piece.piece_type]
            idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
            v += PST[piece.piece_type][idx]
            score += v if piece.color == chess.WHITE else -v
        if board.turn == chess.BLACK:
            score = -score
        return math.tanh(score / self.SQUASH)


# ----------------------------------------------------------------------------
# Encoding: side-to-move perspective. One network plays both colors
# (AlphaZero/Leela convention). Pure functions — no state to guard.
# ----------------------------------------------------------------------------

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """13x8x8: 6 own-piece planes, 6 enemy planes, 1 all-ones bias plane.

    The board is vertically mirrored when black is to move so 'own pawns
    advance upward' always holds. Precondition: any legal Board.
    """
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    us = board.turn
    for sq, piece in board.piece_map().items():
        s = sq if us == chess.WHITE else chess.square_mirror(sq)
        row, col = divmod(s, 8)
        base = 0 if piece.color == us else 6
        planes[base + piece.piece_type - 1, row, col] = 1.0
    planes[12, :, :] = 1.0
    return torch.from_numpy(planes)


def move_to_index(move: chess.Move, turn: bool) -> int:
    """from*64+to in the side-to-move (mirrored-for-black) frame.

    Promotion piece is not encoded — underpromotions collapse (spec'd)."""
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Inverse mapping; resolves promotions to queen. Raises ValueError if
    the index is not legal on this board — callers must mask first."""
    f, t = divmod(idx, 64)
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    mv = chess.Move(f, t)
    if mv not in board.legal_moves:
        mv = chess.Move(f, t, promotion=chess.QUEEN)
    if mv not in board.legal_moves:
        raise ValueError(f"index {idx} illegal on {board.fen()}")
    return mv


def index_legal_mask(board: chess.Board) -> torch.Tensor:
    """4096 bool mask of legal from-to indices in the side-to-move frame."""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    for mv in board.legal_moves:
        mask[move_to_index(mv, board.turn)] = True
    return mask


def fib_schedule(depth: int, width: int, phi_start: int) -> tuple:
    """Beam-width schedule: descending Fibonacci over (1,2,3,5,8,...),
    consecutive levels `width` phi-steps apart, final level at Fibonacci
    position `phi_start`.

    (8,1,1) -> (34,21,13,8,5,3,2,1)   [the original schedule]
    (8,1,2) -> (55,34,21,13,8,5,3,2)  [whole ladder shifted up one]
    (8,2,1) -> (987,377,144,55,21,8,3,1) [width-2 pruning steps]
    Require: depth >= 1, width >= 1, phi_start >= 1.
    Guarantee: strictly decreasing for width >= 1, depth >= 2."""
    assert depth >= 1 and width >= 1 and phi_start >= 1
    need = phi_start + width * (depth - 1)
    fib = [1, 2]
    while len(fib) < need:
        fib.append(fib[-1] + fib[-2])
    return tuple(fib[phi_start - 1 + width * (depth - 1 - i)]
                 for i in range(depth))


def elo_from_score(score: float, games: int):
    """Elo difference implied by a match score with a normal-approx 95% CI.

    score = wins + 0.5*draws over `games`. Laplace clamp keeps s off {0,1}
    so shutout matches yield a finite bound instead of +-inf; treat those
    endpoints as bounds, not estimates. Failure mode: the CI overstates
    precision when draws dominate (draws are treated as Bernoulli halves).
    Require: games >= 1."""
    assert games >= 1
    n = games
    lo_c, hi_c = 0.5 / n, 1 - 0.5 / n

    def d(s):
        s = min(max(s, lo_c), hi_c)
        return 400 * math.log10(s / (1 - s))

    s = score / n
    se = math.sqrt(max(s * (1 - s), lo_c * hi_c) / n)
    return d(s), d(s - 1.96 * se), d(s + 1.96 * se)


# ----------------------------------------------------------------------------
# Opponent ladder for Elo estimation.
# ----------------------------------------------------------------------------

class RandomMover:
    """Uniform-legal baseline. No calibrated Elo exists for random play —
    folklore numbers vary by hundreds of points; this rung is ordinal only."""
    name, anchor = "random", None

    def move(self, board):
        return random.choice(list(board.legal_moves))

    def close(self):
        pass


class HeuristicMover:
    """1-ply greedy over -Phi(child), mate preferred. The rung between
    random and Stockfish; also what the beam's leaf blend anneals away
    from, so beating it cleanly is the first meaningful milestone.
    Failure mode: undefined on terminal boards (callers gate)."""
    name, anchor = "heuristic-1ply", None

    def __init__(self, scorer: HeuristicScorer):
        self.scorer = scorer

    def move(self, board):
        best_val, best = -float("inf"), []
        for mv in board.legal_moves:
            board.push(mv)
            val = float("inf") if board.is_checkmate() \
                else -self.scorer.phi(board)
            board.pop()
            if val > best_val:
                best_val, best = val, [mv]
            elif val == best_val:
                best.append(mv)
        return random.choice(best)

    def close(self):
        pass


class StockfishMover:
    """UCI engine wrapper. anchor is set only under UCI_LimitStrength +
    UCI_Elo — the sole setting Stockfish calibrates (long TC, CCRL 40/4);
    Skill Level rungs are ordinal. Fails fast (raises) if the binary is
    missing or not UCI. Require: close() after use (subprocess)."""

    def __init__(self, path, skill=None, elo=None, movetime=0.02):
        import chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.limit = chess.engine.Limit(time=movetime)
        if elo is not None:
            self.engine.configure({"UCI_LimitStrength": True,
                                   "UCI_Elo": elo})
            self.name, self.anchor = f"stockfish-elo{elo}", elo
        else:
            skill = 0 if skill is None else skill
            self.engine.configure({"Skill Level": skill})
            self.name, self.anchor = f"stockfish-skill{skill}", None

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


# ----------------------------------------------------------------------------
# Network: Leela-mini. Shared conv trunk, policy head (4096), value head.
# ----------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Joint policy+value net. Value is tanh in (-1,1), side-to-move
    perspective — matches Phi's scale and the terminal +-1 reward.
    Failure mode: garbage-in on tensors not shaped (B,13,8,8)."""

    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        layers = [nn.Conv2d(13, channels, 3, padding=1), nn.ReLU()]
        for _ in range(blocks - 1):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(16 * 64, POLICY_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * 64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh(),
        )

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @staticmethod
    def masked_policy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Softmax renormalized over legal indices only.
        Guarantee: probs sum to 1 over legal moves; illegal probs are 0."""
        logits = logits.masked_fill(~mask, -1e9)
        return F.softmax(logits, dim=-1)


# ----------------------------------------------------------------------------
# Fibonacci beam sampler: the annealed sampled-lookahead exploration arm.
# ----------------------------------------------------------------------------

class Line:
    """One surviving lookahead trajectory. Immutable root_move; board and
    value advance as the line deepens; terminal lines freeze."""

    __slots__ = ("board", "root_move", "value", "terminal")

    def __init__(self, board, root_move, value=0.0, terminal=False):
        self.board = board
        self.root_move = root_move
        self.value = value       # always root-mover perspective
        self.terminal = terminal


class FibonacciBeamSampler:
    """Beam widths follow descending Fibonacci per fib_schedule(phi_depth,
    phi_width, phi_start) — default (34,21,13,8,5,3,2,1): root candidates
    narrow to 1 (consecutive ratio ~ phi^width, the per-layer annealing).
    When phi_start > 1 the last level holds F(phi_start) lines and a final
    temperature pick reduces to 1.

    Per round: every surviving line fetches up to pair_k continuations
    uniformly (flat distribution), leaves are batch-evaluated with the blend
    eps*Phi + (1-eps)*V_net, the line commits to the continuation maximal
    for that ply's mover ("pick max each pair" — negamax backup), and the
    next beam is temperature-sampled without replacement with an MMR penalty
    on root-move duplicates. The last survivor's root move is played.

    Failure modes: asserts on terminal root; optimism bias vs the true
    minimax value because the opponent is restricted to pair_k sampled
    replies — raise pair_k to tighten (Kearns-style sparse sampling bound).
    """

    def __init__(self, scorer: HeuristicScorer, tau: float = 0.25,
                 pair_k: int = 2, mmr_lambda: float = 0.15,
                 phi_depth: int = 8, phi_width: int = 1,
                 phi_start: int = 1):
        self.scorer = scorer
        self.tau = tau
        self.pair_k = pair_k
        self.mmr_lambda = mmr_lambda
        self.widths = fib_schedule(phi_depth, phi_width, phi_start)
        # Stamped by every select() (spec/search-mcts.spec.md):
        #   last_value  — root-perspective backed-up value acted on; the
        #                 prior estimate in the cross-move signed :Delta:
        #   last_margin — :Move-margin: max - mean(final top_k), >= 0;
        #                 decisiveness of the winner over the field
        self.last_margin = 0.0
        self.last_value = 0.0

    # -- leaf evaluation ---------------------------------------------------
    @torch.no_grad()
    def _blend_values(self, boards, net, eps):
        """eps*Phi + (1-eps)*V_net, batched; side-to-move perspective.
        Terminal boards short-circuit: mate = -1 for the mated side, else 0.
        Guarantee: one value per board, each in [-1, 1]."""
        vals = [None] * len(boards)
        live_idx = []
        for i, b in enumerate(boards):
            if b.is_checkmate():
                vals[i] = -1.0                     # side to move is mated
            elif b.is_game_over():                 # stalemate / material
                vals[i] = 0.0
            else:
                live_idx.append(i)
        if live_idx:
            if eps < 1.0:
                x = torch.stack([board_to_tensor(boards[i])
                                 for i in live_idx]).to(DEVICE)
                _, v_net = net(x)
                v_net = v_net.cpu().tolist()
            else:
                v_net = [0.0] * len(live_idx)
            for j, i in enumerate(live_idx):
                h = self.scorer.phi(boards[i]) if eps > 0.0 else 0.0
                vals[i] = eps * h + (1 - eps) * v_net[j]
        return vals

    @staticmethod
    def _to_root(value_stm: float, leaf: chess.Board, root_turn: bool):
        """Perspective flip: leaf value is side-to-move; negate when the
        leaf mover differs from the root mover (the negamax sign)."""
        return value_stm if leaf.turn == root_turn else -value_stm

    # -- rounds --------------------------------------------------------------
    def _fan_out(self, lines, net, eps, root_turn):
        """Extend each non-terminal line by its best fetched continuation.
        Maintain: mate/draw lines freeze and are never extended."""
        cand_boards, owners = [], []       # flat batch across all lines
        for li, line in enumerate(lines):
            if line.terminal:
                continue
            legal = list(line.board.legal_moves)
            if not legal:                  # frozen by rule, defensive
                line.terminal = True
                continue
            for mv in random.sample(legal, min(self.pair_k, len(legal))):
                b = line.board.copy(stack=False)
                b.push(mv)
                cand_boards.append(b)
                owners.append(li)
        if not cand_boards:
            return
        vals = self._blend_values(cand_boards, net, eps)
        # negamax commit: the mover at line.board picks the max of the
        # fetched pair from their own perspective = -value(leaf stm)
        best = {}                          # line idx -> (mover_val, board)
        for b, li, v in zip(cand_boards, owners, vals):
            mover_val = -v
            if li not in best or mover_val > best[li][0]:
                best[li] = (mover_val, b)
        for li, (_, b) in best.items():
            line = lines[li]
            line.board = b
            if b.is_game_over():
                # checkmate: leaf stm is mated -> stm value -1 -> root flip
                line.value = self._to_root(
                    -1.0 if b.is_checkmate() else 0.0, b, root_turn)
                line.terminal = True
            else:
                line.value = None          # filled by batch pass below
        # batch-fill values of freshly advanced non-terminal lines
        pending = [l for l in lines if not l.terminal and l.value is None]
        if pending:
            vals = self._blend_values([l.board for l in pending], net, eps)
            for l, v in zip(pending, vals):
                l.value = self._to_root(v, l.board, root_turn)

    def _prune(self, lines, width):
        """Temperature-sample `width` survivors without replacement.
        Score = value/tau - mmr_lambda * (already-picked same-root count).
        Guarantee: len(result) == min(width, len(lines))."""
        if len(lines) <= width:
            return lines
        pool = list(lines)
        picked, root_counts = [], Counter()
        while len(picked) < width:
            logits = torch.tensor(
                [l.value / self.tau
                 - self.mmr_lambda * root_counts[l.root_move.uci()]
                 for l in pool])
            i = int(torch.multinomial(F.softmax(logits, dim=0), 1).item())
            chosen = pool.pop(i)
            picked.append(chosen)
            root_counts[chosen.root_move.uci()] += 1
        return picked

    # -- entry ---------------------------------------------------------------
    def select(self, root: chess.Board, net, eps: float) -> chess.Move:
        """Run the beam. Require: root has legal moves.
        Guarantee: returned move is legal at root; the played move is the
        argmax over root backed-up values (:Root-commit:); last_value and
        last_margin are stamped on EVERY exit path, so a cross-move :Delta:
        consumer never reads a stale estimate.

        Exploration lives upstream (uniform fetch, temperature/MMR
        survival, the eps-mixture's direct-pi branch); the commit is
        deterministic so next volley's re-search audits the exploited
        choice, not a lottery. Failure mode: sampled backups are optimistic
        when a refutation went unsampled — the signed :Delta: next volley
        is the correction channel."""
        legal = list(root.legal_moves)
        assert legal, "beam called on terminal board"
        root_turn = root.turn
        if len(legal) == 1:
            # Forced move: stamp the child's actual value, not a
            # fabricated 0.0 — a phantom estimate poisons :Delta:.
            b = root.copy(stack=False)
            b.push(legal[0])
            v = self._blend_values([b], net, eps)[0]
            self.last_value = self._to_root(v, b, root_turn)
            self.last_margin = 0.0
            return legal[0]
        fetch = random.sample(legal, min(self.widths[0], len(legal)))
        boards = []
        for mv in fetch:
            b = root.copy(stack=False)
            b.push(mv)
            boards.append(b)
        vals = self._blend_values(boards, net, eps)
        lines, mate_mv, rvs = [], None, []
        for mv, b, v in zip(fetch, boards, vals):
            rv = self._to_root(v, b, root_turn)   # checkmate child -> +1
            rvs.append(rv)
            if b.is_checkmate() and mate_mv is None:
                mate_mv = mv
            lines.append(Line(b, mv, rv, b.is_game_over()))
        if mate_mv is not None:
            # Proven win: max backup with certainty — stamped, not stale.
            self.last_value = 1.0
            self.last_margin = 1.0 - sum(rvs) / len(rvs)
            return mate_mv
        finalists = lines
        for i, width in enumerate(self.widths[1:]):
            last = (i == len(self.widths) - 2)
            if last and width == 1:
                # :Root-commit: argmax supersedes the final collapse —
                # sampling one line only to discard it is dead work.
                break
            lines = self._prune(lines, width)
            if len(lines) > 1:
                finalists = lines          # last top_k (>1) survivor set
            if not last:
                self._fan_out(lines, net, eps, root_turn)
        # :Move-margin: over the final top_k (root-perspective backed-up
        # values, all extended to equal depth). >= 0; distinct from :Delta:.
        best = max(finalists, key=lambda l: l.value)
        self.last_margin = best.value \
            - sum(l.value for l in finalists) / len(finalists)
        self.last_value = best.value
        assert best.root_move in root.legal_moves
        return best.root_move


# ----------------------------------------------------------------------------
# Behavior policy: epsilon mixture over (beam search, actor softmax).
# ----------------------------------------------------------------------------

class BehaviorPolicy:
    """With prob epsilon the beam picks; otherwise sample pi. The exact
    mixture probability of a beam pick is intractable (marginal over seven
    stochastic pruning rounds), so no importance ratio is computed anywhere
    downstream — the Q-learning stance, bias annealing away with epsilon.
    Failure mode: undefined on terminal boards (asserted in the beam)."""

    def __init__(self, sampler: FibonacciBeamSampler,
                 eps_start=0.9, eps_end=0.0, anneal_episodes=2000):
        self.sampler = sampler
        self.eps_start, self.eps_end = eps_start, eps_end
        self.anneal_episodes = anneal_episodes
        self.epsilon = eps_start

    def anneal(self, episode: int):
        frac = min(1.0, episode / max(1, self.anneal_episodes))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def select(self, board: chess.Board, net: ActorCritic):
        """Returns (move, action_idx, legal_mask, used_beam)."""
        mask = index_legal_mask(board)
        if self.epsilon > 0 and random.random() < self.epsilon:
            move = self.sampler.select(board, net, self.epsilon)
            return move, move_to_index(move, board.turn), mask, True
        x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
        logits, _ = net(x)
        pi = ActorCritic.masked_policy(logits.squeeze(0).cpu(), mask)
        idx = int(torch.multinomial(pi, 1).item())
        return index_to_move(idx, board), idx, mask, False


# ----------------------------------------------------------------------------
# Trainer: self-play -> n-step negamax A2C update -> persistence.
# ----------------------------------------------------------------------------

class ReplayEpisode:
    """Aligned transition arrays for one game."""

    def __init__(self):
        self.states, self.actions = [], []
        self.rewards, self.masks, self.beam_flags = [], [], []

    def push(self, s, a, r, mask, used_beam):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.masks.append(mask)
        self.beam_flags.append(used_beam)

    def __len__(self):
        return len(self.states)


class Trainer:
    """Owns the loop. See Behavioral blocks in the module docstring."""

    def __init__(self, n_step=8, gamma=0.99, lr=1e-3, entropy_coef=0.01,
                 value_coef=0.5, shaping_beta=1.0,
                 max_plies=200, checkpoint_every=25,
                 eps_start=0.9, eps_end=0.0, anneal_episodes=2000,
                 beam_tau=0.25, pair_k=2, mmr_lambda=0.15,
                 phi_depth=8, phi_width=1, phi_start=1):
        self.n_step, self.gamma = n_step, gamma
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.shaping_beta = shaping_beta
        self.max_plies, self.checkpoint_every = max_plies, checkpoint_every

        self.scorer = HeuristicScorer()
        self.net = ActorCritic().to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.beam_kwargs = dict(tau=beam_tau, pair_k=pair_k,
                                mmr_lambda=mmr_lambda, phi_depth=phi_depth,
                                phi_width=phi_width, phi_start=phi_start)
        sampler = FibonacciBeamSampler(self.scorer, **self.beam_kwargs)
        self.behavior = BehaviorPolicy(sampler, eps_start, eps_end,
                                       anneal_episodes)
        self.episode = 0
        self._load_if_exists()
        self._init_db()

    # -- persistence ----------------------------------------------------
    def _load_if_exists(self):
        if os.path.exists(CKPT_PATH):
            ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
            self.net.load_state_dict(ck["net"])
            self.opt.load_state_dict(ck["opt"])
            self.episode = ck["episode"]
            print(f"resumed from checkpoint at episode {self.episode}")

    def _save(self):
        torch.save({"net": self.net.state_dict(),
                    "opt": self.opt.state_dict(),
                    "episode": self.episode}, CKPT_PATH)

    def _init_db(self):
        self.db = sqlite3.connect(DB_PATH)
        self.db.execute("""CREATE TABLE IF NOT EXISTS metrics (
            episode INTEGER PRIMARY KEY, ts REAL, epsilon REAL, plies INTEGER,
            outcome TEXT, loss REAL, policy_loss REAL, value_loss REAL,
            entropy REAL, beam_frac REAL)""")
        self.db.commit()

    # -- self-play -------------------------------------------------------
    def play_episode(self) -> tuple:
        """One self-play game under the mixture. Rewards are shaped per-ply
        from the mover's perspective; terminal mate reward is +1 to the
        mover. Truncates to a draw at max_plies."""
        board = chess.Board()
        ep = ReplayEpisode()
        outcome = "draw"
        for _ in range(self.max_plies):
            if board.is_game_over(claim_draw=True):
                break
            s = board_to_tensor(board)
            phi_s = self.scorer.phi(board)
            move, idx, mask, used_beam = self.behavior.select(board, self.net)
            board.push(move)

            terminal = board.is_game_over(claim_draw=True)
            r = 0.0
            if board.is_checkmate():
                r = 1.0  # mover delivered mate
                outcome = "mate"
            # potential-based shaping, negamax sign: Phi(s') is opponent-view
            phi_next = 0.0 if terminal else self.scorer.phi(board)
            r += self.shaping_beta * (self.gamma * (-phi_next) - phi_s)

            ep.push(s, idx, r, mask, used_beam)
        return ep, outcome

    # -- returns ----------------------------------------------------------
    @torch.no_grad()
    def nstep_returns(self, ep: ReplayEpisode) -> torch.Tensor:
        """Negamax n-step targets:
        G_t = sum_{k<K} (-gamma)^k r_{t+k} + (-gamma)^K V(s_{t+K}),
        K = min(n_step, remaining); bootstrap term dropped at terminal.
        The (-1)^k alternation is the two-player perspective flip."""
        T = len(ep)
        tail_idx = [t + self.n_step for t in range(T)
                    if t + self.n_step < T]
        v_cache = {}
        if tail_idx:
            xs = torch.stack([ep.states[i] for i in tail_idx]).to(DEVICE)
            _, vals = self.net(xs)
            for i, v in zip(tail_idx, vals.cpu().tolist()):
                v_cache[i] = v
        G = torch.zeros(T)
        for t in range(T):
            g, sign = 0.0, 1.0
            K = min(self.n_step, T - t)
            for k in range(K):
                g += sign * (self.gamma ** k) * ep.rewards[t + k]
                sign = -sign
            if t + self.n_step < T:  # non-terminal tail -> bootstrap
                g += sign * (self.gamma ** K) * v_cache[t + self.n_step]
            G[t] = g
        return G

    # -- update -----------------------------------------------------------
    def update(self, ep: ReplayEpisode) -> dict:
        """One A2C step on the episode batch.
        Require: len(ep) > 0. Guarantee: grad norm clipped at 1.0.
        No importance weights: beam-chosen actions train the policy
        unweighted (search-as-teacher, AlphaZero-flavored improvement);
        bias vanishes once epsilon anneals to 0."""
        assert len(ep) > 0
        S = torch.stack(ep.states).to(DEVICE)
        A_idx = torch.tensor(ep.actions, dtype=torch.long, device=DEVICE)
        masks = torch.stack(ep.masks).to(DEVICE)

        G = self.nstep_returns(ep).to(DEVICE)
        logits, V = self.net(S)
        pi = ActorCritic.masked_policy(logits, masks)
        pi_a = pi.gather(1, A_idx.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        log_pi_a = pi_a.log()

        adv = (G - V).detach()                       # A = R - V(S)
        entropy = -(pi.clamp_min(1e-12).log() * pi).sum(dim=1).mean()
        policy_loss = -(adv * log_pi_a).mean()
        value_loss = F.mse_loss(V, G)
        loss = policy_loss + self.value_coef * value_loss \
            - self.entropy_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        beam_frac = sum(ep.beam_flags) / len(ep)
        return {"loss": loss.item(), "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(), "entropy": entropy.item(),
                "beam_frac": beam_frac}

    # -- loop ---------------------------------------------------------------
    def train(self, episodes: int):
        start = self.episode
        for e in range(start, episodes):
            self.episode = e
            self.behavior.anneal(e)
            ep, outcome = self.play_episode()
            stats = self.update(ep)
            self.db.execute(
                "INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?)",
                (e, time.time(), self.behavior.epsilon, len(ep), outcome,
                 stats["loss"], stats["policy_loss"], stats["value_loss"],
                 stats["entropy"], stats["beam_frac"]))
            if e % self.checkpoint_every == 0 or e == episodes - 1:
                self._save()
                self.db.commit()
                print(f"ep {e:5d} eps={self.behavior.epsilon:.3f} "
                      f"plies={len(ep):3d} {outcome:5s} "
                      f"loss={stats['loss']:+.4f} "
                      f"vloss={stats['value_loss']:.4f} "
                      f"H={stats['entropy']:.3f} "
                      f"beam={stats['beam_frac']:.2f}")
        self._save()
        self.db.commit()
        print(f"done: everything through episode {self.episode} "
              f"checkpointed to {CKPT_PATH}, metrics in {DB_PATH}")

    # -- evaluation -----------------------------------------------------
    @torch.no_grad()
    def evaluate(self, games: int = 20) -> float:
        """Greedy policy (agent) vs uniform-random mover. Returns score in
        [0,1] with draws counted 0.5. Agent alternates colors."""
        score = 0.0
        for g in range(games):
            board = chess.Board()
            agent_color = chess.WHITE if g % 2 == 0 else chess.BLACK
            for _ in range(self.max_plies):
                if board.is_game_over(claim_draw=True):
                    break
                if board.turn == agent_color:
                    mask = index_legal_mask(board)
                    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
                    logits, _ = self.net(x)
                    pi = ActorCritic.masked_policy(logits.squeeze(0).cpu(), mask)
                    board.push(index_to_move(int(pi.argmax()), board))
                else:
                    board.push(random.choice(list(board.legal_moves)))
            res = board.result(claim_draw=True)
            if res == "1-0":
                score += 1.0 if agent_color == chess.WHITE else 0.0
            elif res == "0-1":
                score += 1.0 if agent_color == chess.BLACK else 0.0
            else:
                score += 0.5
        rate = score / games
        print(f"eval: {rate:.2%} vs random over {games} games")
        return rate

    # -- elo ---------------------------------------------------------------
    @torch.no_grad()
    def agent_move(self, board, use_beam: bool):
        """Eval-time move: argmax pi, or the eps=0 net-guided beam (the
        agent *with* search — how Leela-style ratings are quoted).
        Require: board not terminal."""
        if use_beam:
            return self.behavior.sampler.select(board, self.net, 0.0)
        mask = index_legal_mask(board)
        x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
        logits, _ = self.net(x)
        pi = ActorCritic.masked_policy(logits.squeeze(0).cpu(), mask)
        return index_to_move(int(pi.argmax()), board)

    @torch.no_grad()
    def _match(self, opponent, games: int, move_fn):
        """Color-alternating match core. move_fn(board) -> agent Move.
        Truncations at max_plies adjudicated as draws (score bias toward
        0.5 — callers report the count). Returns (score, trunc, agent_moves,
        agent_seconds). Require: opponent.close() is the caller's job."""
        score, trunc, n_moves, t_agent = 0.0, 0, 0, 0.0
        for g in range(games):
            board = chess.Board()
            agent_color = chess.WHITE if g % 2 == 0 else chess.BLACK
            plies = 0
            while not board.is_game_over(claim_draw=True) \
                    and plies < self.max_plies:
                if board.turn == agent_color:
                    t0 = time.time()
                    board.push(move_fn(board))
                    t_agent += time.time() - t0
                    n_moves += 1
                else:
                    board.push(opponent.move(board))
                plies += 1
            if not board.is_game_over(claim_draw=True):
                trunc += 1
            res = board.result(claim_draw=True)
            if res == "1-0":
                score += 1.0 if agent_color == chess.WHITE else 0.0
            elif res == "0-1":
                score += 1.0 if agent_color == chess.BLACK else 0.0
            else:
                score += 0.5
        return score, trunc, n_moves, t_agent

    @torch.no_grad()
    def elo_match(self, opponent, games: int, use_beam: bool):
        """One ladder rung: match + Elo report."""
        score, trunc, _, _ = self._match(
            opponent, games, lambda b: self.agent_move(b, use_beam))
        diff, lo, hi = elo_from_score(score, games)
        shutout = score in (0.0, float(games))
        bound = " (bound only — shutout match)" if shutout else ""
        line = (f"vs {opponent.name:18s} score {score:5.1f}/{games} "
                f"D={diff:+6.0f} [95%: {lo:+.0f},{hi:+.0f}]{bound}"
                f"{'  trunc-draws=%d' % trunc if trunc else ''}")
        if opponent.anchor is not None and not shutout:
            line += (f"  -> agent ~ {opponent.anchor + diff:.0f} Elo "
                     f"[{opponent.anchor + lo:.0f},{opponent.anchor + hi:.0f}]"
                     f" (CCRL-40/4-anchored, short-TC caveat)")
        elif opponent.anchor is not None:
            line += f"  -> agent <= ~{opponent.anchor + diff:.0f} Elo"
        print(line)
        return score, diff, lo, hi

    def elo_ladder(self, games: int, use_beam: bool, stockfish_path: str,
                   stockfish_skill, stockfish_elo, movetime: float):
        """Run the rung sequence. Random and heuristic rungs always run;
        Stockfish rungs run if the binary opens. A calibrated Elo number
        exists only from the UCI_Elo rung upward — everything below is
        ordinal progress."""
        rungs = [RandomMover(), HeuristicMover(self.scorer)]
        try:
            if stockfish_skill is not None:
                rungs.append(StockfishMover(stockfish_path,
                                            skill=stockfish_skill,
                                            movetime=movetime))
            rungs.append(StockfishMover(stockfish_path,
                                        elo=stockfish_elo,
                                        movetime=movetime))
        except FileNotFoundError:
            print(f"stockfish not found at '{stockfish_path}' — "
                  f"running uncalibrated rungs only")
        mode = "beam(eps=0)" if use_beam else "argmax-policy"
        print(f"elo ladder: {games} games/rung, agent={mode}, "
              f"max_plies={self.max_plies}")
        for opp in rungs:
            try:
                self.elo_match(opp, games, use_beam)
            finally:
                opp.close()

    def sweep(self, depths, widths, starts, games: int):
        """Grid the schedule knobs, measure score vs compute. Each config
        plays `games` vs heuristic-1ply with eps=1.0 leaves (pure heuristic
        search — the net never runs), isolating search strength from net
        quality. Reported sec/move is the real training-time beam cost at
        high epsilon. Failure mode: small `games` gives wide Elo CIs —
        treat sec/move as exact and score as noisy."""
        opp = HeuristicMover(self.scorer)
        print(f"sweep: {games} games/config vs {opp.name}, eps=1.0, "
              f"max_plies={self.max_plies}")
        print(f"{'depth':>5} {'width':>5} {'start':>5}  "
              f"{'schedule':<28} {'score':>9} {'D':>6} {'s/move':>7}")
        for d in depths:
            for w in widths:
                for s in starts:
                    kw = dict(self.beam_kwargs,
                              phi_depth=d, phi_width=w, phi_start=s)
                    sampler = FibonacciBeamSampler(self.scorer, **kw)
                    score, trunc, n, t = self._match(
                        opp, games,
                        lambda b: sampler.select(b, self.net, 1.0))
                    diff, _, _ = elo_from_score(score, games)
                    sched = str(sampler.widths)
                    if len(sched) > 28:
                        sched = sched[:25] + '...'
                    print(f"{d:5d} {w:5d} {s:5d}  {sched:<28} "
                          f"{score:5.1f}/{games:<3d} {diff:+6.0f} "
                          f"{t / max(1, n):7.3f}"
                          f"{'  trunc=%d' % trunc if trunc else ''}")
        opp.close()

    # -- interactive ------------------------------------------------------
    @torch.no_grad()
    def play_human(self):
        """Human enters UCI moves (e.g. e2e4); agent replies greedily."""
        board = chess.Board()
        human = chess.WHITE
        while not board.is_game_over(claim_draw=True):
            print(board, "\n")
            if board.turn == human:
                try:
                    mv = chess.Move.from_uci(input("your move (uci): ").strip())
                except ValueError:
                    print("bad format")
                    continue
                if mv not in board.legal_moves:
                    print("illegal")
                    continue
                board.push(mv)
            else:
                mask = index_legal_mask(board)
                x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
                logits, v = self.net(x)
                pi = ActorCritic.masked_policy(logits.squeeze(0).cpu(), mask)
                mv = index_to_move(int(pi.argmax()), board)
                print(f"agent: {mv.uci()}  (V={v.item():+.3f})")
                board.push(mv)
        print(board, "\nresult:", board.result(claim_draw=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="train",
               choices=["train", "eval", "play", "elo", "sweep"])
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--anneal-episodes", type=int, default=2000)
    p.add_argument("--eps-start", type=float, default=0.9)
    p.add_argument("--n-step", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--beam-tau", type=float, default=0.25)
    p.add_argument("--pair-k", type=int, default=2)
    p.add_argument("--mmr-lambda", type=float, default=0.15)
    p.add_argument("--use-beam", action="store_true",
                   help="rate the agent with the eps=0 net-guided beam")
    p.add_argument("--stockfish-path", type=str, default="stockfish")
    p.add_argument("--stockfish-skill", type=int, default=0,
                   help="ordinal rung; pass -1 to skip")
    p.add_argument("--stockfish-elo", type=int, default=1320,
                   help="calibrated rung (UCI_LimitStrength), floor 1320")
    p.add_argument("--movetime", type=float, default=0.02)
    p.add_argument("--phi-depth", type=int, default=8)
    p.add_argument("--phi-width", type=int, default=1)
    p.add_argument("--phi-start", type=int, default=1)
    p.add_argument("--sweep-depths", type=str, default="4,6,8")
    p.add_argument("--sweep-widths", type=str, default="1,2")
    p.add_argument("--sweep-starts", type=str, default="1,2")
    args = p.parse_args()

    t = Trainer(n_step=args.n_step, max_plies=args.max_plies,
                eps_start=args.eps_start, anneal_episodes=args.anneal_episodes,
                beam_tau=args.beam_tau, pair_k=args.pair_k,
                mmr_lambda=args.mmr_lambda, phi_depth=args.phi_depth,
                phi_width=args.phi_width, phi_start=args.phi_start)
    if args.mode == "train":
        t.train(args.episodes)
    elif args.mode == "eval":
        t.evaluate(args.games)
    elif args.mode == "elo":
        skill = None if args.stockfish_skill < 0 else args.stockfish_skill
        t.elo_ladder(args.games, args.use_beam, args.stockfish_path,
                     skill, args.stockfish_elo, args.movetime)
    elif args.mode == "sweep":
        t.sweep([int(x) for x in args.sweep_depths.split(",")],
                [int(x) for x in args.sweep_widths.split(",")],
                [int(x) for x in args.sweep_starts.split(",")],
                args.games)
    else:
        t.play_human()


if __name__ == "__main__":
    main()