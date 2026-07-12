"""Merge 2 — afterstate TD(λ) Q-learning from reward (spec/q-learning.spec.md).

Self-play on Merge 0's ChessEnv with a raw-board value net (Merge 1's 769-dim encode). Move selection is
SOFTMAX over afterstate values with an annealed temperature τ (behavior policy); the LEARNING target is the
greedy/max afterstate value bootstrapped into a λ-return (off-policy). Every game — win, loss, OR draw —
feeds a uniform replay buffer. Strength is measured via the shared Elo harness and the run emits a
KILL-CHECK line for Optuna (tune_qlearn.py). All knobs read from QLEARN_* env vars.

  value    : V(encode(board)) = tanh(net(x)) in [-1,1], White-absolute (arch=linear default, mlp behind flag)
  behavior : softmax(σ·V(afterstate)/τ), σ=+1 White / −1 Black; τ warms τ_start->τ_floor over WARMUP·training
  target   : G^λ over rewards + GREEDY afterstate bootstrap (value_targets.lambda_return); λ=0 ⇒ 1-step Q
  buffer   : deque(maxlen) of (encode(s), G^λ), sampled uniformly with replacement
  objective: final SF@1320-anchored Elo (measure_elo.measure) — NOT loss

Trained in EPOCHS (fresh self-play games/epoch from the current policy), patience-stopped; λ is
variance-adaptive across epochs (noisy targets -> lower λ). Usage: python qlearn.py [epoch_games] [max_epochs]
"""
import os
import sys
import json
import math
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import chess
import chess.engine

from chess_rl import ChessEnv
from cem_loop import encode, encode_features, NIN, NFEAT
from value_targets import lambda_return
from measure_ladder import random_mover, heuristic_mover
from measure_elo import measure, play as elo_play, elo_diff
from search_policy import search_move

METRICS = os.environ.get("QLEARN_METRICS", "data/qlearn_metrics.jsonl")   # override so parallel
# experiment arms don't truncate/interleave the console run's live plot stream

# --- QLEARN_* knobs (Optuna sweeps these via env) -------------------------------------------
ALPHA      = float(os.environ.get("QLEARN_ALPHA", "3e-3"))     # SGD/Adam step size α
GAMMA      = float(os.environ.get("QLEARN_GAMMA", "0.99"))     # discount (shorter mate worth more)
LAMBDA     = float(os.environ.get("QLEARN_LAMBDA", "0.7"))     # TD↔MC dial; 0=1-step Q, 1=MC
LAMBDA_START = float(os.environ.get("QLEARN_LAMBDA_START", "0.95"))  # λ anneal: MC-heavy while V is
# untrained (trust real outcomes), decaying toward LAMBDA (trust the bootstrap) via the SAME asymptotic
# :anneal: shape as τ. Set QLEARN_LAMBDA_START <= LAMBDA to disable. Variance adaptation stacks on top
# (the Watkins-flavored opposite force: noisy/exploratory data spikes σ_TD and trims λ reactively).
LAMBDA_WARMUP = float(os.environ.get("QLEARN_LAMBDA_WARMUP", os.environ.get("QLEARN_WARMUP", "0.5")))
# λ's e-folding fraction; defaults to WARMUP so the two dials move in lock-step unless split.
TAU_START  = float(os.environ.get("QLEARN_TAU_START", "0.7"))
TAU_FLOOR  = float(os.environ.get("QLEARN_TAU_FLOOR", "0.05"))
WARMUP     = float(os.environ.get("QLEARN_WARMUP", "0.5"))     # THE tuned knob: fraction of training over
#   which τ slides τ_start->τ_floor (then holds). The single temperature warmup-ratio Optuna sweeps.
BATCH      = int(os.environ.get("QLEARN_BATCH", "256"))
BUFFER     = int(os.environ.get("QLEARN_BUFFER", "100000"))
BATCH_GAMES= int(os.environ.get("QLEARN_BATCH_GAMES", "20"))   # games per generate->train cycle; the user's
# x/y: epoch_games / batch_games = num batches per epoch. Clamped to <= epoch_games. Decoupled from the
# plot cadence (LOG_EVERY) and from the SGD minibatch (BATCH, in POSITIONS).
BUFFER_EPOCHS = float(os.environ.get("QLEARN_BUFFER_EPOCHS", "0"))  # >0: cap the replay window at ~this many
# EPOCHS of positions (epoch_games x ~64 plies) instead of the absolute BUFFER cap. Controls the policy-
# admixture horizon: stored targets are FROZEN λ-returns biased toward the policy that generated them, so a
# smaller horizon = fresher, less-admixed targets at the cost of variance smoothing. 0 = legacy 100k cap.
TRAIN_STEPS= int(os.environ.get("QLEARN_TRAIN_STEPS", "200"))  # SGD minibatches per iteration
ARCH       = os.environ.get("QLEARN_ARCH", "linear")
HIDDEN     = int(os.environ.get("QLEARN_HIDDEN", "64"))        # mlp hidden width (capacity rung)
PLY_CAP    = int(os.environ.get("QLEARN_PLY_CAP", "160"))
ELO_GAMES  = int(os.environ.get("QLEARN_ELO_GAMES", "20"))
PROXY_GAMES= int(os.environ.get("QLEARN_PROXY_GAMES", "20"))
SEED       = int(os.environ.get("QLEARN_SEED", "0"))
EPOCH_GAMES= int(os.environ.get("QLEARN_EPOCH_GAMES", "2000"))  # fresh self-play games generated per epoch
MAX_EPOCHS = int(os.environ.get("QLEARN_MAX_EPOCHS", "3"))
PATIENCE   = int(os.environ.get("QLEARN_PATIENCE", "2"))        # stop after N non-improving epochs
ADAPTIVE_LAMBDA = os.environ.get("QLEARN_ADAPTIVE_LAMBDA", "1") == "1"
BEHAVIOR   = os.environ.get("QLEARN_BEHAVIOR", "softmax")      # softmax = 1-ply (Merge 2) | search =
# depth-2 negamax over V (Merge 4, spec :Search-policy:) for BOTH behavior and measurement; the
# λ-return TARGETS stay 1-ply greedy (unchanged semantics).
SEARCH_W   = int(os.environ.get("QLEARN_SEARCH_WIDTH", "8"))   # top-K root moves get reply expansion
SEARCH_D   = int(os.environ.get("QLEARN_SEARCH_DEPTH", "2"))   # search depth for behavior/measure (2|3)
ENC        = os.environ.get("QLEARN_ENC", "pst")               # pst = raw 769 planes | kc = Merge 6
# KnightCap-donor features (spec/eval-features.spec.md :Phase-B:, 809-dim). Changes the input
# manifold -> part of every checkpoint and study identity.
ENC_FN     = encode_features if ENC == "kc" else encode
NIN_ENC    = NFEAT if ENC == "kc" else NIN
ZCA        = os.environ.get("QLEARN_ZCA", "")                  # E3 (purist capacity #1): path to
# models/zca.npz -> train in the WHITENED space x' = Z(x - mu) (conditioning only, no capacity
# change); native searcher gets back-converted raw weights (w_raw = Z w', b_raw = b' - w_raw·mu).
_ZCA_Z = _ZCA_MU = None
if ZCA:
    _z = np.load(ZCA)
    _ZCA_Z, _ZCA_MU = _z["Z"].astype(np.float32), _z["mu"].astype(np.float32)
    _base_enc = ENC_FN
    def ENC_FN(board, _e=_base_enc):                           # noqa: F811 — deliberate wrap
        return _ZCA_Z @ (_e(board) - _ZCA_MU)
RSEARCH_MOD = os.environ.get("QLEARN_RSEARCH_MOD", "rsearch")  # native module name (rsearch2 =
# side-built v2/v3 while the primary .pyd is file-locked by an older run)
PARGEN     = int(os.environ.get("QLEARN_PARGEN", "0"))         # Merge 9 (spec/parallel-generation.
# spec.md): >0 = generate games in native parallel batches of this size (play_games); opponent =
# frozen SELF at QLEARN_PARGEN_OPP_D (0=random); SF/graded rungs remain the python serial path,
# so PARGEN and OPP=graded are mutually exclusive by construction.
PARGEN_OPP_D = int(os.environ.get("QLEARN_PARGEN_OPP_D", "1"))
PARGEN_EPS = float(os.environ.get("QLEARN_PARGEN_EPS", "0.1")) # ε: state-coverage exploration
PARGEN_THREADS = int(os.environ.get("QLEARN_PARGEN_THREADS", "12"))
PARGEN_SOFTMAX = int(os.environ.get("QLEARN_PARGEN_SOFTMAX", "0"))  # Merge 11 :Softmax-tau::
# 1 = native generation samples moves ∝ exp(σ·v1/τ) with the trainer's OWN annealed τ —
# optimism-directed exploration (inflated init values attract visits until deflated); pairs
# with the optimistic zero seed. Set eps=0 when on (τ subsumes uniform exploration).
# Requires rsearch module ≥ v3.6 (play_games tau kwarg).
SURPRISE = int(os.environ.get("QLEARN_SURPRISE", "0"))  # Merge 14 :Rating-surprise:
# (spec/relative-reward.spec.md): the trivium's OUTCOME term uses z_rel = 2·(s − E[s|ratings])
# instead of raw z on GRADED games — a draw vs a stronger rung pays positive surprise, so
# non-terminating games carry signal. E from Bradley-Terry over declared rung ratings with
# the SF@1320 anchor pinning the gauge (:Anchor-pinned:); the agent's own rating updates
# online (Elo K). Mirror self-play keeps raw z (:No-self-reference: — symmetry ⇒ e=0.5).
SURPRISE_K = 32.0                                        # standard Elo K (declared constant)
GRPO = int(os.environ.get("QLEARN_GRPO", "0"))           # Merge 14 :GRPO-group: — "GRPO and
# Elo, nothing fancier" (operator): per-chunk GROUP z-normalization of the agent-relative
# Elo-surprise scores, replicating RL_v2/ac_trainer._whiten_group (our own donor code;
# std<1e-8 -> all zeros, the zero-variance convention). The whitened group advantage
# REPLACES z_out in the trivium's outcome term — above-average trajectories in each group
# get positive weight. Graded games only; mirror self-play stays raw (:No-self-reference:).
RUNG_ELO = {"random": 400.0, "heuristic": 800.0, "sf-skill0": 1000.0, "sf-skill2": 1100.0,
            "sf-skill5": 1200.0, "sf-skill10": 1280.0, "sf-1320": 1320.0}  # declared constants
REPLAY_T = float(os.environ.get("QLEARN_REPLAY_T", "0"))  # Merge 15 :Replay-temperature: —
# admixtures of better batches (operator design): after :GRPO-group: scores the chunk, the
# chunk is RESAMPLED (with replacement, same size) with weights exp(advantage / T): low T ⇒
# train mostly on the elite games (CEM/self-imitation end), high T ⇒ uniform. A stratified
# floor keeps one game of each outcome class (W/D/L) present so draws never fully swamp nor
# vanish (class-balancing instinct). 0 = off. Tuner-searchable (new manifold knob).
STALE_REHEAT = float(os.environ.get("QLEARN_STALE_REHEAT", "0"))    # Merge 11 :Stale-reheat::
# k>0 scales the behavior temperature by (1 + k·stale), capped x4 — the operator's local-
# minimum rule INSIDE a lane: each informative failure (rejected crown / clearly-below epoch)
# raises exploration; a new crown resets stale, so τ cools on progress by construction.
RSEARCH_D  = int(os.environ.get("QLEARN_RSEARCH_DEPTH", "0"))  # Merge 8 (spec/rust-search.spec.md):
# >0 -> generation/measurement moves, TDLeaf leaf targets, and predicted replies come from the
# native FULL-WIDTH alpha-beta + quiescence at this depth (KnightCap-grade targets). Greedy
# behavior (no softmax) — faithful mode's semantics. Requires enc=kc + linear arch.
TRIVIUM    = os.environ.get("QLEARN_TRIVIUM", "")              # "a,b,c" (sum 1) -> compound target
# (S&B §12 averaged backups; KataGo mixed value targets): G = a·λ-return + b·search value at t
# + c·final outcome z. "" = off (pure λ-return). Operator-proposed uniform trivium = "0.34,0.33,0.33".
_TRIV = tuple(float(x) for x in TRIVIUM.split(",")) if TRIVIUM else None
TRIVIUM_END = os.environ.get("QLEARN_TRIVIUM_END", "")         # :Trivium-anneal:: end triple —
# weights slide start->end over training progress via the shared :anneal: shape ("" = static)
_TRIV_END = tuple(float(x) for x in TRIVIUM_END.split(",")) if TRIVIUM_END else None
TRIV_WARMUP = float(os.environ.get("QLEARN_TRIVIUM_WARMUP", "0.4"))
RAMP       = os.environ.get("QLEARN_RAMP", "0") == "1"         # spec :Ramp-filter: (KnightCap td.c,
# RAMP_FACTOR=0): favorable temporal differences on UNPREDICTED opponent moves are zeroed —
# opponent blunders teach nothing; unfavorable surprises always teach. Terminal difference is
# ramped unconditionally (td.c:207). TDLEAF mode only.
KC_FAITHFUL = os.environ.get("QLEARN_KC_FAITHFUL", "0") == "1" # Merge 7 (spec/knightcap-full.
# spec.md :Faithful-mode:): the donor training recipe WHOLE — RAMP on, online per-GAME plain-
# SGD updates (no buffer/batches/Adam), lambda=0.7 fixed, gamma=1, no anneals/adaptation,
# greedy behavior. Homegrown mechanisms are excluded by construction in this mode.
OPP_REACH  = float(os.environ.get("QLEARN_OPP_REACH", "0"))    # spec :Opponent-diet: (intervention-
# queue arm 1): probability a graded generation game is played vs the rung ABOVE the current
# matchmaking rung — guarantees above-current-strength signal in the training distribution
# (KnightCap's rising FICS pool). Reach scores never feed the matchmaking window.
OPP        = os.environ.get("QLEARN_OPP", "self")              # self | graded (spec :Graded-opponents:):
# generation vs an opponent LADDER (random -> heuristic -> SF skill 0/2/5/10 -> SF@1320) with
# matchmaking that holds the score near 50% — KnightCap's actual headline finding (cs/9901002:
# online play vs graded opponents; their self-play TDLeaf variant was substantially weaker).
TDLEAF     = os.environ.get("QLEARN_TDLEAF", "0") == "1"       # Merge 5 (spec/tdleaf.spec.md
# :TDLeaf-mode:): generation plays the SEARCH policy and the stored training pair becomes
# (PV-leaf encoding, λ-return bootstrapped on MINIMAX root values) — TD errors between successive
# search values, gradients at the leaves whose static evals produced them (Baxter et al. 1999).
# build_targets/buffer/anneals/ratchet are untouched; only the (x, gv) pair choose() emits changes.
CONFIRM    = os.environ.get("QLEARN_CONFIRM", "1") == "1"      # spec :Confirmed-crown:: a candidate
# best must SURVIVE an independent re-measure before it is crowned — max over noisy epoch
# samples otherwise ratchets on luck (S&B 015, the outer-loop twin of :Backed-bootstrap:).
ANCHOR     = os.environ.get("QLEARN_ANCHOR", "1")              # gate mode (spec :Anchor-ratchet: v2):
# 0=off | 1=collapse guard (revert only when strength < REVERT_FRAC*best — mild wandering may EXPLORE,
# collapses are caught) | hard=revert every non-improving epoch (stalls under a noisy gate — see D8).
REVERT_FRAC= float(os.environ.get("QLEARN_REVERT_FRAC", "0.5"))
RESUME     = os.environ.get("QLEARN_RESUME", "0") == "1"       # continue from models/qlearn.pt: weights +
# optimizer + CUMULATIVE game count (τ/λ schedules advance on cumulative progress, so a matured net does
# not get re-annealed to MC-heavy/exploratory). Buffer and patience start fresh (buffer targets are
# frozen-at-generation and regenerate from the CURRENT policy anyway). Optuna trials always run fresh.
CKPT       = os.environ.get("QLEARN_CKPT", "models/qlearn.pt")  # checkpoint path; Optuna trials write to
# a SEPARATE file (tune_qlearn passes models/qlearn_optuna.pt) so a study never clobbers the console
# run's resumable model.
FREEZE_EPOCH = os.environ.get("QLEARN_FREEZE_EPOCH", "1") == "1"  # generation + bootstrap from a FROZEN
# copy of the net, synced at epoch boundaries -> each epoch is a true policy-iteration step (data cleanly
# attributable to pi_{k-1}); the LIVE net still takes every SGD step. Off -> generator drifts continuously.
K_ADAPT    = float(os.environ.get("QLEARN_K_ADAPT", "0.5"))     # variance-adaptive λ strength
EPOCH_ELO_GAMES = int(os.environ.get("QLEARN_EPOCH_ELO_GAMES", "0"))  # >0: cheap live Elo vs SF each log step
LOG_EVERY  = int(os.environ.get("QLEARN_LOG_EVERY", "100"))    # emit a plotted metrics point every N games
RESULTS    = "data/qlearn_results.jsonl"
TAG        = os.environ.get("QLEARN_TAG", "")

if KC_FAITHFUL:
    # :Faithful-mode: overrides — the donor recipe admits no anneals, adaptation, or softmax
    # dithering; values cite knightcap.h/local.h. (Env knobs are read above; enforced here.)
    RAMP, TDLEAF = True, True
    LAMBDA, LAMBDA_START, ADAPTIVE_LAMBDA = 0.7, 0.0, False    # TD_LAMBDA 0.7, fixed
    GAMMA = 1.0                                                # no discount in td.c
    TAU_START = TAU_FLOOR = 0.02                               # greedy play (their behavior)


class ValueNet(nn.Module):
    """V(x) = tanh(head(x)) ∈ [-1,1], White-absolute. arch=linear is a learnable piece-square table."""

    def __init__(self, arch="linear", hidden=64, nin=NIN):
        super().__init__()
        if arch == "mlp":
            self.head = nn.Sequential(nn.Linear(nin, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.head = nn.Linear(nin, 1)

    def forward(self, x):
        return torch.tanh(self.head(x)).squeeze(-1)


class QAgent:
    """Raw-board value net + afterstate softmax behavior and greedy target/measurement policies."""

    def __init__(self, dev, arch=ARCH):
        self.dev = dev
        self.net = ValueNet(arch, HIDDEN, NIN_ENC).to(dev)
        self._rs = None                    # native searcher, rebuilt via sync_rsearch()

    def weights_flat(self):
        """(weights_list, bias) of the linear head in RAW feature space — the native searcher's
        constructor diet. Under ZCA the head lives in whitened space; convert back."""
        w = self.net.head.weight.detach().cpu().reshape(-1).double().numpy()
        b = float(self.net.head.bias.detach().cpu().reshape(-1)[0])
        if _ZCA_Z is not None:
            w_raw = _ZCA_Z.astype(np.float64) @ w
            return list(w_raw), b - float(w_raw @ _ZCA_MU.astype(np.float64))
        return list(w), b

    def sync_rsearch(self):
        """(Re)build the native Searcher from the CURRENT net weights (Merge 8). Call after
        every weight load/sync on the net that generates games (init, epoch sync, revert)."""
        if RSEARCH_D <= 0:
            return
        assert ARCH == "linear" and ENC == "kc", "rsearch requires linear arch + kc encoding"
        import importlib
        Searcher = importlib.import_module(RSEARCH_MOD).Searcher
        w, b = self.weights_flat()
        self._rs = Searcher(w, b)

    @torch.no_grad()
    def afterstate_values(self, board):
        """(moves, encodings[M,nin], white-absolute values[M]) — one batched forward over legal afterstates."""
        moves = list(board.legal_moves)
        X = np.empty((len(moves), NIN_ENC), dtype=np.float32)
        for i, mv in enumerate(moves):
            board.push(mv)
            X[i] = ENC_FN(board)
            board.pop()
        xt = torch.from_numpy(X).to(self.dev)
        v = self.net(xt).detach().cpu().numpy().reshape(-1)
        return moves, X, v

    @torch.no_grad()
    def value_fn(self, X):
        """Batched evaluator (n,769)->(n,) for :Search-policy:."""
        return self.net(torch.from_numpy(X).to(self.dev)).detach().cpu().numpy().reshape(-1)

    def choose(self, board, tau, rng):
        """Behavior move + the position's GREEDY (off-policy max) value + chosen afterstate encoding.
        BEHAVIOR=search picks the move by depth-2 root softmax; bootstrap gv stays 1-ply greedy.
        TDLEAF (spec :TDLeaf-mode:) returns the PV-LEAF encoding and the MINIMAX greedy value —
        the whole TDLeaf(λ) delta lives in this one substitution."""
        if TDLEAF and RSEARCH_D > 0 and self._rs is not None:
            # Merge 8: native full-width targets — value is BACKED by construction (no beam),
            # leaf is the quiescence-resolved PV leaf, pred is the true best reply
            mv_u, val, leaf_fen, pred_u, _n = self._rs.search(board.fen(), RSEARCH_D)
            leaf_enc = ENC_FN(chess.Board(leaf_fen))
            pred = chess.Move.from_uci(pred_u) if pred_u else None
            return chess.Move.from_uci(mv_u), leaf_enc, float(val), pred
        if TDLEAF:
            return search_move(board, self.value_fn, max(tau, 1e-6), rng, SEARCH_W,
                               depth=SEARCH_D, return_leaf=True, encode_fn=ENC_FN)
        moves, X, v = self.afterstate_values(board)
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        greedy_v = float(v.max() if sign > 0 else v.min())
        if BEHAVIOR == "search":
            mv = search_move(board, self.value_fn, max(tau, 1e-6), rng, SEARCH_W, depth=SEARCH_D,
                             encode_fn=ENC_FN)
            return mv, X[moves.index(mv)], greedy_v
        logits = sign * v / max(tau, 1e-6)
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        i = rng.choices(range(len(moves)), weights=p, k=1)[0]
        return moves[i], X[i], greedy_v

    def greedy_move(self, board):
        """Exploit policy — proxy strength + the Elo objective. Search argmax under
        BEHAVIOR=search or TDLEAF (measurement consistent with the trained-for policy)."""
        if TDLEAF and RSEARCH_D > 0 and self._rs is not None:
            return chess.Move.from_uci(self._rs.search(board.fen(), RSEARCH_D)[0])
        if TDLEAF or BEHAVIOR == "search":
            return search_move(board, self.value_fn, 0.0, None, SEARCH_W, depth=SEARCH_D,
                               encode_fn=ENC_FN)
        moves, _, v = self.afterstate_values(board)
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        return moves[int(v.argmax() if sign > 0 else v.argmin())]


def anneal(start, end, progress, w):
    """Asymptotic schedule SHARED by τ and λ — one intent, one shape: exponential decay from start toward
    end with e-folding fraction w (the single tunable scalar, the "warmup ratio"). ~63% of the anneal is
    done at progress=w, ~92% at 2.5w; the rate of change decays and the value NEVER reaches end — τ keeps
    exploration room forever, λ never fully surrenders the MC signal. progress ∈ [0,1] over total games."""
    return end + (start - end) * math.exp(-progress / max(w, 1e-6))


def tau_at(games_played, total_games, cum=0):
    """Temperature via the shared :anneal: — τ decays τ_start -> τ_floor with e-folding fraction WARMUP,
    approaching the floor asymptotically (never touching it). `cum` = prior games from a resumed
    checkpoint, so a matured net keeps its late-schedule τ instead of re-exploring from τ_start."""
    return anneal(TAU_START, TAU_FLOOR, (cum + games_played) / max(1, cum + total_games), WARMUP)


class OpponentLadder:
    """spec :Graded-opponents: — generation opponents, weak->strong, with :Matchmaking: that holds
    the agent's score near 50% (KnightCap's FICS effect). Rungs: random -> heuristic -> SF Skill
    Level 0/2/5/10 -> SF@1320; SF rungs only if an engine is found (never crashes generation).
    Failure mode: an SF that dies mid-run raises through move_fn — the run fails loudly, by design."""

    WINDOW, UP, DOWN = 20, 0.6, 0.4

    def __init__(self):
        self.rungs = [("random", random_mover), ("heuristic", heuristic_mover)]
        self.sf, self._cfg = None, None
        import glob
        import chess.engine
        sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
        if sfp:
            try:                                   # never hang generation on a bad UCI handshake
                self.sf = chess.engine.SimpleEngine.popen_uci(sfp[0], timeout=20.0)
                self.sf_lim = chess.engine.Limit(time=0.05)
                for lvl in (0, 2, 5, 10):
                    self.rungs.append((f"sf-skill{lvl}",
                                       self._sf_mover({"UCI_LimitStrength": False, "Skill Level": lvl})))
                self.rungs.append(("sf-1320", self._sf_mover({"UCI_LimitStrength": True, "UCI_Elo": 1320})))
            except Exception as e:
                print(f"stockfish unavailable ({e}) -> ladder truncated at heuristic", flush=True)
                self.sf = None
        self.i = 1                                 # start rung: heuristic
        self.window = deque(maxlen=self.WINDOW)

    def _sf_mover(self, cfg):
        def mv(board):
            if self._cfg != cfg:
                self.sf.configure(cfg)
                self._cfg = cfg
            return self.sf.play(board, self.sf_lim).move
        return mv

    def move_fn(self):
        return self.rungs[self.i][1]

    def name(self):
        return self.rungs[self.i][0]

    def report(self, score):
        """Feed one game's agent score (1/0.5/0); move a rung when the FULL window says mismatch."""
        self.window.append(score)
        if len(self.window) == self.WINDOW:
            m = sum(self.window) / self.WINDOW
            if m > self.UP and self.i < len(self.rungs) - 1:
                self.i += 1
                self.window.clear()
                print(f"ladder UP -> {self.name()}", flush=True)
            elif m < self.DOWN and self.i > 0:
                self.i -= 1
                self.window.clear()
                print(f"ladder DOWN -> {self.name()}", flush=True)

    def close(self):
        if self.sf is not None:
            self.sf.quit()


def play_game(agent, tau, rng, env, opp_move=None, agent_white=True):
    """One generation game. Return (xs, greedy_vals, z, predicted, signs): per-decision
    afterstate encodings, their greedy values, the White-absolute terminal outcome, and — for
    the :Ramp-filter: — predicted[t] (was the move leading INTO decision t the one the search
    at t-1 expected?) and signs[t] (+1 White decider, -1 Black). Self-play when opp_move is
    None (both sides = agent; the "reply" to decision t-1 is decision t's own move); graded
    (spec :Graded-opponents:) otherwise — xs/gvs collect ONLY the agent's decisions and the
    reply is the opponent's first move after the agent's."""
    board = env.reset()
    xs, gvs, predicted, signs = [], [], [], []
    pending_pred, reply_actual = None, None
    z, done = 0.0, False
    while not done and env.plies < PLY_CAP:
        if opp_move is None or (board.turn == chess.WHITE) == agent_white:
            signs.append(1.0 if board.turn == chess.WHITE else -1.0)
            if TDLEAF:
                mv, x_after, gv, pr = agent.choose(board, tau, rng)
            else:
                mv, x_after, gv = agent.choose(board, tau, rng)
                pr = None
            actual_reply = mv if opp_move is None else reply_actual
            predicted.append(pending_pred is not None and actual_reply == pending_pred)
            xs.append(x_after)      # the afterstate we train V on
            gvs.append(gv)          # greedy value available at THIS position (bootstrap source)
            pending_pred, reply_actual = pr, None
        else:
            mv = opp_move(board)
            if reply_actual is None:
                reply_actual = mv
        board, z, done = env.step(mv)
    return xs, gvs, z, predicted, signs


def build_targets(xs, gvs, z, lam, predicted=None, signs=None, triv=None, z_out=None):
    """λ-return targets for each visited afterstate. Reward is 0 until the terminal (White-absolute z);
    the bootstrap for step k is the GREEDY value at position k+1 (off-policy max), 0 at the terminal step.
    `lam` is the (possibly variance-adapted) eligibility-trace λ for this epoch.
    RAMP (spec :Ramp-filter:, KnightCap td.c): targets are rebuilt from per-step residuals
    δ_k = γ·gv_{k+1} − gv_k (terminal: z − gv_T); a residual FAVORABLE to the decider
    (sign-adjusted) is zeroed when the transition into k+1 was unpredicted — and the terminal
    residual is ramped unconditionally (td.c:207). With no filtering this reduction is
    algebraically the standard λ-return."""
    T = len(xs)
    if T == 0:
        return [], []
    if triv is None:
        triv = _TRIV
    zc = z if z_out is None else z_out       # :Rating-surprise:: outcome COMPONENT only —
    # the λ-return's terminal reward stays raw z (game truth); the trivium's c-term anchors
    # on the peer-relative surprise when provided.
    if not (RAMP and TDLEAF) or predicted is None:
        rewards = [0.0] * T
        rewards[-1] = z
        dones = [False] * T
        dones[-1] = True
        boot = [gvs[k + 1] if k + 1 < T else 0.0 for k in range(T)]
        G = lambda_return(rewards, boot, dones, lam, gamma=GAMMA)
        if triv is not None:
            a, b, c = triv
            G = [a * G[k] + b * gvs[k] + c * zc for k in range(T)]
        return xs, G
    deltas = [0.0] * T
    for k in range(T):
        terminal = (k + 1 == T)
        d = (z - gvs[k]) if terminal else (GAMMA * gvs[k + 1] - gvs[k])
        ramped = terminal or not predicted[k + 1]
        if ramped and d * signs[k] > 0:                        # favorable surprise -> no credit
            d = 0.0
        deltas[k] = d
    G, acc = [0.0] * T, 0.0
    for k in range(T - 1, -1, -1):                             # G_k = gv_k + Σ (γλ)^{j-k} δ_j
        acc = deltas[k] + GAMMA * lam * acc
        G[k] = gvs[k] + acc
    if triv is not None:                                       # compound target (operator trivium):
        a, b, c = triv                                         # λ-return : search value : outcome
        G = [a * G[k] + b * gvs[k] + c * zc for k in range(T)]
    return xs, G


_PVAL = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)   # P N B R Q K(=0)


def material_points(x):
    """White-absolute material in pawns from the 769-dim encoding: Σ value·(white_count − black_count)."""
    m = 0.0
    for pt in range(6):
        m += _PVAL[pt] * (x[pt * 64:(pt + 1) * 64].sum() - x[(pt + 6) * 64:(pt + 7) * 64].sum())
    return float(m)


def adapt_lambda(lam_base, td_sigma):
    """Variance-adaptive λ (spec :Variance-adaptive-λ:). Noisy TD-targets (high sdev) → trust the bootstrap
    MORE (lower λ). Fed the PREVIOUS epoch's TD-error sdev to avoid circularity. Off → returns λ_base."""
    if not ADAPTIVE_LAMBDA:
        return lam_base
    sig = min(1.0, max(0.0, td_sigma))          # TD-errors on tanh values ∈ [-2,2]; sdev typically ≤ ~1
    return float(np.clip(lam_base * (1.0 - K_ADAPT * sig), 0.1, lam_base))


def evaluate_greedy(agent, games):
    """GREEDY (optimal) policy diagnostics — EVERY score shown in the UI comes from THIS policy, not from the
    exploratory softmax self-play. Returns (wr_random, wr_heuristic, avg_material, avg_turns): greedy win-rates
    vs a random mover and vs the 1-ply PST heuristic, plus the agent's average final material margin (nominal
    pawns, agent − opponent) and average game length (full moves) in the vs-heuristic games."""
    def run(opp, track):
        s, tot_mat, tot_plies = 0.0, 0.0, 0
        for g in range(games):
            aw = (g % 2 == 0)
            b = chess.Board(); plies = 0
            while not b.is_game_over() and plies < PLY_CAP:
                b.push(agent.greedy_move(b) if (b.turn == chess.WHITE) == aw else opp(b))
                plies += 1
            if b.is_checkmate():
                s += 1.0 if ((b.turn == chess.BLACK) == aw) else 0.0
            else:
                s += 0.5
            if track:
                mat = material_points(encode(b))
                tot_mat += mat if aw else -mat
                tot_plies += plies
        return (s / games, tot_mat / games, tot_plies / (2 * games)) if track else s / games
    wr_rand = run(random_mover, False)
    wr_heur, avg_mat, avg_turns = run(heuristic_mover, True)
    return wr_rand, wr_heur, avg_mat, avg_turns


def greedy_elo(agent, sf, sf_lim, n):
    """Anchored Elo + NOMINAL score of the GREEDY policy vs SF@1320 over n games (cheap live measure).
    Returns (elo, pts): elo clamps at the -280 floor whenever pts == 0, so pts (W + D/2, raw points out
    of n) is the honest low-end signal — plot THAT, the Elo is derived."""
    sW, sD, _sL, _ = elo_play(agent.greedy_move, lambda b: sf.play(b, sf_lim).move, n)
    pts = sW + 0.5 * sD
    return round(1320 + elo_diff(pts / n, n)), pts


def learned_piece_worth(agent):
    """Learned piece values read straight off the LINEAR net's weights: mean weight over each piece's 64
    white squares minus its 64 black squares, normalized to pawn=1. This is the empirical answer to "do
    pieces build up their own worth from checkmate-only reward?" (Beal & Smith 1997: yes, TD self-play
    recovers ~1/3/3/5/9). Returns None for non-linear archs (weights not per-feature interpretable)."""
    if ARCH != "linear":
        return None
    w = agent.net.head.weight.detach().cpu().numpy().reshape(-1)
    raw = [float(w[t * 64:(t + 1) * 64].mean() - w[(t + 6) * 64:(t + 7) * 64].mean()) for t in range(5)]
    pawn = raw[0]
    if abs(pawn) < 1e-8:               # pawn weight still ~0 -> ratios meaningless this early
        return {"pawn_raw": round(pawn, 6), "N": None, "B": None, "R": None, "Q": None}
    return {"pawn_raw": round(pawn, 6),
            "N": round(raw[1] / pawn, 2), "B": round(raw[2] / pawn, 2),
            "R": round(raw[3] / pawn, 2), "Q": round(raw[4] / pawn, 2)}


def main():
    epoch_games = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCH_GAMES
    max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_EPOCHS
    total_games = epoch_games * max_epochs                 # τ warmup denominator (max budget)
    log_every = max(1, min(LOG_EVERY, epoch_games))        # a plotted point every this many games (dense)
    batch_games = max(1, min(BATCH_GAMES, epoch_games))    # games per generate->train cycle (x/y batches)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    dev = torch.device(os.environ.get("QLEARN_DEV", "cuda" if torch.cuda.is_available() else "cpu"))

    agent = QAgent(dev)
    # :Faithful-mode:: plain SGD, one step per game (td.c uses bare TD_ALPHA·dw); otherwise Adam
    opt = (torch.optim.SGD(agent.net.parameters(), lr=ALPHA) if KC_FAITHFUL
           else torch.optim.Adam(agent.net.parameters(), lr=ALPHA))
    cum_games = 0                          # prior training absorbed by the checkpoint (drives schedules)
    resume_bar = -1.0
    agent_elo_r = 600.0                    # :Rating-surprise: online agent rating (anchored gauge)
    # resume from the BEST-visited checkpoint by default (QLEARN_RESUME_BEST=0 for latest): chained
    # runs then RATCHET — each leg starts from the best policy any leg ever reached, so a late collapse
    # in one leg cannot poison the next.
    ckpt_path = CKPT.replace(".pt", "_best.pt") \
        if (os.environ.get("QLEARN_RESUME_BEST", "1") == "1"
            and os.path.exists(CKPT.replace(".pt", "_best.pt"))) else CKPT
    if RESUME and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        if (ck.get("arch", "linear") == ARCH and ck.get("enc", "pst") == ENC
                and bool(ck.get("zca", False)) == bool(ZCA)):
            try:
                agent.net.load_state_dict(ck["state_dict"])
            except RuntimeError as e:                  # hidden-width / shape mismatch -> fresh net
                print(f"resume shape mismatch ({e}) -> fresh net", flush=True)
                ck = {}
            if "opt_state" in ck:
                try:
                    opt.load_state_dict(ck["opt_state"])
                except Exception:
                    pass                   # optimizer format drift -> fresh Adam moments, weights kept
            cum_games = int(ck.get("cum_games", 0))
            resume_bar = float(ck.get("strength", -1.0))   # persist the acceptance bar across legs
            agent_elo_r = float(ck.get("rating", 600.0))   # :Rating-surprise: rating rides the ckpt
            print(f"RESUMED {ckpt_path} (+{cum_games} prior games, bar {resume_bar:.2f})", flush=True)
        else:
            print(f"resume requested but arch mismatch ({ck.get('arch')} != {ARCH}) -> fresh net", flush=True)
    # FROZEN GENERATOR (policy iteration): gen plays the games and supplies the bootstrap values, synced
    # from the live net only at epoch boundaries -> epoch k's data is cleanly "games from pi_{k-1} at tau_k".
    gen = agent
    if FREEZE_EPOCH:
        gen = QAgent(dev)
        gen.net.load_state_dict(agent.net.state_dict())
    gen.sync_rsearch()                     # Merge 8: native searcher tracks the generator
    if not RESUME and os.path.exists(CKPT.replace(".pt", "_best.pt")):
        os.remove(CKPT.replace(".pt", "_best.pt"))     # fresh run = new lineage; never gate vs an old one
    buf_cap = int(epoch_games * 64 * BUFFER_EPOCHS) if BUFFER_EPOCHS > 0 else BUFFER
    buf = deque(maxlen=buf_cap)
    env = ChessEnv(ply_cap=PLY_CAP)
    ladder = OpponentLadder() if OPP == "graded" else None     # spec :Graded-opponents:
    os.makedirs("data", exist_ok=True)
    open(METRICS, "w").close()   # fresh training curve per run

    # optional persistent Stockfish for a cheap LIVE per-epoch Elo (opened once, reused)
    sf, sf_lim = None, None
    if EPOCH_ELO_GAMES > 0:
        import glob
        sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
        if sfp:
            try:                                   # never hang a run on an unresponsive UCI handshake
                sf = chess.engine.SimpleEngine.popen_uci(sfp[0], timeout=20.0)
                sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
                sf_lim = chess.engine.Limit(time=0.05)
            except Exception as e:
                print(f"stockfish unavailable ({e}) -> live Elo off this run", flush=True)
                sf, sf_lim = None, None

    print(f"Merge 2 Q-learning | <={max_epochs} epochs x {epoch_games} games | patience {PATIENCE} | "
          f"arch={ARCH} enc={ENC}({NIN_ENC}) a={ALPHA} g={GAMMA} lam={max(LAMBDA_START, LAMBDA)}->{LAMBDA}{'~adapt' if ADAPTIVE_LAMBDA else ''} "
          f"warmup={WARMUP} buffer={buf_cap}{f' (~{BUFFER_EPOCHS:g} epochs)' if BUFFER_EPOCHS > 0 else ''} "
          f"batch={batch_games}g x{epoch_games // batch_games}/epoch (sgd {BATCH}pos) "
          f"behavior={'TDLEAF' if TDLEAF else BEHAVIOR}"
          f"{f'(d{SEARCH_D} w{SEARCH_W})' if (TDLEAF or BEHAVIOR == 'search') else ''} "
          f"opp={OPP}{f'+reach{OPP_REACH:g}' if OPP_REACH > 0 else ''} "
          f"{'KC-FAITHFUL(ramp,online-sgd) ' if KC_FAITHFUL else ('ramp ' if RAMP else '')}"
          f"{f'RSEARCH-d{RSEARCH_D} ' if RSEARCH_D > 0 else ''}"
          f"gen={'frozen/epoch' if FREEZE_EPOCH else 'live'} dev={dev}\n", flush=True)

    games_played = 0
    lam_eff = max(LAMBDA, LAMBDA_START)    # annealed base λ; adapts each batch to the prior batch's TD variance
    ep_sf_pts, ep_sf_n = 0.0, 0            # epoch-pooled nominal SF score -> the Elo-based patience signal
    run_sf_pts, run_sf_n = 0.0, 0          # RUN-pooled SF score -> the high-resolution Optuna objective
    best_strength, stale, epoch, ep_games = resume_bar, 0, 0, 0
    wr_rand = wr_heur = 0.0
    zs_log, losses_log, since_log = [], [], 0
    epoch_strength = None               # accumulate between plotted samples
    t = time.time()
    while games_played < total_games:
        chunk = min(batch_games, total_games - games_played, epoch_games - ep_games)
        tau = tau_at(games_played, total_games, cum_games)     # metrics row + serial behavior
        if STALE_REHEAT > 0:
            tau *= min(1.0 + STALE_REHEAT * stale, 4.0)        # :Stale-reheat:
        new_pairs = []
        game_data = []                          # per game: (xs, gvs, z, predicted, signs)
        if PARGEN > 0:
            # Merge 9: native parallel batch — greedy+ε behavior, exact-min targets, TDLeaf
            # records; opponent = frozen self at PARGEN_OPP_D (0 = random)
            import importlib
            _pg = importlib.import_module(RSEARCH_MOD).play_games
            gw, gb = gen.weights_flat()
            batch = _pg(gw, gb, gw, gb, PARGEN_OPP_D, chunk, PARGEN_THREADS,
                        max(RSEARCH_D, 2), PARGEN_EPS, PLY_CAP,
                        (SEED + 1) * 1_000_003 + cum_games + games_played,
                        tau if PARGEN_SOFTMAX else 0.0)   # :Softmax-tau:: trainer's annealed τ
            for z, aw, recs in batch:
                game_data.append(([ENC_FN(chess.Board(f)) for f, _v, _p in recs],
                                  [v for _f, v, _p in recs], z,
                                  [p for _f, _v, p in recs],
                                  [1.0 if aw else -1.0] * len(recs), None))
        else:
            for _ in range(chunk):
                tau = tau_at(games_played, total_games, cum_games)
                if STALE_REHEAT > 0:
                    tau *= min(1.0 + STALE_REHEAT * stale, 4.0)  # :Stale-reheat:
                if ladder is not None:
                    aw = (games_played % 2 == 0)               # alternate colors vs the ladder
                    reach = (OPP_REACH > 0 and ladder.i < len(ladder.rungs) - 1
                             and rng.random() < OPP_REACH)     # spec :Opponent-diet:
                    rung_i = ladder.i + 1 if reach else ladder.i
                    opp_fn = ladder.rungs[rung_i][1]
                    xs, gvs, z, pred_f, sgn = play_game(gen, tau, rng, env, opp_fn, aw)
                    s = 0.5 if z == 0.0 else (1.0 if (z > 0) == aw else 0.0)
                    if not reach:                              # window = rung-MATCHED games only
                        ladder.report(s)
                    z_out = None
                    if SURPRISE:
                        # :Rating-surprise:: e = BT expectation vs the DECLARED rung rating
                        # (:Anchor-pinned: gauge); agent rating updates online; the outcome
                        # component becomes the clipped surprise, White-absolute.
                        r_opp = RUNG_ELO.get(ladder.rungs[rung_i][0], 800.0)
                        e_exp = 1.0 / (1.0 + 10.0 ** ((r_opp - agent_elo_r) / 400.0))
                        agent_elo_r += SURPRISE_K * (s - e_exp)
                        z_out = max(-1.0, min(1.0, 2.0 * (s - e_exp))) * (1.0 if aw else -1.0)
                else:
                    xs, gvs, z, pred_f, sgn = play_game(gen, tau, rng, env)
                    z_out = None                               # :No-self-reference: mirror keeps raw z
                game_data.append((xs, gvs, z, pred_f, sgn, z_out))
        if GRPO:
            # :GRPO-group: over this chunk's graded games — whiten, clip, nothing fancier
            idx = [i for i, g in enumerate(game_data) if g[5] is not None and g[4]]
            if idx:
                sc = [game_data[i][5] * game_data[i][4][0] for i in idx]   # agent-relative
                gm = sum(sc) / len(sc)
                gstd = (sum((x - gm) ** 2 for x in sc) / len(sc)) ** 0.5
                gz = [0.0] * len(sc) if gstd < 1e-8 else [(x - gm) / gstd for x in sc]
                for j, i in enumerate(idx):
                    g = game_data[i]
                    game_data[i] = (g[0], g[1], g[2], g[3], g[4],
                                    max(-1.0, min(1.0, gz[j])) * g[4][0])  # back to White-absolute
        if REPLAY_T > 0 and game_data:
            # :Replay-temperature: — resample the chunk toward its above-average games
            def _adv(g):
                return (g[5] * g[4][0]) if (g[5] is not None and g[4]) else 0.0
            def _cls(g):                                       # outcome class for the floor
                return 0 if g[2] == 0.0 else (1 if g[2] > 0 else -1)
            ws = [math.exp(min(10.0, _adv(g) / REPLAY_T)) for g in game_data]
            tot_w = sum(ws)
            picks = [game_data[rng.choices(range(len(game_data)), weights=ws)[0]]
                     for _ in range(len(game_data))] if tot_w > 0 else list(game_data)
            for c in {_cls(g) for g in game_data}:             # stratified floor: keep every class
                if not any(_cls(g) == c for g in picks):
                    picks[rng.randrange(len(picks))] = next(g for g in game_data if _cls(g) == c)
            game_data = picks
        triv_now = _TRIV
        if _TRIV is not None and _TRIV_END is not None:        # :Trivium-anneal:
            _f = math.exp(-((cum_games + games_played) / max(1, cum_games + total_games))
                          / max(TRIV_WARMUP, 1e-6))
            triv_now = tuple(e + (s - e) * _f for s, e in zip(_TRIV, _TRIV_END))
        for xs, gvs, z, pred_f, sgn, z_out in game_data:
            zs_log.append(z)
            xt, G = build_targets(xs, gvs, z, lam_eff, pred_f, sgn, triv_now, z_out)
            for x, g in zip(xt, G):
                new_pairs.append((x, g))
                if not KC_FAITHFUL:
                    buf.append((x, g))
            if KC_FAITHFUL and xt:
                # :Faithful-mode: online per-GAME update (td.c td_update): one plain-SGD step
                # on this game's filtered targets. MSE-mean grad = their per-move d·(1-tanh²)·x
                # accumulation up to the 1/T factor absorbed by α.
                Xg = torch.from_numpy(np.stack(xt)).to(dev)
                yg = torch.tensor(G, dtype=torch.float32, device=dev)
                pg = agent.net(Xg)
                loss_g = torch.mean((pg - yg) ** 2)
                opt.zero_grad()
                loss_g.backward()
                opt.step()
                losses_log.append(loss_g.item())
            games_played += 1
            ep_games += 1

        # TD-error sdev on this batch's fresh pairs (pre-train) -> next batch's variance-adaptive λ
        td_sigma = 0.0
        if new_pairs:
            with torch.no_grad():
                Ve = agent.net(torch.from_numpy(np.stack([p[0] for p in new_pairs])).to(dev)).cpu().numpy().reshape(-1)
            td_sigma = float(np.std(np.array([p[1] for p in new_pairs], dtype=np.float32) - Ve))

        steps = max(1, round(TRAIN_STEPS * chunk / epoch_games))   # gradient steps proportional to the batch
        if not KC_FAITHFUL and len(buf) >= BATCH:
            for _ in range(steps):
                batch = random.sample(buf, BATCH)
                X = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
                y = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=dev)
                pred = agent.net(X)
                loss = torch.mean((pred - y) ** 2)
                opt.zero_grad(); loss.backward(); opt.step()
                losses_log.append(loss.item())

        # λ schedule (shared :anneal: shape with τ): MC-heavy early (V untrained -> trust real game
        # outcomes), decaying asymptotically toward the tuned base (never reaching it); the variance-
        # adaptive trim stacks on top as the opposite-direction Watkins force.
        lam_base_t = anneal(max(LAMBDA_START, LAMBDA), LAMBDA,
                            (cum_games + games_played) / max(1, cum_games + total_games), LAMBDA_WARMUP)
        lam_eff = adapt_lambda(lam_base_t, td_sigma)
        since_log += chunk

        # one metrics row PER BATCH (cheap fields: loss, τ, λ, σ, reward, decisive, piece worth) so the
        # dashboard can show per-batch changes + a moving average. The EXPENSIVE greedy/SF evals run only
        # on the sample cadence — and ALWAYS at an epoch boundary / end of training, so the patience
        # signal below never reads a stale eval. Eval-less rows carry None in the eval fields.
        do_eval = since_log >= log_every or games_played >= total_games or ep_games >= epoch_games
        if do_eval:
            agent.sync_rsearch()           # greedy evals run on the LIVE net's searcher
            # ALL displayed scores come from the GREEDY (optimal) policy — consistent with the Elo
            wr_rand, wr_heur, avg_mat, avg_turns = evaluate_greedy(agent, PROXY_GAMES)
            epoch_elo, sf_pts = greedy_elo(agent, sf, sf_lim, EPOCH_ELO_GAMES) if sf is not None else (None, None)
            if sf_pts is not None:
                ep_sf_pts += sf_pts; ep_sf_n += EPOCH_ELO_GAMES  # pool the epoch's SF games for patience
                run_sf_pts += sf_pts; run_sf_n += EPOCH_ELO_GAMES
            since_log = 0
            if ep_games >= epoch_games:            # boundary sample -> THE goal curve (Elo over epochs)
                epoch_strength = round(100.0 * ((ep_sf_pts / ep_sf_n) if ep_sf_n else 0.0)
                                       + (wr_rand + wr_heur), 3)
        row = {"games_played": games_played, "epoch": epoch + 1, "tau": round(tau, 4),
               "lam_eff": round(lam_eff, 4), "td_sigma": round(td_sigma, 4),
               "loss": round(float(np.mean(losses_log)) if losses_log else 0.0, 5), "buf": len(buf),
               "avg_points": round(avg_mat, 2) if do_eval else None,
               "avg_turns": round(avg_turns, 1) if do_eval else None,
               "epoch_elo": epoch_elo if do_eval else None,
               "sf_pts": sf_pts if do_eval else None,
               "sf_n": EPOCH_ELO_GAMES if (do_eval and sf is not None) else None,
               "piece_vals": learned_piece_worth(agent),
               "avg_reward": round(float(np.mean(zs_log)), 3) if zs_log else None,
               "decisive": round(float(np.mean(np.abs(zs_log))), 3) if zs_log else None,
               "epoch_strength": epoch_strength if (do_eval and ep_games >= epoch_games) else None,
               "wr_vs_random": round(wr_rand, 3) if do_eval else None,
               "wr_vs_heuristic": round(wr_heur, 3) if do_eval else None,
               "opp_rung": ladder.name() if ladder is not None else None,
               "ts": int(time.time())}
        with open(METRICS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if do_eval:
            print(f"g {games_played} | ep {epoch+1} | lam {lam_eff:.3f} | td_sig {td_sigma:.3f} | "
                  f"loss {row['loss']:.4f} | pts {row['avg_points']} turns {row['avg_turns']} | "
                  f"sf {sf_pts if sf_pts is not None else '-'}/{EPOCH_ELO_GAMES} (elo {epoch_elo if epoch_elo is not None else '-'}) | vs_heur {wr_heur:.2f} | "
                  f"{time.time()-t:.0f}s", flush=True)
            t = time.time()
        zs_log, losses_log = [], []            # per-batch accumulators reset every row

        if ep_games >= epoch_games:                 # epoch boundary -> patience on the ELO signal
            epoch += 1; ep_games = 0
            if FREEZE_EPOCH:                        # policy-iteration step: pi_k <- live net
                gen.net.load_state_dict(agent.net.state_dict())
            gen.sync_rsearch()
            os.makedirs("models", exist_ok=True)    # per-epoch checkpoint -> a killed run resumes from here
            torch.save({"state_dict": agent.net.state_dict(), "arch": ARCH, "enc": ENC, "zca": bool(ZCA),
                        "opt_state": opt.state_dict(), "cum_games": cum_games + games_played,
                        "ts": int(time.time())}, CKPT)
            # Patience tracks the SAME objective Optuna maximizes: this epoch's aggregated score vs
            # SF@1320 (ALL the epoch's live-Elo games pooled -> steadier than any single 12-game sample),
            # scaled to DOMINATE the proxy term. The proxy (wr_rand+wr_heur) breaks ties while every
            # epoch is still a shutout (score 0), and is the sole signal when EPOCH_ELO_GAMES=0.
            ep_sf_score = (ep_sf_pts / ep_sf_n) if ep_sf_n else 0.0
            strength = 100.0 * ep_sf_score + (wr_rand + wr_heur)
            print(f"epoch {epoch} strength: sf {ep_sf_pts}/{ep_sf_n} + proxy {wr_rand + wr_heur:.2f} "
                  f"-> {strength:.2f} (best {best_strength:.2f})", flush=True)
            challenged = False
            if CONFIRM and sf is not None and strength > best_strength + 1e-3:
                challenged = True
                # :Confirmed-crown:: candidate best -> independent re-measure, pooled decision.
                _c_elo, c_pts = greedy_elo(agent, sf, sf_lim, EPOCH_ELO_GAMES)
                c_rand, c_heur, _cm, _ct = evaluate_greedy(agent, PROXY_GAMES)
                confirmed = (100.0 * (ep_sf_pts + c_pts) / (ep_sf_n + EPOCH_ELO_GAMES)
                             + ((wr_rand + wr_heur) + (c_rand + c_heur)) / 2.0)
                run_sf_pts += c_pts; run_sf_n += EPOCH_ELO_GAMES   # confirmation joins the run pool
                print(f"crown check: epoch {strength:.2f} -> confirmed {confirmed:.2f} "
                      f"({'kept' if confirmed > best_strength + 1e-3 else 'rejected'})", flush=True)
                crown_pool = (ep_sf_pts + c_pts, ep_sf_n + EPOCH_ELO_GAMES)  # :Crown-rung: pooled SF games
                strength = confirmed
            ep_sf_pts, ep_sf_n = 0.0, 0
            if strength > best_strength + 1e-3:
                best_strength, stale = strength, 0
                # KEEP-BEST: RL curves are non-monotone (policy->data feedback); keep the best VISITED
                torch.save({"state_dict": agent.net.state_dict(), "arch": ARCH, "enc": ENC, "zca": bool(ZCA),
                            "opt_state": opt.state_dict(), "cum_games": cum_games + games_played,
                            "strength": strength, "rating": agent_elo_r, "ts": int(time.time())},
                           CKPT.replace(".pt", "_best.pt"))
                if challenged:
                    # :Crown-rung:: every KEPT crown feeds the ladder from its pooled SF games —
                    # zero extra compute; the row is the crown's own evidence (d2-greedy scale).
                    _pts, _n = crown_pool
                    _s = _pts / _n
                    _se = math.sqrt(max(_s * (1 - _s), 1e-9) / _n)
                    _row = {"merge": 9, "agent": (f"{TAG} " if TAG else "") +
                            f"crown {strength:.2f} d2-greedy (live run)",
                            "ts": int(time.time()), "games": _n,
                            "vs_sf_W": None, "vs_sf_D": None, "vs_sf_L": None,
                            "vs_sf_score": round(_s, 4),
                            "elo": round(1320 + elo_diff(_s, _n)),
                            "elo_lo": round(1320 + elo_diff(_s - 1.96 * _se, _n)),
                            "elo_hi": round(1320 + elo_diff(_s + 1.96 * _se, _n))}
                    with open("data/rl_trend.jsonl", "a") as _fh:
                        _fh.write(json.dumps(_row) + "\n")
            else:
                # Horizon fix (efficiency plan step 2): only INFORMATIVE failures age the
                # run — a REJECTED crown (we challenged and lost) or a clearly-below epoch
                # (< 0.75x bar). Marginal below-bar epochs are ±5-noise and taught nothing;
                # counting them killed every arm at <1,200 games while KnightCap ran 1000s.
                if challenged or strength < 0.75 * best_strength:
                    stale += 1
                best_p = CKPT.replace(".pt", "_best.pt")
                revert = (ANCHOR == "hard") or (ANCHOR == "1" and strength < REVERT_FRAC * best_strength)
                if revert and os.path.exists(best_p):     # collapse (or hard gate) -> revert to anchor
                    ckb = torch.load(best_p, map_location=dev)
                    agent.net.load_state_dict(ckb["state_dict"])
                    if "opt_state" in ckb:
                        opt.load_state_dict(ckb["opt_state"])
                    if FREEZE_EPOCH:
                        gen.net.load_state_dict(agent.net.state_dict())
                    agent.sync_rsearch()
                    gen.sync_rsearch()
                    print(f"anchor: epoch {epoch} REVERTED ({strength:.2f} vs best {best_strength:.2f})",
                          flush=True)
                if stale >= PATIENCE:
                    print(f"early stop: no improvement for {PATIENCE} epochs", flush=True)
                    break

    if sf is not None:
        sf.quit()
    if ladder is not None:
        ladder.close()
    epochs_run = round(games_played / epoch_games, 2)

    # The run's OUTPUT is the best-visited policy, always (keep-best selection, not last-weights luck)
    best_p = CKPT.replace(".pt", "_best.pt")
    if os.path.exists(best_p):
        ckb = torch.load(best_p, map_location=dev)
        agent.net.load_state_dict(ckb["state_dict"])
        agent.sync_rsearch()
        print(f"final measure on BEST-visited (strength {ckb.get('strength')})", flush=True)

    # --- Elo objective (the ONLY strength signal Optuna maximizes) ---
    # ELO_GAMES=0 (population lanes, Merge 11): skip the final measure entirely — the lane's
    # signal is its confirmed-crown bar in _best.pt; rungs are measured on the WINNER only.
    res = measure(agent.greedy_move,
                  (f"{TAG} " if TAG else "") + f"qlearn a{ALPHA} g{GAMMA} lam{LAMBDA} wu{WARMUP} (Merge 2)",
                  ELO_GAMES, merge=2) if ELO_GAMES > 0 else {}
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": agent.net.state_dict(), "arch": ARCH, "enc": ENC, "zca": bool(ZCA),
                "opt_state": opt.state_dict(), "cum_games": cum_games + games_played,
                "rating": agent_elo_r, "ts": int(time.time())}, CKPT)

    result = {"tag": TAG, "seed": SEED, "epoch_games": epoch_games, "epochs_run": epochs_run,
              "resumed": RESUME, "cum_games": cum_games + games_played,
              "pool_sf_pts": run_sf_pts, "pool_sf_n": run_sf_n,
              "alpha": ALPHA, "gamma": GAMMA, "lambda": LAMBDA, "adaptive_lambda": ADAPTIVE_LAMBDA,
              "warmup": WARMUP, "elo": res.get("elo"), "elo_lo": res.get("elo_lo"),
              "elo_hi": res.get("elo_hi"), "vs_sf_score": res.get("vs_sf_score"),
              "vs_random_score": res.get("vs_random_score"), "ts": int(time.time())}
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print(f"\nsaved {CKPT} | appended {RESULTS}")
    elo = res.get("elo")
    # --- Optuna objective: POOLED anchored Elo over EVERY SF game the run played (live per-sample games
    # + the final measure) -> ~5x the resolution of the final alone at zero extra compute. A small ordinal
    # proxy tie-break (10 Elo per 1.0 vs-random score, i.e. <=10 Elo total, far below one real SF point)
    # gives TPE a gradient BELOW the shutout floor instead of a flat objective (S1 failure mode: all
    # trials identical at the floor -> TPE learned nothing).
    final_pts = (res.get("vs_sf_score") or 0.0) * ELO_GAMES
    pool_pts, pool_n = run_sf_pts + final_pts, run_sf_n + ELO_GAMES
    objective = (1320 + elo_diff(pool_pts / max(pool_n, 1), max(pool_n, 1))) + 10.0 * (res.get("vs_random_score") or 0.0)
    print(f"pooled objective: sf {pool_pts:g}/{pool_n} -> {objective:.1f} (final-only elo {elo})")
    print(f"KILL-CHECK elo {objective:.1f}")


if __name__ == "__main__":
    main()
