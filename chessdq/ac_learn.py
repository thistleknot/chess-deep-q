"""Merge 3 — Online On-Policy Actor-Critic with Eligibility Traces (spec/actor-critic.spec.md).

Literal transplant of Sutton & Barto (2018) p.332 "Actor-Critic with Eligibility Traces (episodic)"
adapted to self-play chess in the White-absolute frame:

  per move:  A ~ pi(.|S,theta)
             delta = R + gamma*v(S') - v(S)          (v(terminal)=0; ply-cap = terminal draw)
             z_w  <- gamma*lam_w*z_w + grad_w v(S)
             z_th <- gamma*lam_th*z_th + I*sigma*grad_th ln pi(A|S)
             w  += alpha_w * delta * z_w
             th += alpha_th * (delta * z_th + beta_t * I * grad_th H(pi))
             I  <- gamma*I

FULLY ONLINE and ON-POLICY: no replay buffer, no frozen generator, no greedy-max target — the deadly
triad's off-policy leg is removed by construction (S&B 11.3). Policy = softmax over legal AFTERSTATE
preferences h = theta^T x (same 769-dim encoding as the critic); sigma = +1 White to move, -1 Black.
Entropy bonus (Deep RL in Action ch.5) is THE exploration mechanism; beta anneals, never reaching 0.

Harness (epochs = eval cadence + Elo patience, pooled KILL-CHECK objective, metrics JSONL for the
console, checkpoint/resume) mirrors qlearn.py so the whole pipeline (server, tuner, dashboard) works
unchanged. Usage: python ac_learn.py [epoch_games] [max_epochs]
"""
import os
import sys
import json
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import chess

from chessdq.chess_rl import ChessEnv
from chessdq.cem_loop import encode, NIN
from chessdq.measure_ladder import random_mover, heuristic_mover
from chessdq.measure_elo import measure, play as elo_play, elo_diff

# ---- knobs (QLEARN_* namespace reused so server/tuner wiring stays uniform) ----
ALPHA_W    = float(os.environ.get("QLEARN_ALPHA_W", "3e-3"))    # critic step size
ALPHA_TH   = float(os.environ.get("QLEARN_ALPHA_TH", "1e-4"))   # actor step size (slower — two-timescale)
GAMMA      = float(os.environ.get("QLEARN_GAMMA", "0.99"))
LAMBDA_W   = float(os.environ.get("QLEARN_LAMBDA_W", "0.8"))    # critic trace decay
LAMBDA_TH  = float(os.environ.get("QLEARN_LAMBDA_TH", "0.8"))   # actor trace decay
BETA       = float(os.environ.get("QLEARN_BETA", "0.01"))       # entropy bonus start (anneals to /10)
ENT_WARMUP = float(os.environ.get("QLEARN_ENT_WARMUP", "0.5"))  # beta anneal e-folding fraction
ARCH       = os.environ.get("QLEARN_ARCH", "linear")            # critic arch: linear | mlp
ACTOR_ARCH = os.environ.get("QLEARN_ACTOR_ARCH", "linear")      # actor arch: linear (closed-form
# gradients) | mlp (autograd) — the measured wall: linear preferences over shared piece-square
# weights stayed at 97.7% of uniform entropy under heat×10, γ^t removal, σ-norm ×40, and mate-bank
# curriculum; sharp move preferences need a nonlinear head (spec :Policy-parameterization:).
HIDDEN     = int(os.environ.get("QLEARN_HIDDEN", "64"))
CURRICULUM = float(os.environ.get("QLEARN_CURRICULUM", "0"))    # exploring-starts fraction (0=off):
# probability an episode starts from a LEGAL material-reduced position (kings + 1..6 pieces) — the
# fertile, mate-dense subspace — instead of the draw-swamp opening. Fraction anneals DOWN via the
# shared anneal() (CUR_WARMUP e-fold) as the policy strengthens. Evals/Elo ALWAYS play the standard
# game. Spec :Curriculum-starts:.
CUR_WARMUP = float(os.environ.get("QLEARN_CUR_WARMUP", "0.5"))
MATE_BANK  = os.environ.get("QLEARN_MATE_BANK", "data/mate_bank.jsonl")  # curriculum v2: persistent
# bank of REAL mate-adjacent positions (k plies before an actual self-play mate). v1's random reduced
# positions were stalemate traps (D13: decisive 0.076 <= baseline); starts harvested from reachable
# mates put episodes at the true goal boundary. Bank grows every decisive game, across runs.
ADV_NORM   = os.environ.get("QLEARN_ADV_NORM", "0") == "1"      # normalize the ACTOR's advantage by a
# running sigma of delta (EMA) — the user's Sharpe/sigma idea in its sound home (A2C-standard
# advantage normalization, DRL-in-Action ch.5). Critic keeps RAW delta. Measured motivation:
# |delta|~0.02 (tanh critic + sparse reward) -> actor steps ~1e-5 -> entropy pinned (D9/D10/D11).
PG_DISCOUNT= os.environ.get("QLEARN_PG_DISCOUNT", "1") == "1"   # 1 = textbook I=gamma^t weighting of the
# policy gradient (S&B p.332 box, correct for the DISCOUNTED objective); 0 = the practical A2C variant
# (DRL-in-Action ch.5): drop gamma^t — small bias, but WITHOUT it late-game moves (where every mate
# lives) get gamma^t ~ 0.02 weight and the actor cannot learn to finish games. Measured: D9/D10
# entropy pinned at ~3.26 under the textbook weighting.
ANCHOR     = os.environ.get("QLEARN_ANCHOR", "1")               # gate mode: 0=off | 1=collapse guard
# (revert only when strength < REVERT_FRAC*best — lets mild wandering EXPLORE, catches collapses) |
# hard=revert every non-improving epoch (D8 showed hard gating stalls: a lucky bar froze 6 epochs).
REVERT_FRAC= float(os.environ.get("QLEARN_REVERT_FRAC", "0.5"))
PLY_CAP    = int(os.environ.get("QLEARN_PLY_CAP", "160"))
ELO_GAMES  = int(os.environ.get("QLEARN_ELO_GAMES", "20"))
PROXY_GAMES= int(os.environ.get("QLEARN_PROXY_GAMES", "20"))
SEED       = int(os.environ.get("QLEARN_SEED", "0"))
EPOCH_GAMES= int(os.environ.get("QLEARN_EPOCH_GAMES", "200"))
MAX_EPOCHS = int(os.environ.get("QLEARN_MAX_EPOCHS", "10"))
PATIENCE   = int(os.environ.get("QLEARN_PATIENCE", "2"))
EPOCH_ELO_GAMES = int(os.environ.get("QLEARN_EPOCH_ELO_GAMES", "0"))
LOG_EVERY  = int(os.environ.get("QLEARN_LOG_EVERY", "100"))
BATCH_GAMES= int(os.environ.get("QLEARN_BATCH_GAMES", "20"))    # cheap-metrics row cadence (NOT updates)
RESUME     = os.environ.get("QLEARN_RESUME", "0") == "1"
CKPT       = os.environ.get("QLEARN_CKPT", "models/ac_learn.pt")
TAG        = os.environ.get("QLEARN_TAG", "")
METRICS    = "data/qlearn_metrics.jsonl"
RESULTS    = "data/qlearn_results.jsonl"

_PVAL = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)


def anneal(start, end, progress, w):
    """Shared asymptotic schedule (spec :Entropy-regularization:): decays start->end, never arriving."""
    return end + (start - end) * math.exp(-progress / max(w, 1e-6))


class Critic(nn.Module):
    """v(x) = tanh(head(x)) in [-1,1], White-absolute — same family as Merge 2's ValueNet."""

    def __init__(self, arch="linear", hidden=64):
        super().__init__()
        if arch == "mlp":
            self.head = nn.Sequential(nn.Linear(NIN, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.head = nn.Linear(NIN, 1)

    def forward(self, x):
        return torch.tanh(self.head(x)).squeeze(-1)


class ActorNet(nn.Module):
    """Move-preference head h(x_afterstate) for the softmax policy (mlp actor)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(NIN, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(x).squeeze(-1)


def afterstates(board):
    """(moves, X) for the side to move: every legal move's resulting-position encoding, rows of X."""
    moves = list(board.legal_moves)
    X = np.empty((len(moves), NIN), dtype=np.float32)
    for i, mv in enumerate(moves):
        board.push(mv)
        X[i] = encode(board)
        board.pop()
    return moves, X


class LinearActor:
    """theta^T x preferences with CLOSED-FORM gradients (textbook-exact, fast). Measured wall: cannot
    sharpen (entropy pinned at ~0.98 of uniform across every mechanics fix)."""

    def __init__(self, dev=None):
        self.theta = np.zeros(NIN, dtype=np.float32)

    def prefs(self, X):
        return X @ self.theta

    def reset_traces(self):
        self.z = np.zeros(NIN, dtype=np.float32)

    def select(self, X, sigma, rng):
        logits = sigma * self.prefs(X)
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        a = int(rng.choices(range(len(p)), weights=p)[0])
        xbar = p @ X
        self._glog = sigma * (X[a] - xbar)
        self._gH = -sigma * ((p * (np.log(p + 1e-12) + 1.0)) @ (X - xbar))
        return a, p

    def apply(self, adv, beta_t, I):
        self.z = GAMMA * LAMBDA_TH * self.z + I * self._glog
        self.theta += ALPHA_TH * (adv * self.z + beta_t * I * self._gH)

    def state(self):
        return {"theta": torch.from_numpy(self.theta)}

    def load(self, ck):
        self.theta = ck["theta"].cpu().numpy().astype(np.float32)


class MLPActor:
    """Nonlinear preference head, gradients via autograd; per-parameter eligibility trace tensors
    (same trace algebra as the critic's zw)."""

    def __init__(self, dev, hidden=64):
        self.dev = dev
        self.net = ActorNet(hidden).to(dev)
        self.params = list(self.net.parameters())

    def prefs(self, X):
        with torch.no_grad():
            return self.net(torch.from_numpy(X).to(self.dev)).cpu().numpy()

    def reset_traces(self):
        self.z = [torch.zeros_like(p) for p in self.params]

    def select(self, X, sigma, rng):
        logits = sigma * self.net(torch.from_numpy(X).to(self.dev))
        logp = torch.log_softmax(logits, 0)
        p = torch.exp(logp).detach().cpu().numpy()
        a = int(rng.choices(range(len(p)), weights=p)[0])
        self.net.zero_grad()
        logp[a].backward(retain_graph=True)
        self._glog = [prm.grad.detach().clone() for prm in self.params]
        self.net.zero_grad()
        (-(torch.exp(logp) * logp).sum()).backward()
        self._gH = [prm.grad.detach().clone() for prm in self.params]
        return a, p

    def apply(self, adv, beta_t, I):
        with torch.no_grad():
            for prm, z, gl, gh in zip(self.params, self.z, self._glog, self._gH):
                z.mul_(GAMMA * LAMBDA_TH).add_(I * gl)
                prm.add_(ALPHA_TH * (adv * z + beta_t * I * gh))

    def state(self):
        return {"actor": self.net.state_dict()}

    def load(self, ck):
        self.net.load_state_dict(ck["actor"])


def make_actor(dev):
    return MLPActor(dev, HIDDEN) if ACTOR_ARCH == "mlp" else LinearActor(dev)


def greedy_move_fn(actor):
    """Deterministic actor mode — the measurement policy (all displayed scores come from THIS)."""
    def mv(board):
        moves, X = afterstates(board)
        sigma = 1.0 if board.turn == chess.WHITE else -1.0
        return moves[int(np.argmax(sigma * actor.prefs(X)))]
    return mv


def learned_piece_worth(critic):
    """Piece values off the LINEAR critic's weights, pawn=1 (spec :Piece-worth-observability:)."""
    if ARCH != "linear":
        return None
    w = critic.head.weight.detach().cpu().numpy().reshape(-1)
    raw = [float(w[t * 64:(t + 1) * 64].mean() - w[(t + 6) * 64:(t + 7) * 64].mean()) for t in range(5)]
    pawn = raw[0]
    if abs(pawn) < 1e-8:
        return {"pawn_raw": round(pawn, 6), "N": None, "B": None, "R": None, "Q": None}
    return {"pawn_raw": round(pawn, 6),
            "N": round(raw[1] / pawn, 2), "B": round(raw[2] / pawn, 2),
            "R": round(raw[3] / pawn, 2), "Q": round(raw[4] / pawn, 2)}


def evaluate_greedy(actor, games):
    """Greedy-actor diagnostics vs random + 1-ply PST heuristic (same contract as qlearn)."""
    gmv = greedy_move_fn(actor)

    def run(opp, track):
        s, tot_mat, tot_plies = 0.0, 0.0, 0
        for g in range(games):
            aw = (g % 2 == 0)
            b = chess.Board(); plies = 0
            while not b.is_game_over() and plies < PLY_CAP:
                b.push(gmv(b) if (b.turn == chess.WHITE) == aw else opp(b))
                plies += 1
            if b.is_checkmate():
                s += 1.0 if ((b.turn == chess.BLACK) == aw) else 0.0
            else:
                s += 0.5
            if track:
                x = encode(b)
                m = 0.0
                for pt in range(6):
                    m += float(_PVAL[pt] * (x[pt * 64:(pt + 1) * 64].sum() - x[(pt + 6) * 64:(pt + 7) * 64].sum()))
                tot_mat += m if aw else -m
                tot_plies += plies
        return (s / games, tot_mat / games, tot_plies / (2 * games)) if track else s / games

    wr_rand = run(random_mover, False)
    wr_heur, avg_mat, avg_turns = run(heuristic_mover, True)
    return wr_rand, wr_heur, avg_mat, avg_turns


def greedy_elo(actor, sf, sf_lim, n):
    """Anchored Elo + nominal score of the greedy actor vs SF@1320 (half-point clamp via elo_diff)."""
    gmv = greedy_move_fn(actor)
    sW, sD, _sL, _ = elo_play(gmv, lambda b: sf.play(b, sf_lim).move, n)
    pts = sW + 0.5 * sD
    return round(1320 + elo_diff(pts / n, n)), pts


def load_mate_bank(cap=5000):
    """Persistent bank of mate-adjacent FENs harvested from real decisive self-play games."""
    if not os.path.exists(MATE_BANK):
        return []
    fens = [json.loads(l)["fen"] for l in open(MATE_BANK) if l.strip()]
    return fens[-cap:]


def curriculum_start(rng, n_pieces):
    """Legal material-reduced exploring start: two kings + n random pieces on random squares.
    Rejection-samples until python-chess validates the position (kings apart, mover-legal, no pawns
    on promotion ranks, not already over). Falls back to the standard board if unlucky."""
    types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    minor = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    for _ in range(200):
        b = chess.Board(None)
        squares = rng.sample(range(64), n_pieces + 2)
        b.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        for sq in squares[2:]:
            pt = rng.choice(types)
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                pt = rng.choice(minor)
            b.set_piece_at(sq, chess.Piece(pt, rng.choice([chess.WHITE, chess.BLACK])))
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if b.is_valid() and not b.is_game_over():
            return b
    return chess.Board()


def play_train_game(actor, critic, zw, dev, rng, env, beta_t, start_board=None, dvar=None):
    """One SELF-PLAY episode with fully online AC updates (spec :AC-episodic-traces:).
    Mutates actor and critic weights in place; reuses preallocated zw critic-trace tensors
    (zeroed here at episode start). Returns (z, plies, deltas, ents, fens)."""
    params = list(critic.parameters())
    for z in zw:
        z.zero_()
    actor.reset_traces()
    I = 1.0
    deltas, ents = [], []
    fens = []                                          # trailing positions -> mate-bank harvesting
    board = env.reset(start_board)
    z = 0.0
    done = False
    while not done and env.plies < PLY_CAP:
        fens.append(board.fen())
        if len(fens) > 8:
            fens.pop(0)
        sigma = 1.0 if board.turn == chess.WHITE else -1.0
        moves, X = afterstates(board)
        a, p = actor.select(X, sigma, rng)
        H = float(-(p * np.log(p + 1e-12)).sum())
        ents.append(H / max(1e-9, math.log(max(2, len(moves)))))   # NORMALIZED [0..1]: sharpness that is
        # comparable across positions with different move counts (raw H is confounded by |legal|)

        # critic value + gradient at S (current position), BEFORE the move
        xS = torch.from_numpy(encode(board)).to(dev)
        critic.zero_grad()
        vS = critic(xS)
        vS.backward()
        vS_val = float(vS.item())

        board, z, done = env.step(moves[a])
        terminal = done or env.plies >= PLY_CAP
        if terminal:
            v_next, R = 0.0, z                      # v(terminal) = 0; reward = White-absolute outcome
        else:
            with torch.no_grad():
                v_next = float(critic(torch.from_numpy(encode(board)).to(dev)).item())
            R = 0.0
        delta = R + GAMMA * v_next - vS_val
        deltas.append(delta)
        adv = delta
        if ADV_NORM and dvar is not None:               # scale-free actor signal; critic keeps raw delta
            dvar[0] = 0.99 * dvar[0] + 0.01 * (delta * delta)
            adv = delta / math.sqrt(dvar[0] + 1e-8)

        # critic trace + update (S&B box)
        with torch.no_grad():
            for prm, tr in zip(params, zw):
                tr.mul_(GAMMA * LAMBDA_W).add_(prm.grad)
                prm.add_(ALPHA_W * delta * tr)

        # actor trace + update (closed-form for linear, autograd for mlp — same trace algebra)
        actor.apply(adv, beta_t, I)
        if PG_DISCOUNT:
            I *= GAMMA
    return z, env.plies, deltas, ents, fens


def main():
    epoch_games = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCH_GAMES
    max_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_EPOCHS
    total_games = epoch_games * max_epochs
    log_every = max(1, min(LOG_EVERY, epoch_games))
    batch_games = max(1, min(BATCH_GAMES, epoch_games))
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    np.random.seed(SEED)
    # CPU by default: fully ONLINE updates are thousands of tiny single-row ops per game — per-move
    # CUDA launch/sync overhead dominates any batching gain (measured: ~4 min/game on contended GPU).
    dev = torch.device(os.environ.get("QLEARN_DEV", "cpu"))

    critic = Critic(ARCH, HIDDEN).to(dev)
    actor = make_actor(dev)
    cum_games = 0
    resume_bar = -1.0
    # resume from BEST-visited by default (QLEARN_RESUME_BEST=0 for latest) -> chained runs ratchet
    ckpt_path = CKPT.replace(".pt", "_best.pt") \
        if (os.environ.get("QLEARN_RESUME_BEST", "1") == "1"
            and os.path.exists(CKPT.replace(".pt", "_best.pt"))) else CKPT
    if RESUME and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        if ck.get("arch", "linear") == ARCH and ck.get("actor_arch", "linear") == ACTOR_ARCH:
            try:
                critic.load_state_dict(ck["critic"])
                actor.load(ck)
                cum_games = int(ck.get("cum_games", 0))
                resume_bar = float(ck.get("strength", -1.0))   # PERSIST the acceptance bar across legs:
                # without it a resumed leg's first mediocre epoch overwrites the stored best (night-chain
                # bug: leg2 wrote 2.56 over leg1's 3.85) — the cross-run ratchet requires the bar to travel
                print(f"RESUMED {ckpt_path} (+{cum_games} prior games, bar {resume_bar:.2f})", flush=True)
            except (RuntimeError, KeyError) as e:
                print(f"resume mismatch ({e}) -> fresh", flush=True)
        else:
            print(f"resume arch mismatch -> fresh", flush=True)
    if not RESUME and os.path.exists(CKPT.replace(".pt", "_best.pt")):
        os.remove(CKPT.replace(".pt", "_best.pt"))     # fresh run = new lineage; never gate vs an old one
    zw = [torch.zeros_like(p) for p in critic.parameters()]

    env = ChessEnv(ply_cap=PLY_CAP)
    os.makedirs("data", exist_ok=True)
    open(METRICS, "w").close()

    sf, sf_lim = None, None
    if EPOCH_ELO_GAMES > 0:
        import glob
        import chess.engine
        sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
        if sfp:
            try:                                   # a hung UCI handshake froze a whole run (leg stuck
                sf = chess.engine.SimpleEngine.popen_uci(sfp[0], timeout=20.0)   # 1h+, zero output)
                sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
                sf_lim = chess.engine.Limit(time=0.05)
            except Exception as e:
                print(f"stockfish unavailable ({e}) -> live Elo off this run", flush=True)
                sf, sf_lim = None, None

    print(f"Merge 3 Actor-Critic (online, on-policy, traces) | <={max_epochs} epochs x {epoch_games} | "
          f"patience {PATIENCE} | critic={ARCH} actor={ACTOR_ARCH} a_w={ALPHA_W} a_th={ALPHA_TH} g={GAMMA} "
          f"lam_w={LAMBDA_W} lam_th={LAMBDA_TH} beta={BETA}->~{BETA/10:g} dev={dev}\n", flush=True)

    games_played = 0
    ep_sf_pts, ep_sf_n = 0.0, 0
    run_sf_pts, run_sf_n = 0.0, 0
    best_strength, stale, epoch, ep_games = resume_bar, 0, 0, 0
    wr_rand = wr_heur = 0.0
    zs_log, d_log, h_log, since_log = [], [], [], 0
    epoch_strength = None
    dvar = [1.0]                            # running Var[delta] EMA for :ADV_NORM: (re-warms quickly)
    mate_bank = load_mate_bank()
    if CURRICULUM > 0:
        print(f"mate bank: {len(mate_bank)} positions", flush=True)
    beta_t = BETA
    t = time.time()
    while games_played < total_games:
        chunk = min(batch_games, total_games - games_played, epoch_games - ep_games)
        for _ in range(chunk):
            progress = (cum_games + games_played) / max(1, cum_games + total_games)
            beta_t = anneal(BETA, BETA / 10.0, progress, ENT_WARMUP)
            start_b = None
            if CURRICULUM > 0 and rng.random() < anneal(CURRICULUM, CURRICULUM / 4.0, progress, CUR_WARMUP):
                # v2: prefer REAL mate-adjacent starts from the bank; random-reduced only as fallback
                if mate_bank and rng.random() < 0.7:
                    try:
                        start_b = chess.Board(rng.choice(mate_bank))
                    except ValueError:
                        start_b = curriculum_start(rng, rng.randint(1, 6))
                else:
                    start_b = curriculum_start(rng, rng.randint(1, 6))
            z, _plies, deltas, ents, fens = play_train_game(actor, critic, zw, dev, rng, env,
                                                            beta_t, start_b, dvar)
            if abs(z) == 1.0 and start_b is None and len(fens) >= 2:
                k = rng.randint(1, min(6, len(fens) - 1))       # harvest k-plies-before-mate position
                fen = fens[-(k + 1)]
                mate_bank.append(fen)
                with open(MATE_BANK, "a") as fh:
                    fh.write(json.dumps({"fen": fen, "z": z}) + "\n")
            zs_log.append(z)
            d_log.extend(deltas)
            h_log.extend(ents)
            games_played += 1
            ep_games += 1
        since_log += chunk

        do_eval = since_log >= log_every or games_played >= total_games or ep_games >= epoch_games
        if do_eval:
            wr_rand, wr_heur, avg_mat, avg_turns = evaluate_greedy(actor, PROXY_GAMES)
            epoch_elo, sf_pts = greedy_elo(actor, sf, sf_lim, EPOCH_ELO_GAMES) if sf is not None else (None, None)
            if sf_pts is not None:
                ep_sf_pts += sf_pts; ep_sf_n += EPOCH_ELO_GAMES
                run_sf_pts += sf_pts; run_sf_n += EPOCH_ELO_GAMES
            since_log = 0
            if ep_games >= epoch_games:            # boundary sample -> THE goal curve (Elo over epochs)
                epoch_strength = round(100.0 * ((ep_sf_pts / ep_sf_n) if ep_sf_n else 0.0)
                                       + (wr_rand + wr_heur), 3)
        d_arr = np.array(d_log, dtype=np.float32) if d_log else np.zeros(1, dtype=np.float32)
        row = {"games_played": games_played, "epoch": epoch + 1, "tau": round(beta_t, 5),
               "lam_eff": LAMBDA_W, "td_sigma": round(float(d_arr.std()), 4),
               "loss": round(float((d_arr ** 2).mean()), 5), "buf": 0,
               "avg_points": round(avg_mat, 2) if do_eval else None,
               "avg_turns": round(avg_turns, 1) if do_eval else None,
               "epoch_elo": epoch_elo if do_eval else None,
               "sf_pts": sf_pts if do_eval else None,
               "sf_n": EPOCH_ELO_GAMES if (do_eval and sf is not None) else None,
               "piece_vals": learned_piece_worth(critic),
               "avg_reward": round(float(np.mean(zs_log)), 3) if zs_log else None,
               "decisive": round(float(np.mean(np.abs(zs_log))), 3) if zs_log else None,
               "entropy": round(float(np.mean(h_log)), 4) if h_log else None,
               "epoch_strength": epoch_strength if (do_eval and ep_games >= epoch_games) else None,
               "wr_vs_random": round(wr_rand, 3) if do_eval else None,
               "wr_vs_heuristic": round(wr_heur, 3) if do_eval else None,
               "ts": int(time.time())}
        with open(METRICS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if do_eval:
            print(f"g {games_played} | ep {epoch+1} | beta {beta_t:.4f} | d_sig {row['td_sigma']:.3f} | "
                  f"loss {row['loss']:.4f} | pts {row['avg_points']} turns {row['avg_turns']} | "
                  f"sf {sf_pts if sf_pts is not None else '-'}/{EPOCH_ELO_GAMES} (elo {epoch_elo if epoch_elo is not None else '-'}) | "
                  f"vs_heur {wr_heur:.2f} | {time.time()-t:.0f}s", flush=True)
            t = time.time()
        zs_log, d_log, h_log = [], [], []

        if ep_games >= epoch_games:
            epoch += 1; ep_games = 0
            os.makedirs("models", exist_ok=True)
            torch.save({"critic": critic.state_dict(), **actor.state(), "arch": ARCH,
                        "actor_arch": ACTOR_ARCH, "strength": best_strength,
                        "cum_games": cum_games + games_played, "ts": int(time.time())}, CKPT)
            ep_sf_score = (ep_sf_pts / ep_sf_n) if ep_sf_n else 0.0
            strength = 100.0 * ep_sf_score + (wr_rand + wr_heur)
            print(f"epoch {epoch} strength: sf {ep_sf_pts}/{ep_sf_n} + proxy {wr_rand + wr_heur:.2f} "
                  f"-> {strength:.2f} (best {best_strength:.2f})", flush=True)
            ep_sf_pts, ep_sf_n = 0.0, 0
            if strength > best_strength + 1e-3:
                best_strength, stale = strength, 0
                # KEEP-BEST: RL curves are non-monotone (policy->data feedback); the run's output is
                # the best policy VISITED, not whoever it was when the music stopped.
                torch.save({"critic": critic.state_dict(), **actor.state(),
                            "arch": ARCH, "actor_arch": ACTOR_ARCH,
                            "cum_games": cum_games + games_played,
                            "strength": strength, "ts": int(time.time())},
                           CKPT.replace(".pt", "_best.pt"))
            else:
                stale += 1
                best_p = CKPT.replace(".pt", "_best.pt")
                revert = (ANCHOR == "hard") or (ANCHOR == "1" and strength < REVERT_FRAC * best_strength)
                if revert and os.path.exists(best_p):     # collapse (or hard gate) -> revert to anchor
                    ckb = torch.load(best_p, map_location=dev)
                    critic.load_state_dict(ckb["critic"])
                    actor.load(ckb)
                    print(f"anchor: epoch {epoch} REVERTED ({strength:.2f} vs best {best_strength:.2f})",
                          flush=True)
                if stale >= PATIENCE:
                    print(f"early stop: no improvement for {PATIENCE} epochs", flush=True)
                    break

    if sf is not None:
        sf.quit()
    epochs_run = round(games_played / epoch_games, 2)

    # The run's OUTPUT is the best-visited policy, always (keep-best selection, not last-weights luck)
    best_p = CKPT.replace(".pt", "_best.pt")
    if os.path.exists(best_p):
        ckb = torch.load(best_p, map_location=dev)
        critic.load_state_dict(ckb["critic"])
        actor.load(ckb)
        print(f"final measure on BEST-visited (strength {ckb.get('strength')})", flush=True)

    res = measure(greedy_move_fn(actor), f"ac[{ACTOR_ARCH}] a_w{ALPHA_W} a_th{ALPHA_TH} g{GAMMA} "
                  f"lam_w{LAMBDA_W} lam_th{LAMBDA_TH} b{BETA} (Merge 3)", ELO_GAMES, merge=3)
    os.makedirs("models", exist_ok=True)
    torch.save({"critic": critic.state_dict(), **actor.state(), "arch": ARCH,
                "actor_arch": ACTOR_ARCH, "strength": best_strength,
                "cum_games": cum_games + games_played, "ts": int(time.time())}, CKPT)

    result = {"tag": TAG, "algo": "ac", "seed": SEED, "epoch_games": epoch_games,
              "epochs_run": epochs_run, "resumed": RESUME, "cum_games": cum_games + games_played,
              "pool_sf_pts": run_sf_pts, "pool_sf_n": run_sf_n,
              "alpha_w": ALPHA_W, "alpha_th": ALPHA_TH, "gamma": GAMMA,
              "lambda_w": LAMBDA_W, "lambda_th": LAMBDA_TH, "beta": BETA,
              "elo": res.get("elo"), "elo_lo": res.get("elo_lo"), "elo_hi": res.get("elo_hi"),
              "vs_sf_score": res.get("vs_sf_score"), "vs_random_score": res.get("vs_random_score"),
              "ts": int(time.time())}
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print(f"\nsaved {CKPT} | appended {RESULTS}")
    elo = res.get("elo")
    final_pts = (res.get("vs_sf_score") or 0.0) * ELO_GAMES
    pool_pts, pool_n = run_sf_pts + final_pts, run_sf_n + ELO_GAMES
    objective = (1320 + elo_diff(pool_pts / pool_n, pool_n)) + 10.0 * (res.get("vs_random_score") or 0.0)
    print(f"pooled objective: sf {pool_pts:g}/{pool_n} -> {objective:.1f} (final-only elo {elo})")
    print(f"KILL-CHECK elo {objective:.1f}")


if __name__ == "__main__":
    main()
