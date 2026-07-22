"""Selectable-agent factory (spec/entrypoint.spec.md :Selectable-agent:).

One home for loading a trained agent and returning its move function — shared by :Play-mode: and
:Measure-mode:. Also `AgentAdapter`, which lets any move_fn drive the terminal-interface front-end
(`terminal_board.py`), and `play_loop`, a simple SAN/UCI REPL fallback.
"""
import math
import os
import random

import chess

TAU_ARGMAX = 0.02   # at/below this the champion plays its full-depth deep-search argmax
DIFF_DEPTH = 2      # difficulty band: root sampling over depth-2-backed child values


class ChampionAgent:
    """Champion move source with a :Strength-temperature: dial (spec/archive/
    dynamic-difficulty.spec.md). Callable as a plain move_fn (tau=0 = full strength);
    `root_move(board, tau)` is the dial: tau <= TAU_ARGMAX -> deep argmax at `depth`;
    otherwise softmax-sample over DIFF_DEPTH-backed child values (mover-perspective),
    refusing a repetition when any alternative exists (pinned H2H lesson: deterministic
    or near-deterministic play cycle-locks into fivefold repetition)."""

    def __init__(self, srch, depth=9, diff_depth=DIFF_DEPTH, rng=None):
        self.srch = srch
        self.depth = depth
        self.diff_depth = diff_depth
        self.rng = rng or random.Random()

    def __call__(self, board):
        return self.root_move(board, 0.0)[0]

    def value(self, board):
        """White-absolute value in [-1,1] (tanh of the native static score)."""
        return math.tanh(float(self.srch.score(board.fen())))

    def _child_values(self, board):
        """(moves, mover-perspective backed values at diff_depth)."""
        sgn = 1.0 if board.turn == chess.WHITE else -1.0
        moves, vals = [], []
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                v = sgn * 1e9                       # mover mates: always best
            elif board.is_game_over():
                v = 0.0                             # stalemate / dead draw
            elif self.diff_depth >= 2:
                v = sgn * float(self.srch.search(board.fen(), self.diff_depth - 1)[1])
            else:
                v = sgn * math.tanh(float(self.srch.score(board.fen())))
            board.pop()
            moves.append(mv)
            vals.append(v)
        return moves, vals

    def root_move(self, board, tau=0.0):
        """Return (played_move, argmax_move) at strength-temperature tau."""
        if tau <= TAU_ARGMAX:
            mv = chess.Move.from_uci(self.srch.search(board.fen(), self.depth)[0])
            return mv, mv
        moves, vals = self._child_values(board)
        best_i = max(range(len(moves)), key=lambda i: vals[i])
        mx = vals[best_i]
        ws = [math.exp(min((v - mx) / tau, 0.0)) for v in vals]
        total = sum(ws)
        r = self.rng.random() * total
        pick = len(moves) - 1
        acc = 0.0
        for i, w in enumerate(ws):
            acc += w
            if r <= acc:
                pick = i
                break
        # repetition refusal: if the sampled move repeats, take the best non-repeating one
        board.push(moves[pick])
        repeats = board.is_repetition(2)
        board.pop()
        if repeats:
            for i in sorted(range(len(moves)), key=lambda i: -vals[i]):
                board.push(moves[i])
                ok = not board.is_repetition(2)
                board.pop()
                if ok:
                    pick = i
                    break
        return moves[pick], moves[best_i]


def _load_tower(path, dev):
    """Tolerant ChessResNet loader (bare state_dict OR {'state_dict': ...})."""
    import torch
    from chessdq.resnet_model import ChessResNet
    net = ChessResNet().to(dev)
    ck = torch.load(path, map_location=dev)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    net.load_state_dict(sd); net.eval()
    return net


def _puct_value_fn(net, dev):
    """White-absolute value readout (net value head) for the :Board-readout:."""
    import torch
    from chessdq.resnet_model import encode18
    @torch.no_grad()
    def v(board):
        x = encode18(board).unsqueeze(0).to(dev)
        return float(net(x)[0].item())
    return v


def make_agent(name="champion", playouts=160, engine_time=0.3, depth=9):
    """Return (label, move_fn, value_fn). move_fn(board) -> a legal move (or None); value_fn may be
    None. The default agent is the CHAMPION — the self-play RL net inside native alpha-beta d9,
    measured 1878 (95% 1756..2000) by the multi-anchor MLE ladder vs SF@1500-2300
    (experiments/anchor_ladder.py, 2026-07-18; supersedes the saturated 1320-anchor scout)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "champion":
        path = "models/champion.pt"
        if not os.path.exists(path):
            print(f"{path} not found — promote a graduated net first (see spec/bullet-route.spec.md).")
            return "champion", None, None
        import importlib
        from chessdq.corpus_gen import raw_weights
        w, b = raw_weights(path)                     # kc: ZCA identity-gated; amap: raw 897
        try:
            rsearch4 = importlib.import_module("rsearch4")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "rsearch4 (native search extension) is not built. From the repo root, run:\n"
                "    pip install maturin\n"
                "    cd rsearch && maturin develop --release\n"
                "See README.md 'Building the native search extension' for details."
            ) from e
        srch = rsearch4.Searcher(w, b)
        champ = ChampionAgent(srch, depth=depth)
        return (f"champion(d{depth}, amap-897 search-distilled, +232 Elo H2H over the 1878 net)", champ, champ.value)
    if name == "puct":
        path = "models/tower_puct.pt"
        if not os.path.exists(path):
            print(f"{path} not found — Train it first (main.py -> Train).")
            return "net+PUCT", None, None
        from chessdq.puct_selfplay import puct_move
        net = _load_tower(path, dev)
        return (f"net+PUCT({playouts})",
                (lambda b: puct_move(b, net, dev, playouts)),
                _puct_value_fn(net, dev))
    if name == "engine":
        from chessdq.engine import AlphaBetaEngine, pst_eval
        eng = AlphaBetaEngine(time_limit=engine_time)
        return (f"alpha-beta@{engine_time}s",
                (lambda b: eng.search(b)[0]),
                (lambda b: pst_eval(b)))
    if name == "nnue":
        # :Deployed-agent: — NNUE :Critic-leaf: + :Phi-widening: on the sound alpha-beta, ONE
        # persistent engine so its Zobrist TT carries every scored node forward (:Tree-reuse:).
        from chessdq.engine import AlphaBetaEngine, pst_eval
        path = "models/nnue.pt"
        if os.path.exists(path):
            from chessdq.nnue_model import load_nnue, make_incremental_nnue_eval
            net = load_nnue(path, dev)
            eval_fn = make_incremental_nnue_eval(net, dev)   # Stage 3: ~8x faster -> deeper at time
            label = f"nnue+phi-widen@{engine_time}s"
        else:
            print(f"{path} not found — falling back to pst_eval for the phi-widen agent.")
            eval_fn = pst_eval
            label = f"pst+phi-widen@{engine_time}s"
        eng = AlphaBetaEngine(eval_fn=eval_fn, phi_widen=True, time_limit=engine_time)
        return (label, (lambda b: eng.search(b)[0]), eval_fn)
    if name == "beam":
        from chessdq.play_beam import beam_mover
        label, mv = beam_mover(dev)
        return label, mv, None
    raise ValueError(f"unknown agent '{name}' (champion | puct | engine | beam | nnue)")


class _ValueShim:
    def __init__(self, value_fn):
        self._v = value_fn
        self.last_root_best = None    # argmax root move of the agent's last selection
    def get_q_value(self, board):
        return self._v(board)


class AgentAdapter:
    """Adapts a move_fn to the interface `terminal_board.py` expects: a settable `.board`,
    `.get_best_move()`, `.dqn_agent.get_q_value()`, `.elo_calibrator`, `.difficulty_settings`.
    Lets the spec-governed front-end drive any :Selectable-agent: without the retired DQN."""
    def __init__(self, move_fn, value_fn=None, elo_calibrator=None, difficulty_settings=None):
        self.board = None
        self._move_fn = move_fn
        self.dqn_agent = _ValueShim(value_fn or (lambda b: 0.0))
        self.elo_calibrator = elo_calibrator
        self.difficulty_settings = difficulty_settings or {"enabled": False, "mode": "off", "offset": 0.0}

    def get_best_move(self, temperature=0.0):
        """Move at the controller's :Strength-temperature:. Agents exposing `root_move`
        (the champion) honor it and report their argmax via dqn_agent.last_root_best
        (the :Move-regret: baseline terminal_board reads); others ignore temperature."""
        rm = getattr(self._move_fn, "root_move", None)
        if rm is not None:
            mv, best = rm(self.board, temperature)
            self.dqn_agent.last_root_best = best
            return mv
        self.dqn_agent.last_root_best = None
        return self._move_fn(self.board)


def play_loop(move_fn, label="agent", human_white=True):
    """Simple SAN/UCI REPL fallback (used when the rich terminal_board front-end is not selected)."""
    board = chess.Board()
    print(f"vs {label}. Moves as SAN (Nf3) or UCI (g1f3); 'quit' to exit.")
    while not board.is_game_over():
        print("\n" + str(board))
        if (board.turn == chess.WHITE) == human_white:
            raw = input("Your move: ").strip()
            if raw in ("quit", "q"):
                return
            try:
                mv = board.parse_san(raw)
            except ValueError:
                try:
                    mv = chess.Move.from_uci(raw)
                    if mv not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("Illegal / unparseable; try again.")
                    continue
            board.push(mv)
        else:
            print(f"{label} thinking...")
            mv = move_fn(board)
            if mv is None:
                return
            print(f"{label} plays: {board.san(mv)}")
            board.push(mv)
    print("\n" + str(board))
    print("Result:", board.result())
