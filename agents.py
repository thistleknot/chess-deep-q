"""Selectable-agent factory (spec/entrypoint.spec.md :Selectable-agent:).

One home for loading a trained agent and returning its move function — shared by :Play-mode: and
:Measure-mode:. Also `AgentAdapter`, which lets any move_fn drive the terminal-interface front-end
(`terminal_board.py`), and `play_loop`, a simple SAN/UCI REPL fallback.
"""
import os

import chess


def _load_tower(path, dev):
    """Tolerant ChessResNet loader (bare state_dict OR {'state_dict': ...})."""
    import torch
    from resnet_model import ChessResNet
    net = ChessResNet().to(dev)
    ck = torch.load(path, map_location=dev)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    net.load_state_dict(sd); net.eval()
    return net


def _puct_value_fn(net, dev):
    """White-absolute value readout (net value head) for the :Board-readout:."""
    import torch
    from resnet_model import encode18
    @torch.no_grad()
    def v(board):
        x = encode18(board).unsqueeze(0).to(dev)
        return float(net(x)[0].item())
    return v


def make_agent(name="puct", playouts=160, engine_time=0.3):
    """Return (label, move_fn, value_fn). move_fn(board) -> a legal move (or None); value_fn may be
    None. The default agent is the net+PUCT :Chess-RL-system: (the RL deliverable)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "puct":
        path = "models/tower_puct.pt"
        if not os.path.exists(path):
            print(f"{path} not found — Train it first (main.py -> Train).")
            return "net+PUCT", None, None
        from puct_selfplay import puct_move
        net = _load_tower(path, dev)
        return (f"net+PUCT({playouts})",
                (lambda b: puct_move(b, net, dev, playouts)),
                _puct_value_fn(net, dev))
    if name == "engine":
        from engine import AlphaBetaEngine, pst_eval
        eng = AlphaBetaEngine(time_limit=engine_time)
        return (f"alpha-beta@{engine_time}s",
                (lambda b: eng.search(b)[0]),
                (lambda b: pst_eval(b)))
    if name == "nnue":
        # :Deployed-agent: — NNUE :Critic-leaf: + :Phi-widening: on the sound alpha-beta, ONE
        # persistent engine so its Zobrist TT carries every scored node forward (:Tree-reuse:).
        from engine import AlphaBetaEngine, pst_eval
        path = "models/nnue.pt"
        if os.path.exists(path):
            from nnue_model import load_nnue, make_nnue_eval
            net = load_nnue(path, dev)
            eval_fn = make_nnue_eval(net, dev)
            label = f"nnue+phi-widen@{engine_time}s"
        else:
            print(f"{path} not found — falling back to pst_eval for the phi-widen agent.")
            eval_fn = pst_eval
            label = f"pst+phi-widen@{engine_time}s"
        eng = AlphaBetaEngine(eval_fn=eval_fn, phi_widen=True, time_limit=engine_time)
        return (label, (lambda b: eng.search(b)[0]), eval_fn)
    if name == "beam":
        from play_beam import beam_mover
        label, mv = beam_mover(dev)
        return label, mv, None
    raise ValueError(f"unknown agent '{name}' (puct | engine | beam | nnue)")


class _ValueShim:
    def __init__(self, value_fn):
        self._v = value_fn
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

    def get_best_move(self):
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
