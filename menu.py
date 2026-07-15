"""Single spec-governed entrypoint menu (spec/entrypoint.spec.md :Top-menu:).

Play / Train / Measure / Difficulty over the net+PUCT :Chess-RL-system:. The DQN goal-picker, flat
`.train()` loop, and AHA options are retired (the DQN lives in `legacy/`). Every action traces to a
spec: Play → terminal-interface + chess-rl (net+PUCT deliverable); Train → training-loop
:Stage-controller:; Measure → elo-measurement; Difficulty → dynamic-difficulty / elo-calibration.
"""
import chess

_PLAY_PLAYOUTS = 160          # :Difficulty-mode: strength dial → :Play-mode: / :Measure-mode:


def _pick_agent():
    """:Selectable-agent: — CHAMPION default (self-play RL net + native alpha-beta d9,
    claims 1670); the earlier deliverables stay selectable."""
    print("\nAgent:  1. CHAMPION (self-play RL + d9 search, 1670 claims — default)   "
          "2. net+PUCT   3. alpha-beta engine (baseline)   4. beam   5. nnue (phi-widen)")
    c = input("Choose (1-5, default 1): ").strip() or "1"
    return {"1": "champion", "2": "puct", "3": "engine", "4": "beam", "5": "nnue"}.get(c, "champion")


def play_mode():
    """Human vs a :Selectable-agent: through the terminal-interface front-end (fallback: REPL)."""
    from agents import make_agent, AgentAdapter, play_loop
    name = _pick_agent()
    label, move_fn, value_fn = make_agent(name, playouts=_PLAY_PLAYOUTS)
    if move_fn is None:
        return
    human_white = (input("Play as white? (y/n, default y): ").strip().lower() or "y") == "y"
    try:
        from terminal_board import TerminalChessBoard
        adapter = AgentAdapter(move_fn, value_fn)
        human_color = chess.WHITE if human_white else chess.BLACK
        TerminalChessBoard(chess.Board(), adapter, human_color=human_color).start()
    except Exception as e:
        print(f"(rich board unavailable: {e}; simple mode)")
        play_loop(move_fn, label, human_white)


def train_mode():
    """:Train-mode: → the :Stage-controller: (default: PUCT self-play, the measured climber)."""
    from train_control import train
    print("\nApproach:  1. puct (default — the climber)   2. selfplay (net-minimax expert iteration)")
    approach = {"1": "puct", "2": "selfplay"}.get(input("Choose (1-2, default 1): ").strip() or "1", "puct")
    try:
        iters = int(input("Iterations (default 8): ") or 8)
        games = int(input("Games per iter (default 96): ") or 96)
    except ValueError:
        iters, games = 8, 96
    print(f"Training {approach}: {iters} iters x {games} games — checkpoints each iter, "
          f"Ctrl-C to stop.\n")
    train(approach, iters, games)


def measure_mode():
    """:Measure-mode: — the :Selectable-agent: on the :Ladder: (:Measured-elo:)."""
    from agents import make_agent
    from measure_ladder import random_mover, heuristic_mover, adj_pst, play, elo_diff
    label, move_fn, _ = make_agent(_pick_agent(), playouts=_PLAY_PLAYOUTS)
    if move_fn is None:
        return
    try:
        games = int(input("Games per rung (default 16): ") or 16)
    except ValueError:
        games = 16
    print(f"\n{label} vs ladder, {games} games/rung:")
    for rung, opp in (("random", random_mover), ("heuristic-1ply", heuristic_mover)):
        W, D, L = play(move_fn, opp, games, adj_pst)
        s = (W + 0.5 * D) / (W + D + L)
        print(f"  vs {rung:14s}: {W}W-{D}D-{L}L  score {s:.2f}  elo_diff {elo_diff(s):+.0f}", flush=True)


def difficulty_mode():
    """:Difficulty-mode: — the play-strength dial. For net+PUCT the honest knob is playouts."""
    global _PLAY_PLAYOUTS
    print(f"\nPlay strength = PUCT playouts (more = stronger, slower). Current: {_PLAY_PLAYOUTS}")
    print("  ~40 weak · ~160 default · ~400 strong")
    raw = input(f"Set playouts (blank keeps {_PLAY_PLAYOUTS}): ").strip()
    if raw:
        try:
            _PLAY_PLAYOUTS = max(8, int(raw))
            print(f"Play strength set: {_PLAY_PLAYOUTS} playouts.")
        except ValueError:
            print("Unchanged.")
    print("(Full temperature→Elo calibration + human-tracking difficulty: elo-calibration follow-on.)")


def top_menu():
    """:Top-menu: — the four spec-governed modes."""
    while True:
        print("\n=========  Chess-RL  =========")
        print("1. Play        — human vs the net+PUCT RL agent (or engine baseline)")
        print("2. Train       — PUCT self-play (the :Stage-controller:)")
        print("3. Measure     — agent vs the Elo ladder")
        print("4. Difficulty  — set play strength")
        print("5. Exit")
        c = input("Choose (1-5): ").strip()
        if c == "1":
            play_mode()
        elif c == "2":
            train_mode()
        elif c == "3":
            measure_mode()
        elif c == "4":
            difficulty_mode()
        elif c == "5":
            print("bye.")
            break
        else:
            print("Enter 1-5.")


def main():
    top_menu()


if __name__ == "__main__":
    main()
