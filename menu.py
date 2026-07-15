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
    """Human vs a :Selectable-agent: through the terminal-interface front-end — the ONE
    board, always (operator 2026-07-14). Fail-fast: no silent REPL downgrade — the old
    catch-all swallowed a mid-game error, reset the game, and switched boards."""
    from agents import make_agent, AgentAdapter
    name = _pick_agent()
    label, move_fn, value_fn = make_agent(name, playouts=_PLAY_PLAYOUTS,
                                          depth=_DIFFICULTY["depth"])
    if move_fn is None:
        return
    settings, calibrator = None, None
    if name == "champion" and _DIFFICULTY["choice"] != "full":
        calibrator = _load_calibrator()
        if _DIFFICULTY["choice"] == "fixed":
            settings = {"enabled": True, "mode": "fixed",
                        "fixed_temperature": _DIFFICULTY["tau"]}
            print(f"[difficulty] fixed temperature {_DIFFICULTY['tau']:.2f} "
                  f"(≈{calibrator.policy_elo(_DIFFICULTY['tau']):.0f} Elo)")
        else:
            settings = {"enabled": True, "mode": "auto",
                        "offset_sdev": _DIFFICULTY.get("sigma", 1.0)}
            print(f"[difficulty] dynamic — the opponent will track "
                  f"~{_DIFFICULTY.get('sigma', 1.0):g} sdev above your play")
    human_white = (input("Play as white? (y/n, default y): ").strip().lower() or "y") == "y"
    from terminal_board import TerminalChessBoard
    adapter = AgentAdapter(move_fn, value_fn, elo_calibrator=calibrator,
                           difficulty_settings=settings)
    human_color = chess.WHITE if human_white else chess.BLACK
    TerminalChessBoard(chess.Board(), adapter, human_color=human_color).start()


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


def _load_calibrator():
    """:Absolute-strength-dial: source — measured elo_calibration.json only if its taus
    actually SEPARATE (the shipped file is a degenerate placeholder: all gaps equal);
    otherwise the :Approximate-elo-curve: default anchored at the champion's measured
    d2 band (~1183; flagged approximate/assumed in the readout, per spec)."""
    import os
    from elo_calibration import TemperatureEloCalibrator
    from constants import ELO_CALIBRATION_PATH
    if os.path.exists(ELO_CALIBRATION_PATH):
        try:
            cal = TemperatureEloCalibrator.load(ELO_CALIBRATION_PATH)
            gaps = {round(v["elo_gap"]) for t, v in cal.table.items() if t > 1e-6}
            if len(gaps) > 1:
                return cal
        except Exception:
            pass
    return TemperatureEloCalibrator(anchor_elo=1183, anchor_measured=False)


def _tau_for_elo(cal, target):
    """Inverse dial: temperature whose curve Elo is closest to the target."""
    grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0]
    return min(grid, key=lambda t: abs(cal.policy_elo(t) - target))


# :Difficulty-mode: state consumed by play_mode when the champion is selected.
_DIFficulty_DEFAULT = {"choice": "dynamic", "tau": None, "depth": 9}
_DIFFICULTY = dict(_DIFficulty_DEFAULT)


def difficulty_mode():
    """:Difficulty-mode: — champion strength dial (dynamic / fixed Elo / full) + the
    legacy PUCT playouts knob."""
    global _PLAY_PLAYOUTS, _DIFFICULTY
    print("\nChampion difficulty:")
    print("  1. dynamic (default) — tracks YOUR play, targets ~1 sdev above your level")
    print("  2. fixed Elo        — pick a strength and it stays there")
    print("  3. full strength    — the 1670-claims engine (d9 argmax)")
    c = input("Choose (1-3, default 1): ").strip() or "1"
    if c == "2":
        cal = _load_calibrator()
        tiers = [800, 1000, 1183, 1400, 1572, 1670]
        print("  fixed tiers: " + "  ".join(f"{i+1}.~{e}" for i, e in enumerate(tiers)))
        k = input("Tier (1-6, default 3): ").strip() or "3"
        target = tiers[max(0, min(5, int(k) - 1))] if k.isdigit() else 1183
        if target >= 1670:
            _DIFFICULTY = {"choice": "full", "tau": None, "depth": 9}
        elif target >= 1572:
            _DIFFICULTY = {"choice": "full", "tau": None, "depth": 7}
        else:
            tau = _tau_for_elo(cal, target)
            _DIFFICULTY = {"choice": "fixed", "tau": tau, "depth": 9}
            print(f"fixed strength ≈{target} Elo -> temperature {tau:.2f} "
                  f"({'measured' if not cal.approximate else 'approximate'} curve)")
    elif c == "3":
        _DIFFICULTY = {"choice": "full", "tau": None, "depth": 9}
    else:
        _DIFFICULTY = dict(_DIFficulty_DEFAULT)
        raw = input("Target sigma above your level (default 1.0; 1.5-2 = more pain): ").strip()
        try:
            if raw:
                _DIFFICULTY["sigma"] = max(0.0, float(raw))
        except ValueError:
            pass
    print(f"champion difficulty: {_DIFFICULTY['choice']}"
          + (f" (tau {_DIFFICULTY['tau']:.2f})" if _DIFFICULTY["tau"] else "")
          + (f" (d{_DIFFICULTY['depth']})" if _DIFFICULTY["choice"] == "full" else ""))
    raw = input(f"PUCT playouts (blank keeps {_PLAY_PLAYOUTS}; only affects the puct agent): ").strip()
    if raw:
        try:
            _PLAY_PLAYOUTS = max(8, int(raw))
            print(f"PUCT playouts: {_PLAY_PLAYOUTS}.")
        except ValueError:
            print("Unchanged.")


def played_buffer_mode():
    """:Played-buffer: — fine-tune a champion COPY on archived human games (labels stay
    self-generated: own d2 search values + outcome, proven trivium blend). Promotion is
    duel-gated, never automatic."""
    import glob as _glob
    import subprocess
    import sys as _sys
    n = len(_glob.glob("data/human_games/*.pgn"))
    print(f"\nPlayed buffer: {n} archived game(s) (finished games auto-archive).")
    if not n:
        print("Play and finish a game first — it archives itself.")
        return
    if (input(f"Fine-tune champion copy on {n} game(s)? (y/n, default y): ").strip().lower() or "y") != "y":
        return
    subprocess.run([_sys.executable, "human_replay.py"], cwd=".")
    print("Candidate: models/champion_hb.pt — verdict duel (600g vs champion):")
    print("  python head2head.py kcz:models/champion_hb.pt kcz:models/champion.pt 600 hb_verdict")
    if (input("Run the verdict duel now? (y/n, default n): ").strip().lower()) == "y":
        subprocess.run([_sys.executable, "head2head.py", "kcz:models/champion_hb.pt",
                        "kcz:models/champion.pt", "600", "hb_verdict"], cwd=".")


def top_menu():
    """:Top-menu: — the four spec-governed modes."""
    while True:
        print("\n=========  Chess-RL  =========")
        print("1. Play        — human vs the CHAMPION (or roster agents)")
        print("2. Train       — PUCT self-play (the :Stage-controller:)")
        print("3. Measure     — agent vs the Elo ladder")
        print("4. Difficulty  — set play strength (dynamic tracks you)")
        print("5. Learn from my games — :Played-buffer: fine-tune (duel-gated)")
        print("6. Exit")
        c = input("Choose (1-6): ").strip()
        if c == "1":
            play_mode()
        elif c == "2":
            train_mode()
        elif c == "3":
            measure_mode()
        elif c == "4":
            difficulty_mode()
        elif c == "5":
            played_buffer_mode()
        elif c == "6":
            print("bye.")
            break
        else:
            print("Enter 1-6.")


def main():
    top_menu()


if __name__ == "__main__":
    main()
