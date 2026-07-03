"""Temperature -> ELO calibration, and a regret -> ELO index for scoring the human player.

The idea (per the design discussion): each policy strength has an average ELO. We measure it by
self-play -- the model at argmax (full strength) vs the model at temperature tau -- and turn the
win rate into an ELO gap below full strength. During the SAME calibration we also record the
average per-move regret the weak (tau) policy exhibits. That yields two aligned tables:

    tau  ->  elo_gap        (how much weaker than full strength)
    tau  ->  avg_regret     (how far from best the weak policy plays)

Composed, they give regret -> ELO. So the human player's measured regret against the policy maps
onto the SAME ELO index as the policy itself -- "how well are you playing against the policy" is
read directly as an ELO. Because the difficulty controller matches the opponent to the player,
the opponent's ELO (from tau) and the player's ELO (from regret) should converge as a game goes on.

All ELO numbers are estimates -- noisy for a lightly trained net or short games -- and are labeled
as such in the UI.
"""

import json
import math

import chess

from difficulty import move_regret
from constants import (
    ELO_CALIBRATION_TAUS, ELO_CALIBRATION_GAMES, ELO_CALIBRATION_MAX_MOVES,
    ELO_CALIBRATION_ITERATIONS, ASSUMED_ARGMAX_ELO,
)


def score_to_elo_diff(score, cap=800.0):
    """ELO difference implied by a win-rate score in (0,1)."""
    score = min(max(score, 1e-4), 1.0 - 1e-4)
    return max(-cap, min(cap, -400.0 * math.log10(1.0 / score - 1.0)))


def _interp(x, xs, ys):
    """Linear interpolation of y at x over sorted xs (clamped at the ends)."""
    if not xs:
        return None
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


class TemperatureEloCalibrator:
    """Holds the tau->(elo_gap, avg_regret) table and derives policy/player ELO."""

    def __init__(self, anchor_elo=ASSUMED_ARGMAX_ELO, anchor_measured=False):
        self.anchor_elo = anchor_elo          # full-strength (argmax) policy ELO
        self.anchor_measured = anchor_measured  # True if measured (e.g. vs Stockfish), else assumed
        # Start from an approximate heuristic curve so the ELO shows instantly; calibrate() then
        # replaces it with measured values and clears the approximate flag.
        self.approximate = True
        self.table = self._default_table()     # tau -> {"elo_gap":..., "avg_regret":...}

    @staticmethod
    def _default_table():
        """Heuristic temperature->ELO-gap (and rough regret) curve, used before real calibration.

        Both grow monotonically with temperature: higher tau = weaker play = larger ELO gap below
        full strength and larger (more negative) regret. Values are rough placeholders, clearly
        labeled approximate in the UI until calibrate() measures them.
        """
        gap_max, tau_scale = 400.0, 1.0
        table = {}
        for tau in ELO_CALIBRATION_TAUS:
            gap = gap_max * (1.0 - math.exp(-tau / tau_scale))
            regret = -0.3 * (1.0 - math.exp(-tau / tau_scale))
            table[tau] = {"elo_gap": gap, "avg_regret": regret}
        return table

    # --- calibration -------------------------------------------------------
    def calibrate(self, ai, taus=None, games_per_tau=None, max_moves=None,
                  iterations=ELO_CALIBRATION_ITERATIONS, log=print):
        """Play argmax vs each temperature; record ELO gap and the weak side's avg regret.

        iterations caps the MCTS search per move so calibration finishes quickly; the relative
        strength ordering across temperatures is robust to shallow search.
        """
        taus = taus if taus is not None else ELO_CALIBRATION_TAUS
        games_per_tau = games_per_tau or ELO_CALIBRATION_GAMES
        max_moves = max_moves or ELO_CALIBRATION_MAX_MOVES
        for tau in taus:
            if tau <= 1e-6:
                self.table[tau] = {"elo_gap": 0.0, "avg_regret": 0.0}
                log(f"tau={tau:.2f}: full strength (gap 0)")
                continue
            score, avg_regret = self._match(ai, weak_tau=tau, games=games_per_tau,
                                            max_moves=max_moves, iterations=iterations, log=log)
            gap = score_to_elo_diff(score)
            self.table[tau] = {"elo_gap": gap, "avg_regret": avg_regret}
            log(f"tau={tau:.2f}: strong win-rate={score:.2f} elo_gap={gap:.0f} "
                f"avg_regret={avg_regret:.3f}")
        self.approximate = False   # table is now measured, not the heuristic default
        return self.table

    def _match(self, ai, weak_tau, games, max_moves, iterations=ELO_CALIBRATION_ITERATIONS,
               log=print):
        """Strong (argmax) vs weak (weak_tau); return (strong_score, weak_avg_regret)."""
        value_fn = ai.dqn_agent.get_q_value
        strong_points = 0.0
        regrets = []
        for g in range(games):
            board = chess.Board()
            strong_is_white = (g % 2 == 0)
            moves_played = 0
            while not board.is_game_over() and moves_played < max_moves:
                ai.board = board
                strong_to_move = (board.turn == chess.WHITE) == strong_is_white
                temperature = 0.0 if strong_to_move else weak_tau
                state_before = board.copy()
                move = ai.get_best_move(temperature=temperature, max_moves=max_moves,
                                        iterations=iterations)
                if move is None:
                    break
                if not strong_to_move:
                    # Regret of the weak side's move vs the search's argmax best (free from
                    # the same search via last_root_best).
                    best = ai.dqn_agent.last_root_best
                    regret, _ = move_regret(state_before, move, value_fn, best)
                    regrets.append(regret)
                board.push(move)
                moves_played += 1
            result = board.result()
            if result == "1-0":
                strong_points += 1.0 if strong_is_white else 0.0
            elif result == "0-1":
                strong_points += 1.0 if not strong_is_white else 0.0
            else:
                strong_points += 0.5
        avg_regret = sum(regrets) / len(regrets) if regrets else 0.0
        return strong_points / games, avg_regret

    # --- lookups -----------------------------------------------------------
    def _sorted_taus(self):
        return sorted(self.table)

    def policy_elo(self, tau):
        """ELO of the policy playing at temperature tau (anchor - gap)."""
        taus = self._sorted_taus()
        if not taus:
            return None
        gap = _interp(tau, taus, [self.table[t]["elo_gap"] for t in taus])
        return self.anchor_elo - gap

    def player_elo(self, regret):
        """ELO implied by a player's average regret, on the same index as the policy.

        Regret is <= 0 (0 = played best; more negative = weaker). We invert the aligned
        avg_regret -> elo_gap relationship (both monotonic in tau) and subtract from the anchor.
        """
        taus = self._sorted_taus()
        if not taus:
            return None
        # Build regret -> elo_gap, sorted by ascending regret (more negative first).
        pairs = sorted(((self.table[t]["avg_regret"], self.table[t]["elo_gap"]) for t in taus),
                       key=lambda p: p[0])
        regrets = [p[0] for p in pairs]
        gaps = [p[1] for p in pairs]
        gap = _interp(regret, regrets, gaps)
        return self.anchor_elo - gap

    def is_calibrated(self):
        return len(self.table) >= 2

    # --- persistence -------------------------------------------------------
    def save(self, path):
        with open(path, "w") as fh:
            json.dump({
                "anchor_elo": self.anchor_elo,
                "anchor_measured": self.anchor_measured,
                "approximate": self.approximate,
                "table": {str(k): v for k, v in self.table.items()},
            }, fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            data = json.load(fh)
        cal = cls(anchor_elo=data.get("anchor_elo", ASSUMED_ARGMAX_ELO),
                  anchor_measured=data.get("anchor_measured", False))
        if data.get("table"):
            cal.table = {float(k): v for k, v in data["table"].items()}
            cal.approximate = data.get("approximate", False)
        return cal
