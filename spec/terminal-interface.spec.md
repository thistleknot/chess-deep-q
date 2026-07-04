---
description: 'The terminal human-vs-computer front-end: move entry, in-game commands, and the per-turn board readout'
import:
  - dynamic-difficulty
  - elo-calibration
---

***definitions***

- :Move-entry: is how the human submits a move at the prompt: a UCI string of 2 characters (a from-square, e.g. `e2`, which selects a piece and previews its legal moves) or 4 characters (a complete move, e.g. `e2e4`). Length disambiguates entry from a :Game-command: — every command token is a single letter or a full word, never a 2- or 4-character square string, so the two input classes never collide.
- :Game-command: is a non-move action typed at the same prompt. Each command has a canonical full word and a first-letter shortcut, and BOTH are accepted equivalently: `(h)int`, `(u)ndo`, `(s)ave`, `(l)oad`, `(r)esign`, `(c)ancel`. The shortcut is the word's first letter; the set is collision-free (distinct first letters) and disjoint from :Move-entry: (single letters are never legal squares).
- :Board-readout: is the header printed above the board every time it is rendered: the side to move, the position eval from the fast evaluator (the advantage/disadvantage number), the :Player-score-breakdown:, and — when a :Difficulty-controller: opponent is active — the :Estimated-elo-readout: (opponent + player Elo, labeled approximate or measured).
- :Player-score-breakdown: is a per-side table in the header showing each player's Pieces score (summed material value) and Position score (weighted positional terms: mobility, square control, king safety, pawn structure, space, coordination), plus their sum. It uses the same component weights as the fast evaluator, so White's total minus Black's total equals the position eval (excluding the turn-dependent check bonus). It gives the human a side-by-side view of where the advantage comes from, decomposing the single eval number into material vs. position for each color.
- :Side-selection: is the new-game setup prompt where the human picks their color before play begins (`Play as white or black? (w/b, default: w)`). White is the DEFAULT: empty input, or any input that does not explicitly request black, yields White; Black is chosen only by an explicit black token (a value starting with `b`). This holds uniformly across every entry path (fresh game, load-game, play-from-FEN).

***implementation reqs***

- `terminal_board.py` owns the terminal front-end: `process_input` parses :Move-entry: and :Game-command:; `display_board` renders the :Board-readout:.
- Command dispatch MUST match each :Game-command: against its full word OR its first-letter shortcut (e.g. `command in ('hint', 'h')`), so the on-screen legend and the parser stay in sync.
- The command legend printed to the human MUST show the shortcut form (e.g. `(h)int`) so the available shortcuts are discoverable without documentation.
- `menu.py` owns :Side-selection:. Every prompt that maps a color choice to `human_color` MUST resolve to `chess.WHITE` unless the input explicitly requests black — i.e. select `chess.BLACK` only on an explicit black token and default to `chess.WHITE` for empty or unrecognized input. No path may fall through to Black.
- The :Estimated-elo-readout: MUST be rendered inside `display_board`'s header — once per turn, on the persistent board view — NOT as a transient line that scrolls away after a move.
- The :Player-score-breakdown: MUST be computed by `evaluate_by_player` in `evaluation.py` and rendered inside `display_board`'s header directly below the position eval. `evaluate_by_player` MUST reuse the fast evaluator's material values and component weights so that `white['total'] - black['total']` reconstructs `fast_evaluate_position` (up to the check bonus), keeping the breakdown and the headline eval consistent.

***test reqs***

- A dispatch table asserting that each full word and each single-letter shortcut route to the same action, and that a 2- and 4-character UCI string is parsed as :Move-entry:, not a command.
- An invariant test over several positions (startpos, a mid-game position) asserting `evaluate_by_player(board)['white']['total'] - ...['black']['total']` equals `fast_evaluate_position(board)` for positions not in check, and that each side's `material + position == total`.

***functional specs***

- Every :Game-command: must accept its full word and its first-letter shortcut identically.
  - Given the input `hint` or `h`, Then a hint is shown; likewise `u`=undo, `s`=save, `l`=load, `r`=resign, `c`=cancel.
  - Given a 2- or 4-character legal UCI string, Then it is treated as :Move-entry:, never as a command (no shortcut shadows a square).
- :Side-selection: defaults to White.
  - Given the :Side-selection: prompt, When the human presses Enter with no input, Then they play White.
  - Given any input that does not explicitly request black (e.g. `w`, `white`, a stray token), Then they play White.
  - Given an explicit black token (a value starting with `b`, e.g. `b` or `black`), Then they play Black.
  - This behavior is identical across the fresh-game, load-game, and play-from-FEN entry paths.
- The :Board-readout: must show the :Player-score-breakdown: every turn.
  - Given the board is rendered, Then the header shows, for both White and Black, a Pieces score, a Position score, and their total.
  - Given a position, When the breakdown is computed, Then `white['total'] - black['total']` equals `fast_evaluate_position` for that position (excluding the turn-dependent check bonus), so the per-player totals stay consistent with the single eval number.
- The :Board-readout: must show the estimated Elo every turn while an adjusting/fixed opponent is active.
  - Given a :Difficulty-controller: opponent (auto or fixed) is enabled, When the board is rendered, Then the header shows the :Estimated-elo-readout: alongside the position eval.
  - Given the opponent is at full strength (:Difficulty-controller: disabled), Then only the position eval is shown and no Elo readout is required.
  - Given the active curve is approximate, When the readout is shown, Then it is labeled approximate (never presented as measured).
