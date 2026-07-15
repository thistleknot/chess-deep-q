---
description: 'The single entrypoint contract — main.py boots the spec-governed RL system: Play / Train / Measure / Difficulty, each mode traced to its owning spec. Fills the gap where the top-level menu previously traced to no spec and ran the retired DQN.'
import:
  - chess-rl
  - terminal-interface
  - training-loop
  - elo-measurement
  - dynamic-difficulty
  - self-play-leela
---

***definitions***

- :Entrypoint: is `main.py` — the SOLE entrypoint to the :Chess-RL-system:. It boots directly into the :Top-menu: (no DQN goal-picker). Every action it exposes traces to a governing spec; the retired DQN (`legacy/`) is NOT reachable from :Entrypoint:. This spec exists because the previous numbered menu (1-21) lived only in `menu.py` and traced to NO spec.
- :Top-menu: is the four-mode top level — :Play-mode:, :Train-mode:, :Measure-mode:, :Difficulty-mode: (plus exit) — the durable contract that replaces the stale DQN goal-picker + flat `.train()` loop + un-spec'd AHA options.
- :Selectable-agent: is the agent factory (`agents.py` `make_agent(name) -> (label, move_fn)`): `champion` (DEFAULT since 2026-07-14 — the self-play RL net inside native alpha-beta d9, claims 1670 (1605..1762), `models/champion.pt` via the ZCA-gated conversion; spec/bullet-route.spec.md), `puct` (the net+PUCT arm), `engine` (the alpha-beta baseline), `beam` (experimental), `nnue` (phi-widen deployed arm). SHARED by :Play-mode: and :Measure-mode:.
- :Play-mode: is a human game against a :Selectable-agent:, rendered through the terminal front-end governed by `terminal-interface.spec.md` (`terminal_board.py`: :Move-entry:, :Game-command:, :Board-readout:, White-default :Side-selection:). DEFAULT agent = CHAMPION (the goal-met RL deliverable, 1670 claims — superseding net+PUCT as default 2026-07-14; PUCT remains selectable). The alpha-beta `engine.py` is an OPTIONAL opponent (a baseline to SURPASS per chess-rl :Measured-disposition:), never the default. Operator law: new deliverables are slotted HERE (spec/bakeoff.spec.md :Replication-requirement: — the deliverable must be reachable from the canonical UI), never as parallel scripts.
- :Train-mode: invokes the :Stage-controller: (`train_control.py`, training-loop.spec.md) — the staged, Elo-gated pipeline, NOT a flat game-count loop. Default approach = Stage-3 batched-PUCT self-play (the measured climber); honors the :Run-contract: (checkpoint per run).
- :Measure-mode: runs :Measured-elo: (elo-measurement) — the :Selectable-agent: on the :Ladder: (random / heuristic-1ply / SF@1320). It is the gate authority feeding :Train-mode: transitions and :Difficulty-mode: calibration; the spec-clean successor of the old "Evaluate ELO" option.
- :Difficulty-mode: sets the :Absolute-strength-dial: (:Temperature-elo-curve:, elo-calibration) and toggles the :Difficulty-controller: (dynamic-difficulty) that tracks the seated human — dialing the ONE trained net across the human range, feeding the :Estimated-elo-readout:.

***implementation reqs***

- `main.py` is a thin boot (`setup_environment` → :Top-menu:); the menu lives in `menu.py`. NEW files: `agents.py` (:Selectable-agent: factory), `train_control.py` (:Stage-controller: home, replacing `chess_ai.py`), `play_beam.py` (re-homed from `chess_ai.get_beam_move`).
- No module reachable from :Entrypoint: may import the retired DQN core (`legacy/chess_ai.py`, `legacy/neural_network.py`, `legacy/mcts.py`). The DQN-era AHA options and the flat `.train()` loop are REMOVED — neither has a governing spec.
- Every :Top-menu: action MUST trace to a spec (this file maps each mode to its owner).

***functional specs***

- :Entrypoint: must boot straight into the :Top-menu:.
  - Given `python main.py`, Then the :Top-menu: (Play / Train / Measure / Difficulty / exit) is shown; no DQN goal-picker precedes it.
- :Play-mode: must default to net+PUCT through the terminal-interface front-end.
  - Given :Play-mode: with no explicit agent choice, Then the opponent is net+PUCT; the engine is offered only as an explicit optional opponent; the game runs through `terminal_board.py` with White-default :Side-selection: identical across fresh / load / FEN (fixing the FEN path that hardcoded White).
- :Train-mode: must call the :Stage-controller:, not a flat loop.
  - Given :Train-mode:, Then `train_control.train(approach)` runs the gated staged pipeline (default PUCT self-play) and checkpoints per the :Run-contract:.
- :Measure-mode: must place the :Selectable-agent: on the :Ladder: and report :Measured-elo:.
- :Difficulty-mode: must dial the trained net via the :Absolute-strength-dial:, never the DQN.
