---
description: 'The staged training loop: gate-driven stage control, schedule application, reward assignment, and failure-mode monitoring'
import:
  - elo-measurement
  - annealing-schedule
  - prior-evaluator
  - learned-model
  - teacher-distillation
  - search-mcts
  - value-target
  - self-play-leela
---

***definitions***

- :Stage-controller: is the orchestrator of the three training stages — Stage 1 teacher distillation, Stage 2 annealed off-policy refinement, Stage 3 expert iteration — whose transitions are gated exclusively by :Elo-gate:s, never by wall-clock or game count.
- :Elo-trend: is the persisted per-run sequence of (dataset size, training volume, :Measured-elo:) points — the primary monitored signal across runs; loss curves are secondary diagnostics only.
- :Self-play-game: is one game the agent plays against itself, producing a stream of :Replay-transition:s (Stage 2) or expert-iteration targets (Stage 3) and a set of per-game monitoring metrics.
- :Teacher-agreement: is the fraction of moves in a game where the played move equals the operative :Prior-lineage: member's greedy move — a lock-in / derivative-play signal that should fall as the learner outgrows its prior. (Generalizes the earlier prior-agreement metric.)
- :Opening-diversity: is the count of distinct first moves seen across recent games — a policy-collapse signal that should stay above one.
- :Ahead-but-lost: is the count of games the mover led by a wide evaluation margin yet did not win — a reward-hacking signal.

***implementation reqs***

- The loop lives in `chess_ai.py`; the :Stage-controller: reads :Measured-elo: from the measurement authority and computes :Gated-progress: for the schedule (the schedule itself stays pure).
- Monitoring metrics are stored per game in the existing game-history record and plotted alongside loss and :Elo-trend:s.
- Any concurrent data generation (labelling or self-play) and torch training occupy separate OS processes (:Process-separated-labeling: generalized).

***functional specs***

- Stage transitions must be gated, not scheduled.
  - Given Stage 1 has not cleared the 1200 :Elo-gate:, Then Stage 2 self-play MUST NOT start (self-play data from a weak policy is noise, not signal).
  - Given a gate has not cleared, Then :Gated-progress: clamps at its segment boundary and all schedule coefficients hold.
- Every run must obey the :Run-contract: regardless of stage.
  - Given any run ends, Then checkpoint, dataset, and an :Elo-trend: point persist, so the next ≤5-minute run resumes cumulatively.
- :Self-play-game: must compute :Gated-progress: once at game start and drive all schedule reads from it.

  Input: board — the game position, chess.Board
  Parameters: max_moves ∈ ℤ⁺
  Initialize: progress ← stage-controller's :Gated-progress:   # global to this game, clamped to [0,1]
  Initialize: beta ← schedule.bootstrap_share(progress)        # value-target :Bootstrap-share:, λ = 1 − beta
  Initialize: teacher_hits ← 0                                  # global to this game

  Loop while board is not game-over and move_count < max_moves:
      state_before ← copy of board                               # transient
      move ← search selects a move at progress                   (Require: move is legal)
      record whether move equals the operative prior's greedy move into teacher_hits
      push move onto board
      When board is checkmate:
          reward ← +1 if the checkmated side is the opponent else -1   # White-absolute
          done ← true
      Otherwise When board is stalemate or insufficient material:
          reward ← 0
          done ← true
      Otherwise:
          reward ← shaped_reward_weight(progress) × (γ·Φ(board) − Φ(state_before))   # potential-based (Ng '99); Φ = normalized operative-prior score
          done ← false
      store the transition with its value target as the :Lambda-return: at λ = 1 − beta, bootstrapped
        from the step's :Search-value: (Stage 2 transition / Stage 3 expert-iteration targets)
      When buffer has at least BATCH_SIZE samples:
          run one training step honoring :Demo-share: and append its loss
  Assert: every stored terminal reward is one of {+1, 0, -1}

  Given White mates on the final move, When the game ends, Then the last stored reward MUST be +1.
  Given late-game progress, When a quiet move is scored, Then shaped reward magnitude SHOULD be small versus a terminal ±1.

- :Teacher-agreement: must be derived as teacher_hits / move_count and stored on the game record.
  - Given a game where every move matched the operative prior, Then :Teacher-agreement: is 1.0 (maximal lock-in warning).
- :Opening-diversity: must be updated from first moves across games and stored for plotting.
  - If :Opening-diversity: stays at 1 across many games, Then policy collapse is flagged for the user.
- :Ahead-but-lost: must be incremented when a side held a wide evaluation lead yet failed to win.
- Monitoring metrics must be observable: the training-plot step must render :Teacher-agreement:, :Opening-diversity:, and the :Elo-trend: to disk after training.
