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

- :Stage-controller: is the orchestrator of the three training stages — Stage 1 teacher distillation, Stage 2 annealed off-policy refinement, Stage 3 expert iteration — whose transitions are gated exclusively by :Elo-gate:s, never by wall-clock or game count. MEASURED RECONCILIATION (RL_FINDINGS): of the three, **Stage-3 batched-PUCT self-play** (the AlphaStar-hybrid, `puct_selfplay.py`) is the demonstrated climber (reached parity with heuristic-1ply); Stage-1 Stockfish distillation is a valid optional WARM-START; the Stage-2 off-policy and linear-value paths were tried and plateaued. So `train_control.py` defaults :Train-mode: to Stage-3 PUCT self-play; the full gated multi-stage sequence stays the contract but is honest that only the PUCT stage has shown climb on this hardware.
- :Elo-trend: is the persisted per-run sequence of (dataset size, training volume, :Measured-elo:) points — the primary monitored signal across runs; loss curves are secondary diagnostics only.
- :Self-play-game: is one game the agent plays against itself, producing a stream of :Replay-transition:s (Stage 2) or expert-iteration targets (Stage 3) and a set of per-game monitoring metrics.
- :Teacher-agreement: is the fraction of moves in a game where the played move equals the operative :Prior-lineage: member's greedy move — a lock-in / derivative-play signal that should fall as the learner outgrows its prior. (Generalizes the earlier prior-agreement metric.)
- :Opening-diversity: is the count of distinct first moves seen across recent games — a policy-collapse signal that should stay above one.
- :Ahead-but-lost: is the count of games the mover led by a wide evaluation margin yet did not win — a reward-hacking signal.
- :Stop-training: is the three-gate rule for ending a training stage. adv = G − V(s); the baseline drives its mean toward 0, so E[adv²] is Bellman-residual energy — a cheap proxy for critic/policy fit ON THE SELF-PLAY DISTRIBUTION, not strength. Low E[adv²] is FOUR-way ambiguous: nothing left to learn (victory), entropy collapse (the policy stopped visiting surprising states), a frozen distribution, or the critic having memorised/overfit the self-play data — and three of the four mimic convergence. So a stage MUST require ALL of: a robust (median/Huber, mate-spike-resistant — raw MAD is insufficient) EWMA of E[adv²] below θ for K consecutive checkpoints, AND the external :Measured-elo: rung plateaued (the expensive probe that alone catches overfit), AND policy entropy at or above a floor defined so it holds :Opening-diversity: > 1 (rejecting collapse). A DISTINCT stop retires the search arm, not training: when the beam's measured excess over the zero-search policy (:Compute-frontier: advantage vs the bare-policy rfr) reaches 0, search is distilled into the policy and ε may retire.
- :Value-of-information: is the law unifying every stop/allocate decision in the system — spend compute where estimates still disagree, stop where they have stopped disagreeing. adv-variance measures disagreement across STATES (training), :Search-window-reuse:'s δ across PASSES (per line), :Move-margin: across SIBLINGS (per move), :Phi-rotation:'s ordering-stability across LAYERS (width). One statistic at four scopes; it is the epistemic (uncertainty-reduction) term of expected free energy.

***implementation reqs***

- The loop lives in `train_control.py` (the DQN `chess_ai.py` is retired to `legacy/`); the :Stage-controller: reads :Measured-elo: from the measurement authority and computes :Gated-progress: for the schedule (the schedule itself stays pure).
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
- :Stop-training: must require all three gates; any single gate alone MUST NOT stop a stage.
  - Given robust-EWMA(E[adv²]) < θ for K checkpoints but the rung :Measured-elo: still climbing, Then training continues (the cheap proxy saturated, capability did not).
  - Given robust-EWMA(E[adv²]) < θ and the rung eval plateaued but policy entropy below its floor (:Opening-diversity: at 1), Then STOP is refused and collapse is flagged — low variance here is collapse, not convergence.
  - Given all three hold (low robust adv² for K checkpoints, rung plateau, entropy ≥ floor), Then the stage may stop.
  - Given the beam's :Compute-frontier: advantage over the zero-search policy reaches 0, Then the search/ε arm retires (search distilled into the policy), independently of the training-stop gates.
- The :Value-of-information: law must be the shared rationale for every allocate/stop decision: compute is spent where estimates still disagree (high adv-variance states, unstable :Phi-rotation: layers, low :Move-margin: moves, :Delta:-divergent lines) and withdrawn where they agree.
