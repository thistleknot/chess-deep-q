---
description: 'PUCT tree search with batched leaf evaluation, lineage-blended leaves, and two regression-pinned sign/selection conventions'
import:
  - annealing-schedule
  - prior-evaluator
  - learned-model
  - teacher-distillation
---

***definitions***

- :PUCT-search: is the tree search that selects child actions by Q + c_puct · P · √N/(1+n), where P is the policy prior; the Russian-doll progressive-narrowing search is retained as the small-budget profile of the same contract.
- :Leaf-value: is the estimate assigned to a search-leaf position, formed by blending the operative :Prior-lineage: member with the :Learned-value: by :Teacher-leaf-weight:.
- :Negamax-backup-convention: is the sign rule for backing a leaf value up the path: leaf values are produced White-absolute, MUST be converted to side-to-move-relative at the leaf, and MUST be sign-flipped BEFORE each parent update so each node's Q is always the mover's value at that node. (Regression pin: the backup previously left the leaf White-absolute and flipped after the first update, corrupting Q for Black-to-move nodes and for any simulation whose depth parity put White at the deepest decision — the net-as-Black collapsed in 9–13 plies.)
- :Root-selection-rule: is the budget-aware root move choice: when visits-per-child are below MIN_ROOT_VISITS, the root move MUST be argmax mean-Q over *visited* children; visit-count argmax is permitted only above the threshold. (Regression pin: at ~200 simulations over ~21 root moves every child got ~10–13 visits — counts too flat to separate a free-queen capture with Q=0.93 from quiet moves with Q~0.6, so the search ignored the free queen.)
- :Batched-leaf-evaluation: is the requirement that pending leaves are collected (wave collection or virtual loss) and evaluated in one network batch per wave, exploiting the batch-evaluate API; per-leaf single inference is non-conformant (~2–3 ms each vs 18k–122k/s batched).
- :Progressive-narrowing: is the per-level reduction of sampled move breadth (many candidates near the root, few deep), the small-budget profile's focusing mechanism.
- :Search-value: is the negamax-backed value estimate at a node after its backups — the value the search already computed to select a move. It is exposed in the :White-absolute-frame: so the value-target stage can bootstrap from it for free (no extra inference), and is the improved on-policy value that makes the value target off-policy-safe (tree-backup).
- :Quiescence-search: is the requirement that the net-minimax profile evaluates a leaf ONLY at a quiet position: at a fixed-depth frontier that is non-quiet (mid-capture sequence, hanging piece, in check), the search extends captures (and checks) until quiet before applying the value net. Without it, fixed-depth-2 hands the net non-quiet positions where any static eval is noise — the horizon effect that, not eval quality, caps net-minimax below a quiescence-equipped alpha-beta of *worse* eval. The primary profile also needs the standard stack: iterative deepening, a transposition table, and move ordering.
- :Policy-move-ordering: is the consumer of the :Policy-head: inside the net-minimax profile (minimax does not otherwise read P): the policy distribution orders moves for search and selects the beam. This closes a dead-weight defect (the policy head would otherwise be unused in the primary profile) and makes cloning self-reinforcing — a better policy yields better ordering, hence deeper effective search, hence better cloning targets.

***implementation reqs***

- The search lives in `mcts.py`; it reads coefficients from the :Annealing-schedule:, it does not compute its own schedule inline.
- Constant: MIN_ROOT_VISITS — the visits-per-child threshold of the :Root-selection-rule:.
- :Gated-progress: must be threaded to the leaf evaluation, not stopped at the search entry point.
- `net_search.py` (batched fixed-depth minimax with beam pruning, ~0.9 s/move at depth 2) is the interim conformant fast profile until PUCT lands.

***test reqs***

- The historical failing position — opponent has just left a queen en prise (e.g. `rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w`) — with a ~200-simulation budget, to assert the :Root-selection-rule:.
- A mirror pair of positions backed up through alternating plies, to assert the :Negamax-backup-convention: for both colors.
- A midgame position plus a stub network returning a fixed value, to assert the leaf blend endpoints.

***functional specs***

- The :Negamax-backup-convention: must hold on every backup.
  - Given a leaf value backed up through alternating plies, Then the sign presented to each node is that node's mover-relative value (flip BEFORE each update, starting from a side-to-move-relative leaf).
- The :Root-selection-rule: must hold at small budgets.
  - Given the free-queen test position and a ~200-simulation budget, When the search completes, Then the capture is the root choice.
  - Given visits-per-child below MIN_ROOT_VISITS, Then unvisited children are ineligible and the choice is argmax mean-Q; Given visits above the threshold, Then visit-count argmax is permitted.
- :Batched-leaf-evaluation: must bound network calls.
  - Given N pending leaves in a wave and network max batch B, Then network calls <= ⌈N/B⌉.
- :Leaf-value: must blend by :Teacher-leaf-weight: along the :Prior-lineage:, so earlier priors guide early search and the learned network takes over late.
  - Given progress 0, When a non-terminal leaf is evaluated, Then :Leaf-value: is approximately the operative prior's normalized score.
  - Given progress 1, Then :Leaf-value: is approximately the :Learned-value:.
- Terminal leaves must short-circuit the blend with fixed outcome values in the mover-relative frame required by the :Negamax-backup-convention:.
  - Given a checkmate leaf, Then the value is a loss for the side to move, independent of the blend weight.
- If network evaluation fails, Then the leaf must fall back one :Prior-lineage: step rather than abort the simulation.
- PUCT priors must come from the annealed policy blend.
  - Given :Teacher-policy-weight: w, Then P is the frozen teacher policy snapshot blended with the current learned policy by w, softened by :Prior-bias-temperature: before selection.
  - Given rising progress, When a node expands, Then reliance on teacher/prior bias loosens, letting the learner explore off-book moves.
- :Progressive-narrowing: must be preserved in the small-budget profile; the annealed sample widths come from the :Annealing-schedule:, keeping one source of truth.
  - Exploration must stay structured (PUCT/UCB plus weighted sampling); widening must not degrade to uniform-random legal moves except as a last-resort fallback.
- The net-minimax profile must apply the value net only to quiet leaves (:Quiescence-search:).
  - Given a fixed-depth frontier position that is non-quiet (a capture is pending, a piece hangs, or the side to move is in check), When it would be evaluated, Then the search first extends captures/checks until quiet and evaluates there.
  - Given a quiet frontier position, Then the value net is applied directly.
- The :Policy-head: must have a consumer in the net-minimax profile via :Policy-move-ordering:.
  - Given the policy distribution P at a node, When moves are ordered/pruned for search, Then ordering and beam selection follow P (better P → better ordering → deeper effective search → better cloning targets).
- The batching/pruning tension must be acknowledged, not assumed away.
  - :Batched-leaf-evaluation: (parallel frontiers) and deep alpha-beta pruning (sequential) pull opposite ways; the beam-pruned batched minimax is a deliberate middle path. Depth targets MUST NOT be specified assuming true alpha-beta branching factors — expect weaker pruning than incremental-update CPU NNUE.
- The search must expose its :Search-value: alongside the chosen move and visit distribution, in the :White-absolute-frame:, so the value-target stage can bootstrap from it.
  - Given a completed search at a node, Then the negamax-backed node value is available to the caller without re-running inference.
