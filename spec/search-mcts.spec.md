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
- :Search-window-reuse: is the receding-horizon carry of the scored sampled subtree across moves: after the played move and the opponent's reply, the surviving subtree is kept as a warm start, the window shifts by the resolved plies, and one fresh fetch_k frontier layer is added — so effective horizon accumulates across moves at constant per-move budget, densifying the sparse sample along the played line. It is LEAKY: only the subtree under the opponent's ACTUAL reply survives, so a reply outside the sampled top_k re-seeds (the ponder-miss); policy-guided fetch reuses more than random fetch. The realized advantage :Delta: = (a move's backed-up value after the deeper re-search) − (its value estimated when it was selected) is the SIGNED TD error (unlike the non-negative :Move-margin:), and is exactly the :Search-value: the value-target stage bootstraps from — one search yields the move, the margin, and the training target.
- :Phi-rotation: is scheduled iterative deepening over the Fibonacci widths: each pass shifts one phi-step of width into one phi-step of depth, per layer, gated on that layer's top_k ORDERING being unchanged across the last two passes (unstable ordering ⇒ the layer is unsaturated ⇒ hold its width). Stability is NECESSARY, NOT SUFFICIENT — two jointly-too-shallow passes can agree yet be wrong; that error is corrected later by :Delta:, so premature deepening is bounded-lag, not permanent. Narrowing applies to EXPANSION ONLY, never RETENTION: scored candidates stay in the table (root-layer candidates indefinitely — bounded and cheap, they are the decision; deeper shelved siblings under a recency/decay eviction, being re-derivable), and narrowing only stops spending fan-out outside the shrunk top_k. If the incumbent line's backed-up value drops between passes (:Delta: < 0), the shelved siblings are re-opened at the shallowest divergent layer — aspiration-window fail-low re-search — so the rotation is not a one-way ratchet into a refuted line. (Together with :Search-window-reuse: this re-derives iterative deepening + aspiration windows + transposition retention for the sampled beam.)
- :Decision-rule: is that the played move is the argmax over ROOT MOVES of their backed-up (alternating max/min) value; raw comparison of leaf scores ACROSS DEPTHS is FORBIDDEN — an interior leaf-eval is superseded by its subtree's backup, and a high value sitting behind an opponent refutation is unreachable. In a sampled beam these backed-up values are sample-OPTIMISTIC (an unsampled opponent refutation inflates the alternating max/min): correct as the target, biased high until :Search-window-reuse: densifies and the :Delta: < 0 re-widening of :Phi-rotation: refutes — so :Decision-rule: and that re-widening are one anti-optimism loop.
- :Move-margin: is max − mean(top_k) over the surviving lines' root-perspective backed-up values — the DECISIVENESS of the winner over the field it beat (always ≥ 0, distinct from the signed :Delta:). It calibrates compute and difficulty: margin ≥ θ_easy for two consecutive passes terminates deepening early (easy-move detection) and marks the position as safe for :Strength-temperature: sampling; a low margin marks the position as ambiguous — exactly where deepening compute (and honest play) belong.
- :Root-commit: is the rule that the beam's FINAL pick is argmax over root-perspective backed-up values (τ→0 at the terminal level only), NEVER a temperature sample. The exploration budget is spent UPSTREAM — uniform fetch, temperature/MMR survival pruning, and the ε-mixture's direct-π branch — so committing the max at the root does not starve on-policy data (the played max is already max-over-a-random-subset, itself stochastic across draws). Committing the exploited move is also what makes the next volley's :Delta: a clean AUDIT: δ measures the move you would actually play, not a lottery; and δ's SPREAD across draws is the winner's-curse / sample-optimism magnitude that drives pair_k and the fetch random→policy shift.
- :Volley-growth: is the bounded schedule-deepening across re-searches (volleys): volley t uses fib_schedule(depth₀+t, 1, max(1, phi_start₀−t)), so the root width is HELD and one thin level is APPENDED at the bottom per volley — append-down, NEVER shift-down (shifting deletes a level at the phi_start floor rather than deepening). Planned growth of +k depth requires phi_start₀ = k+1; growth caps at +(phi_start₀−1). Appended tail levels are width 1–2 (single-digit Fibonacci), so a deep thin tail is one noisy principal variation — prefer widening the appended level or holding (:Growth-gate:) over ratcheting width-1 tails.
- :Growth-gate: extends depth by another level only WHILE the root :Move-margin: < θ_easy (position still ambiguous — deepening can change the decision) OR the incumbent line's :Delta: is unstable across the last two volleys (backups still revising); a high, stable margin BANKS the compute (easy move). This is :Value-of-information: steering the growth schedule, not merely early-stop.
- :Reuse-precedence: ranks the two horizon-growth mechanisms: the schedule (:Volley-growth:) is MINOR and bounded (+phi_start₀−1); the receding window (:Search-window-reuse:) is MAJOR and compounds every volley the opponent's reply stays inside the carried tree, but LEAKS to zero when it does not. The reuse HIT RATE (fraction of opponent replies found in the sampled subtree) MUST be logged — it decides when policy-guided fetch pays over uniform fetch, and pair_k is itself a reuse-persistence knob (larger pair_k covers more replies at linear cost), not only a per-move quality knob.

***implementation reqs***

- The search lives in `mcts.py`; it reads coefficients from the :Annealing-schedule:, it does not compute its own schedule inline.
- Constant: MIN_ROOT_VISITS — the visits-per-child threshold of the :Root-selection-rule:.
- :Gated-progress: must be threaded to the leaf evaluation, not stopped at the search entry point.
- `net_search.py` (batched fixed-depth minimax with beam pruning, ~0.9 s/move at depth 2) is the interim conformant fast profile until PUCT lands.
- The alternating max/min fold is implemented as ONE sign trick — a parent takes `max over −child` in the side-to-move frame — so after an even ply count the value is back in root perspective. Its two code sites are the `(−1)^k` alternation in the n-step returns and the `mover_val = -v` negamax commit in the beam fan-out; both trace to :Negamax-backup-convention:.

***test reqs***

- The historical failing position — opponent has just left a queen en prise (e.g. `rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w`) — with a ~200-simulation budget, to assert the :Root-selection-rule:.
- A mirror pair of positions backed up through alternating plies, to assert the :Negamax-backup-convention: for both colors.
- A midgame position plus a stub network returning a fixed value, to assert the leaf blend endpoints.
- A TRAP position where a move wins material immediately but loses it to a forced reply (poisoned pawn / trapped queen, the Qxb7-behind-...Rb8 shape): its eval right after the capture is positive but its backed-up value is negative. Assert (a) :Decision-rule: DECLINES it — the move backs up to the min over the opponent's replies (not the best leaf in its subtree), so the quiet alternative is played; and (b) a beam whose pair_k is below the refutation's policy rank backs the trap up OPTIMISTICALLY (the pre-capture positive value), pinning the sample-optimism bias, which the next move's :Search-window-reuse: :Delta: < 0 corrects.

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
- :Decision-rule: must select over root backed-up values, never raw cross-depth scores.
  - Given leaf values at different depths, When the root move is chosen, Then selection is argmax over ROOT moves' alternating-max/min backed-up values; a raw leaf score is never compared to one at a different depth.
  - Given a root move whose high value stands behind an opponent refutation in the beam, Then that value MUST be backed up (min at the opponent ply) before the move is eligible.
  - Given a move, When its value is formed, Then it is the leaf value at the END of its principal variation, folded up one ply at a time by alternating max (the mover) / min (the opponent) — equivalently `max over −child` in the side-to-move frame — and the position's eval IMMEDIATELY AFTER the move is superseded the instant the move is expanded. The best raw leaf anywhere in the subtree is NOT the value: every deep score is gated by the opponent's min nodes between the root and it (raw max over all leaves assumes the opponent cooperates). Worked fold — White to move, Qxb7 (eval +1.0 right after the capture) vs Nf3: after Qxb7 ...Rb8 traps the queen, White's deepest choice max(−7.0,−7.5)=−7.0, Black's node min(−7.0,+1.0)=−7.0, so Qxb7 backs up to −7.0 (the +1.0 is overwritten); Nf3 backs up to +0.2; argmax(−7.0,+0.2) ⇒ play Nf3.
- :Phi-rotation: must narrow expansion without discarding scored candidates, and self-correct on :Delta: < 0.
  - Given a layer narrowed at an earlier pass, When a later pass needs a shelved sibling, Then it is still in the table (root-layer indefinitely, deeper siblings under decay) and re-openable — narrowing stopped fan-out, not retention.
  - Given the incumbent line's backed-up value drops between passes (:Delta: < 0), Then the shelved siblings at the shallowest divergent layer are re-opened (aspiration fail-low re-search).
  - Given a layer's top_k ordering changed across the last two passes, Then its width is held, not rotated into depth.
- :Move-margin: must gate early termination of deepening.
  - Given :Move-margin: ≥ θ_easy for two consecutive passes, Then deepening terminates early for that move (easy-move detection); Given a low margin, Then deepening continues (the compute belongs there).
- :Root-commit: must confine stochasticity to fetch and survival, not the final pick.
  - Given the terminal beam level, Then the played move is argmax over root backed-up values; Given the fetch and survival-prune stages, Then temperature/MMR sampling applies there (and the ε-mixture's π branch supplies on-policy exploration).
- :Volley-growth: must append-down with a non-shrinking root, and :Growth-gate: must condition extension.
  - Given volley t, Then widths = fib_schedule(depth₀+t, 1, max(1, phi_start₀−t)) and widths[0] is non-decreasing across volleys (append-down, never shift-down).
  - Given the root :Move-margin: ≥ θ_easy and stable :Delta: across two volleys, Then depth is held and compute banked; Given a low margin or unstable :Delta:, Then one level may be appended.
- :Reuse-precedence: must log reuse and cap schedule growth.
  - Given a volley, Then the reuse hit rate (opponent reply found in the carried subtree) is logged; horizon beyond +(phi_start₀−1) MUST come from :Search-window-reuse:, not the schedule.
- :Search-window-reuse: must carry the subtree and expose :Delta:.
  - Given the opponent's reply lies in the retained subtree, Then it is reused as a warm start and only a fresh frontier layer is expanded; Given the reply lies outside the sampled top_k, Then the window re-seeds (ponder-miss).
  - Given a move's deeper re-search completes, Then :Delta: = (new backed-up value) − (prior selection estimate) is exposed as the :Search-value: the value-target bootstraps from.
- The search must expose its :Search-value: alongside the chosen move and visit distribution, in the :White-absolute-frame:, so the value-target stage can bootstrap from it.
  - Given a completed search at a node, Then the negamax-backed node value is available to the caller without re-running inference.
