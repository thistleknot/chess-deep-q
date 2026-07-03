# chess-deep-q spec

Source-of-truth specs for the RL alignment described in `../prompt.md` ("apply RL to a
system such as chess when you already have a known evaluator/prior"). Authored with the
`spec` skill (rendered layer). **Spec-driven development: the spec is authoritative; code traces
to it, not the reverse.**

**Single entry point: [`chess-rl.spec.md`](chess-rl.spec.md)** — the root rendered spec. Its
`import:` list (training-loop, rl-categorization, dynamic-difficulty, elo-calibration,
terminal-interface) transitively closes over all 13 specs below, so following imports from that
one file reaches the entire set. This index gives the human reading order; `chess-rl.spec.md` is
the machine-followable root.

Import order flows bottom-up:

1. **elo-measurement** — the real Stockfish anchor, measured Elo, and the gates that own all
   progress. No imports; everything gates on it.
2. **annealing-schedule** — the shared coefficient service (prior → learned handoff), driven by
   Elo-gated progress. Imports 1.
3. **prior-evaluator** — the fixed heuristic prior and the :Prior-lineage: it starts
   (heuristic → distilled teacher → learned net). Imports 1–2.
4. **learned-model** — the dual-head residual tower (value + policy), value-target convention,
   batch-evaluate API, reward frame and sign. Imports 2–3.
5. **teacher-distillation** — Stage 1: Stockfish distillation under the 5-minute cumulative
   run contract (process-separated labelling, dedup'd dataset, Elo trend). Imports 1–4.
6. **search-mcts** — PUCT search with batched leaf evaluation, the exposed search value, and two
   regression-pinned conventions (negamax backup sign; argmax-Q root selection at small budgets).
   Imports 2–5.
7. **value-target** — the value-head learning rule: a search-bootstrapped λ-return (tree-backup,
   off-policy-safe) shared by the refinement and self-play stages; TD(0) and Monte-Carlo are its
   endpoints. Imports 2, 4, 6.
8. **self-play-leela** — Stage 3: expert iteration (visit-count policy distillation, λ-return
   value target, Dirichlet carve-out, surpass-teacher gate, optional σ-matched early opponent).
   Imports 1–2, 4–7.
9. **training-loop** — the staged loop: gate-driven stage controller, reward assignment,
   failure-mode monitoring. Imports 1–8.
10. **dynamic-difficulty** — adapt the opponent's move-selection temperature to the human
    player's skill band (regret-tracked, *relative*). Imports learned-model + search-mcts.
11. **elo-calibration** — the temperature→*absolute*-Elo dial: calibrate one net to any target
    strength on the human curve. Imports elo-measurement + dynamic-difficulty.
12. **rl-categorization** — qualified three-stage classification (supervised distillation →
    off-policy λ-return refinement → expert iteration; never SARSA/PPO/literal-DQN).
13. **terminal-interface** — the terminal human-vs-computer front-end: move entry, in-game
    commands (full word + first-letter shortcut), and the per-turn board readout that surfaces
    the estimated Elo. Imports dynamic-difficulty + elo-calibration.

The load-bearing idea across the set: a **prior lineage** — the hand heuristic bootstraps the
distilled Stockfish teacher, the teacher bootstraps the self-play learner, and the learner
eventually surpasses the teacher — where every handoff is **annealed toward the learned model**
(never toward randomness; root Dirichlet noise in self-play games is the one bounded, constant
carve-out) and **gated by measured Elo against a real Stockfish anchor**, never by wall-clock or
game count. Failure modes (teacher lock-in, reward hacking, policy collapse, stagnation at a
gate) are monitored, not assumed away. Two search bugs found by measurement are pinned as
regression specs in search-mcts: the negamax backup sign convention and argmax-Q root selection
at small simulation budgets.

The value head learns a **search-bootstrapped λ-return** (value-target), not literal TD(0) and not
pure Monte-Carlo — the bootstrap share β anneals down from lean-on-the-distilled-value (low
variance early) toward the ground-truth game outcome (AlphaZero MC) as strength is proven. The
policy learns by MCTS visit distillation (expert iteration), never policy gradient. An optional
σ-matched "just-above" opponent (self-play-leela) can sharpen early training but is annealed out in
favor of full-strength symmetric self-play, since a weakened opponent yields lower-quality targets.
