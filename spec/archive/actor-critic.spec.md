# Merge 3 — Online On-Policy Actor–Critic with Eligibility Traces

Grounding: Sutton & Barto (2018) §13.5–13.6 ("One-step Actor–Critic", "Actor–Critic with Eligibility
Traces (episodic)" — the pseudocode box on p.332 is transplanted below), §11.3 (deadly triad), §6.8
(afterstates); Deep RL in Action ch. 5 (advantage actor-critic practice: entropy regularization).
Predecessor evidence: Merge 2 capped at pooled ~650 vs SF@1320 (data/experiments.md ceiling report).

***concepts (committed)***

- :AC-episodic-traces: is the S&B p.332 "Actor–Critic with Eligibility Traces (episodic)" box, adapted
  to self-play chess in the White-absolute frame. Per MOVE t (one ply; both colors share θ, w):
  ```
  A ~ π(·|S, θ)                                  # stochastic policy IS the behavior — on-policy
  take A → S′, R  (R = 0 until terminal; terminal R = z ∈ {−1, 0, +1}, White-absolute)
  δ  = R + γ·v̂(S′,w) − v̂(S,w)                   # v̂(terminal) ≐ 0; ply-cap counts as terminal draw
  z_w ← γ·λ_w·z_w + ∇_w v̂(S,w)                  # critic trace (accumulating, per S&B box)
  z_θ ← γ·λ_θ·z_θ + I·σ_t·∇_θ ln π(A|S,θ)       # actor trace; σ_t = +1 White to move, −1 Black
  w  ← w + α_w·δ·z_w
  θ  ← θ + α_θ·(δ·z_θ + β_t·I·∇_θ H(π(·|S,θ)))  # entropy bonus, see :Entropy-regularization:
  I  ← γ·I                                       # discounted policy-gradient weighting (S&B box)
  ```
  Traces z_w, z_θ and I reset at every episode start. FULLY ONLINE: updates happen as moves occur and
  the transition is never revisited — S&B §13.5: "fully online, incremental algorithm, with states,
  actions, and rewards processed as they occur and then never revisited." NO replay buffer, NO frozen
  generator, NO greedy-max target: those were Merge 2's off-policy machinery and MUST NOT appear here.
  σ_t folds the zero-sum frame into the actor: each side ascends ITS OWN objective under the shared
  White-absolute δ (Black's good outcome is δ<0; σ=−1 makes its policy-gradient ascend that).
- :Policy-parameterization: — softmax over legal AFTERSTATES (soft-max in action preferences, S&B
  §13.1): π(a|s,θ) = softmax_a( σ·h(x_a; θ) ), preferences h = θᵀx over the SAME 769-dim encoding as
  the critic (`cem_loop.encode`), afterstate x_a = position after the candidate move. Linear actor v1
  (bias-free — a shared bias cancels in the softmax). Closed forms (exact for linear):
  ∇_θ ln π(A|S,θ) = σ·(x_A − Σ_b π(b|s)·x_b)
  ∇_θ H(π)        = −σ·Σ_b π(b|s)·(ln π(b|s) + 1)·(x_b − x̄),  x̄ = Σ_c π(c|s)·x_c
  The critic v̂ reuses ValueNet (linear default; mlp behind `QLEARN_ARCH`) with per-parameter trace
  tensors via autograd — uniform across archs.
- :Entropy-regularization: (Deep RL in Action §5.3 practice) — the exploration mechanism. β_t anneals
  via the SHARED `anneal()` (fast-then-slow, NEVER reaching the floor): β_t = anneal(β, β/10, progress,
  0.5). Replaces Merge 2's τ schedule entirely: the policy is stochastic by construction, and entropy
  keeps it from premature determinism; there is no ε and no temperature.
- :Deadly-triad-disposition: — the triad (S&B §11.3) = function approximation + bootstrapping +
  OFF-POLICY training; instability requires ALL THREE. This method keeps FA and bootstrapping but is
  ON-POLICY (behavior ≡ π_θ; the critic learns v_π under π's own state distribution — the convergent
  regime for linear FA, Tsitsiklis & Van Roy 1997). "Model the transition probabilities to fix the
  triad" is a CONFUSION: model-based RL (S&B ch. 8) is an orthogonal axis; in chess the transition
  model is KNOWN and deterministic and already exploited via afterstates (§6.8) — the only stochastic
  element is the opponent, which self-play models with the policy itself. Planning with the known
  model (search) is Merge 4, not a triad repair.
- :Honest-expectation: (falsifiable) — the actor is still a 1-ply REACTIVE policy. Prediction: pooled
  objective lands in Merge 2's ~650 band UNLESS direct policy credit assignment + inherently
  stochastic play changes the draw dynamics (measured by: decisive rate vs Merge 2's flat 0.084,
  vs-random score, first WIN vs SF). Either outcome is the rung's finding. If reward starvation
  persists, the book-grounded next levers are Deep RL in Action ch. 8 (intrinsic curiosity for sparse
  rewards) and Merge 4 (search over the learned evaluation — the repo-evidenced 1320-crosser).

- :Anchor-ratchet: (shared with Merge 2; `QLEARN_ANCHOR=1` default, console "anchor to best (gate)") —
  the checkpoint is embedded in a PROPER ACCEPTANCE LOOP, not kept as a loose file: each epoch is a
  PROPOSAL trained from the anchor's lineage; the pooled epoch-strength gate ACCEPTS (anchor := new
  best, `*_best.pt` written) or REJECTS (weights REVERTED to the anchor before the next proposal).
  This is AlphaGo Zero's evaluator/gating loop (Silver et al. 2017: the candidate must beat the
  champion to become the generator) realized at epoch granularity — greedy hill-climbing over policy
  space with SGD-epochs as the proposal distribution. Fresh (non-resume) runs DELETE any stale
  `*_best.pt` first — a lineage never gates against a different lineage.
  **v2 semantics (measured, D7/D8 paired evidence):** `QLEARN_ANCHOR` ∈ {0=off, 1=COLLAPSE GUARD
  (default; revert only when strength < `QLEARN_REVERT_FRAC`·best, default 0.5 — mild wandering may
  explore, collapses are caught), hard=revert every non-improving epoch}. HARD gating stalls under a
  noisy 24-game gate: D8's lucky epoch-4 bar froze 6 straight proposals (1,200 games discarded) and
  reached a LOWER best (3.03) than ungated D7 (4.87) — the same reason DeepMind dropped gating in
  AlphaZero. Conversely D7 shows why selection is still mandatory: it peaked at epoch 2 and ENDED
  weaker than it started. Resolution: exploration continues (no hard revert), the collapse guard
  bounds the damage, KEEP-BEST does the selection, and the FINAL MEASURE ALWAYS loads `*_best.pt` —
  the run's output is the best-visited policy by construction; resume-from-best ratchets across runs.

- :Curriculum-starts: (`QLEARN_CURRICULUM` fraction, 0=off) — EXPLORING STARTS (S&B ch.5 Monte Carlo
  ES) + reverse curriculum, targeting the measured reward starvation (decisive rate ~9%: self-play
  from the opening is a DRAW SWAMP for a weak policy — the user's "route around low-return terrain").
  With probability `curriculum_t`, an episode starts from a LEGAL MATERIAL-REDUCED position (kings +
  n random pieces, n annealed upward) instead of the opening: near-endgame states are the fertile
  subspace where even weak policies find mates → dense terminal signal → value gradients. The
  curriculum fraction anneals DOWN (and piece count UP) with cumulative progress via the shared
  `anneal()` — walk the corridor outward from the goal toward the full game. Positions are validated
  (`board.is_valid()`, not in check for the side not to move, no pawns on promotion ranks); evals and
  Elo measures ALWAYS play the standard game (curriculum shapes TRAINING data only, never the
  yardstick). Training-mechanics change ⇒ REGIME bump when studied.

***carried over from Merge 2 unchanged*** (see spec/q-learning.spec.md for definitions)
:Calibrated-elo: (n-aware half-point clamp) · pooled :Elo-objective: v2 + tie-break · :Elo-patience: ·
:Piece-worth-observability: (read from the CRITIC's linear weights; also from the ACTOR's θ — logged
separately as `piece_vals` (critic) since the actor learns move PREFERENCES, not values) ·
:Reward-trace: · :Study-resume: (REGIME "ac-episodic-traces|onpolicy|entropy|v1") ·
:Checkpoint-resume: (checkpoint `models/ac_learn.pt`: θ + critic w + optimizer-free + cum_games;
trials write `models/ac_optuna.pt`) · sample-size vocabulary (games/epoch drives eval cadence and
patience; `batch_games` is NOT an update cadence here — updates are per-move — it remains only the
generation accounting unit for logging).

***implementation reqs***

- `ac_learn.py` mirrors qlearn.py's harness (env knobs, metrics JSONL row shape → SAME dashboard,
  pooled KILL-CHECK objective, epoch/patience loop, checkpoint/resume) with the :AC-episodic-traces:
  core replacing generation+replay+SGD. New env knobs: `QLEARN_ALPHA_W`, `QLEARN_ALPHA_TH`,
  `QLEARN_LAMBDA_W`, `QLEARN_LAMBDA_TH`, `QLEARN_BETA`. Metrics row reuses `lam_eff` (:= λ_w) and
  `tau` (:= β_t, the exploration dial analog) so the console renders without changes; `td_sigma` :=
  running sdev of δ over the logged window (same variance diagnostic, no adaptation).
- `server.py` TrainReq gains `algo: "q"|"ac"` + `alpha_theta`, `lambda_theta`, `entropy_beta`;
  /api/train/start dispatches to qlearn.py or ac_learn.py and passes the AC envs. Form: algo selector;
  AC-only numeric fields are API-only (pydantic defaults).
- `tune_qlearn.py` gains `[algo]` CLI arg (fingerprint component; REGIME switches per algo). AC
  manifold (6 algorithmic dims, infrastructure passed fixed as ever):
  γ [0.95,0.999] · α_w [1e-4,1e-2] log · α_θ [1e-5,1e-3] log (actor slower than critic — standard
  two-timescale practice) · λ_w [0.3,0.95] · λ_θ [0.3,0.95] · β [1e-3,1e-1] log.
  Priors: {γ .99, α_w 3e-3, α_θ 1e-4, λ_w .8, λ_θ .8, β .01}.
- Acceptance: (1) smoke 30 games — δ finite, traces bounded, entropy decreasing with β anneal, piece
  worth readable, KILL-CHECK pooled objective emitted, resume round-trips; (2) study S6 differentiates
  across trials; (3) deep run D7 (200×10, patience 10, elo 60) lands in the ledger with the verdict
  reconciled back into THIS spec's :Honest-expectation:.
