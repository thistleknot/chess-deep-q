# Lessons Learned — the chess-RL reconstruction campaign

Distilled from `data/experiments.md` (full ledger) as of 2026-07-10. Each lesson is paired
with the measurement that taught it. Outcome context: the campaign plateaued at ~880–950 for
its whole life until the KnightCap reconstruction completed its final fidelity line (native
full-width quiescence search) — then jumped to **1212 (1109..1300, 7W vs SF@1320) in a single
day**, with the first confirmed training crowns arriving the same hour.

## Method

1. **Replicate before inventing — and replication means the CODE, not the paper.**
   Weeks of homegrown mechanisms (annealing schedules, variance-adaptive λ, beam-width
   sweeps) held a ~900 plateau. The decisive mechanisms — RAMP blunder filter, online
   per-game updates, trained coefficient values, the real search depth — were all in
   KnightCap's `td.c`/`large_coeffs.h`/`eval.h` and are barely mentioned in the paper.
   (Ledger: Merges 5–8 vs the S/E/D experiment series.)

2. **A universal failure across independent arms points at a shared TARGET defect, not a
   knob.** Four arms (fresh/seeded init, PST/donor features, self-play/graded opponents)
   all declined monotonically → one cause: maximization bias in the bootstrap (unbacked
   optimistic beam values). One fix flipped 490-collapse into 862-parity. Stop tuning when
   every arm fails the same way; start diffing the target construction.

3. **Maximization bias is a defect CLASS, not a bug.** It reappeared three times: max over
   unbacked beam values (:Backed-bootstrap:), ratcheting on noisy epoch maxima
   (:Confirmed-crown:), and KnightCap's own RAMP filter exists because opponent blunders
   are unearned positive surprises. S&B skill 013–015 named all three before we found them.

4. **Never make decisions on samples that carry no new information.** 24-game epoch
   strengths swing ±5; the old ratchet crowned phantoms (12.08 → re-measures 3–5). Grade
   and pivot on SIGNAL events (new samples, confirmed crowns, rung climbs), not wall-clock
   ticks; confirm candidate bests before believing them.

5. **Single-variable arms with pre-registered falsification are what make failure cheap.**
   Every dead arm (d2→d3 beam, init variants, diet) closed a hypothesis permanently in
   ≤600 games. The intervention queue ordered by information-per-hour beat enthusiasm.

6. **Match the comparator's EXPERIMENT SHAPE, not just its totals.** KnightCap's +500 was
   ~300 games × deep targets. We ran thousands of games × shallow targets and called the
   scale "comparable." Games × target-quality is the budget, not games.

7. **The evaluation only needs to ORDER moves; regression to calibrated values is a harder
   problem than the game requires.** Texel/DeepChess (outcome-supervised, ranking) reached
   the same strength as champion TD in a 2-minute fit on 522 positions (954 vs 837 unfitted,
   same measurement). Sometimes the family, not the tuning, is the choice.

## Technical

8. **Depth amplifies eval holes; features fill them; but SEARCH QUALITY gates everything.**
   The PST at depth 11 collapsed (926); kc features held; and the first-ever ladder climb
   into SF rungs came the moment full-width d4 quiescence search drove play — same eval.

9. **Quiescence is not optional.** Static evals at mid-capture positions are lies; KnightCap
   evaluated only settled positions. Our beam's recapture-only extension was a half measure.

10. **The hot loop belongs in native code; the learning loop belongs in torch; don't confuse
    the shapes.** Tree search = millions of tiny sequential branchy ops (GPU launch overhead
    20× the node cost). Rust/PyO3 gave 2M nodes/s — ~2000× the Python beam — in ~450 lines,
    with eval parity 2e-9 against the numpy encoder. CPU-native search + torch training +
    Python orchestration is the right split at this scale.

11. **Linear-over-features has real headroom when the features are right** — but init scale
    transplants are treacherous: rescaling the champion PST ×18 to hit KnightCap's tanh
    calibration amplified per-square noise until material responses inverted. Transplant
    RATIOS in the recipient's own units.

12. **Test probes must be minimal pairs.** Three test failures in one day were bad tests:
    single-square PST signal is the same order as feature signals, so any probe that moves
    a piece between squares measures the PST, not the feature. Toggle exactly one fact.

## Confirmed by the breakthrough (2026-07-10, the 1212 rung)

17. **The load-bearing component gets PROVEN, not argued.** Five arms eliminated every cheap
    recipe line; the one remaining deviation (full-width deep quiescence search) delivered
    +264 Elo (948 → 1212, 95% CI 1109..1300, non-overlapping) on an UNTRAINED eval, and the
    first confirmed training crown (12.57 → 29.78) plus the first-ever ladder climbs into SF
    rungs within one epoch of using it for targets. Elimination + measurement > intuition:
    the fidelity matrix said "weaker" on that row from day one; the ledger had to earn it.

18. **Outcome-supervised fits flatten on unbalanced data.** Texel at 1000 loss-heavy games
    (70% stronger opponents) shrank the eval toward base-rate prediction (1-ply collapsed to
    draw-shuffling); its 20-game smoke had survived on its init. Balanced outcomes or
    anchored regularization are preconditions for the family, not refinements.

19. **Inference-time strength and training-signal strength are the SAME lever.** The deep
    search that added +264 at play is what made TD targets informative enough to climb —
    KnightCap's design is one engine doing both jobs. Splitting them (shallow training
    search + deep measurement, or vice versa) wastes the lever twice.

## Process / Ops

13. **Show progress where the operator looks.** Search-side gains were invisible for a day
    because the dashboard only plotted the raw net ("well fucking show me it too then").
    Every measurement now lands in data/rl_trend.jsonl → the console ladder panel.

14. **`taskkill /T` on a server kills its trainer children.** Detached-flag spawning does
    not reparent. Restart servers WITHOUT /T when a run is live; pid-file re-attach tracks
    the orphan. (Cost: one live leg.)

15. **Cheap advisers are force multipliers when kept advisory.** The sidecar/council pattern
    (propose → search → arbitrate, decisions and vetoes logged) surfaced Texel tuning and a
    real queue reorder; its SF-label proposal was correctly vetoed against the from-scratch
    constraint. Proposals never launch compute directly.

16. **Environment friction to expect on Windows:** AV ransomware shields block build tools
    writing to Documents (whitelist or redirect CARGO_TARGET_DIR); corporate TLS breaks
    crates.io revocation checks (CARGO_HTTP_CHECK_REVOKE=false); no MSVC → stable-gnu
    toolchain works with maturin/PyO3.

20. **Build scripts must never write to live lineage paths.** The seed builder defaulted its
    output onto the active lineage and overwrote the campaign-best checkpoint; git (because
    checkpoints were committed with the code, per the mlflow-pegging directive) held the
    restore. Parameterize outputs; treat `_best.pt` files as append-only treasure.

21. **"Full-width" was never the cost — the eval is.** LMR/aspiration (v3.5) was falsified
    because alpha-beta+TT+ordering already visits ~1/160,000th of the nominal d7 tree
    (~400k nodes at ~2M nps = 0.2s/move); there was nothing left to prune. The remaining
    efficiency lever is per-node cost: every node recomputes all 809 features, but a linear
    head admits exact incremental updates (v += w . delta-features, the NNUE idea) — ~10x
    node speed = ~2 free plies = ~+250 Elo at the same clock. Full-depth *rungs* are shelved
    as trophies; depth from here on comes from cheaper nodes, not longer clocks.

## The one-line version

Theory first (S&B named every bug before we hit it), then stand on the strongest prior art
you can actually run (code + weights + training script), measure one change at a time against
confirmed numbers, and put the hot loop in native code the moment wall-clock becomes the
binding constraint.

## Search-distillation campaign (2026-07-22) — the eval learns what the search knows

Outcome: expert-iteration self-distillation (label the champion's own self-play PV-leaf FENs with the
SAME net's d5 native minimax, drop |label|>0.90, ridge-100 atanh linear fit) produced an eval **+232
Elo stronger than the 1878 TDLeaf champion, native-d9 head-to-head** (robust: A2 +232, A3 +221, A4
+267, all bands exclude 0.5; control champ-vs-champ 0.490). Promoted distillA2. See
`spec/search-distill.spec.md`, `data/search_distill_campaign.md`.

10. **The teacher is eval + search, not the eval — so imitation COMPOUNDS instead of converging to
    the eval.** Distilling the net's own d5-minimax backed values into its static eval makes the
    static eval ~one lookahead stronger; deployed at d9 that converts to strength. This is the
    bitter-lesson mechanism in miniature: search generates a signal stronger than the current eval,
    the eval learns it. The FIRST step captures most of it (A2≈A3≈A4); iteration converges fast and
    weight-averaging the iterates (SWA) DILUTES rather than robustifies.

11. **Your ruler can be 1200 Elo wrong — validate every instrument with a self-control before you
    trust a verdict.** Five instruments lied here, each nearly flipping a verdict:
    - The vs-SF **anchor ladder DRAW-FLOODS strong agents** (120-ply cap, no adjudication → 0.5 at
      upper anchors), compressing ratings together: it reported distillA2 1917 vs champ 1878 (+39,
      "tie") while the direct head-to-head was **+232**. For agents near/above the top anchor, the
      direct native deterministic-diverse-opening duel is the true ruler — NOT the ladder.
    - `anchor_ladder` **bare ckpt = native d9** (correct); an `enc:amap:<ckpt>` spec routes to a
      **d2-beam mover** → a ~1200-Elo phantom (distillA2 showed 700 vs SF at the beam mover, 1917 at
      native d9). Measure deployment strength with the bare ckpt.
    - **head2head is a d2-beam lens** (the canonical repo ruler for ALL prior verdicts) — it does not
      track native-d9 deployment (distillA2 scored +798 d2-beam yet only tied on the draw-flooded
      ladder). Every head2head-screened verdict inherits this caveat.
    - `rsearch4.play_games` applies exploration (eps/tau) to the **agent side only** → champ-vs-itself
      scored 0.000. It is a training-data GENERATOR, not a fair duel ruler.
    - The control that saves you: **A-vs-A must score ~0.5**. Champ-vs-champ = 0.490 (band incl 0.5)
      validated the head-to-head; without it the +232 would be unfalsifiable.

12. **Bad labels beat the fit before the fit begins.** Raw ridge on self-play labels that saturate at
    ±1 (won/lost tactical leaves) blows the fitted eval scale 16× (‖w‖ 10.3 vs champion 0.65,
    corr +0.06) → native-unplayable. Dropping |label|>0.90 fixed it (‖w‖ 0.78, corr +0.39). The
    saturated targets carry no positional gradient; they only wreck the scale the native searcher's
    aspiration windows depend on.

13. **On Windows, spawn-multiprocessing re-imports your module in every worker — keep hot-loop
    workers dependency-light.** Pool workers re-importing a module that `import torch`s loaded CUDA
    DLLs per worker and exhausted the paging file (WinError 1455). The fix was a torch-free worker
    module (`experiments/_duelcore.py`, imports only rsearch4/chess/random). Also: `python - <<EOF`
    heredocs make `__main__` = `<stdin>`, which spawn cannot re-import — run multiprocessing from real
    `.py` files with an `if __name__ == "__main__"` guard.

## Uncertainty-directed MCTS (2026-08-16) — the model learns where it's confused

Outcome: K=16 bootstrap ensemble disagreement identifies positions where the eval is uncertain.
Bounded PUCT (32 sims, soft-policy prior from the eval itself) at those positions makes the agent
+191 Elo stronger than pure greedy. Used as a training-data generator (vs varied opponents), the
hybrid produces decisive games the greedy agent draws → 7,598 novel positions → MC outcome labels
→ refit ridge → **new model beats old champion +191 Elo H2H (20W 20D 0L, 40g), beats heuristic
40-0** (old champion scored 0.250 vs heuristic at d1). `models/champion_umcts.pt`.

14. **Uncertainty-directed compute beats uniform compute — but only for GENERATION, not labeling.**
    The offline test (Gate 1) showed σ_ens correlates with residual (0.127, p=2e-53) but selective
    LABELING of existing positions was negligible (+0.04% RMS). The win comes from using uncertainty
    to direct WHERE THE AGENT PLAYS (generating novel positions in uncertain territory), not which
    existing positions get deeper labels. Generation > labeling when the basis is at its regression
    ceiling.

15. **Self-play with the same agent draws every game and produces no diversity.** The hybrid vs
    itself (50 games): 0 decisive, 82 unique positions. The hybrid vs varied opponents (200 games):
    133 decisive, 7,695 unique positions. Diversity of opponent is the generation mechanism, not
    self-play. Same lesson as KnightCap (graded opponents, not self-play) in a new context.

16. **Game outcomes are free labels and they work.** MC return (γ^t × z) from decisive games as
    training targets bypassed the need for expensive deep-search labeling (d5 = 3s/pos, infeasible
    for 7k+ positions without rsearch4). The decisive games ARE the ground truth — positions in
    winning games have positive value, positions in losing games have negative value. No search
    needed.
