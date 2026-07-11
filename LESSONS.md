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
