# Dispositioned specs — index of everything superseded by the canon

Canon: [`spec/trivium.spec.md`](trivium.spec.md). Live operating protocols:
[`expectations.spec.md`](expectations.spec.md) (grading rubric),
[`intervention-queue.spec.md`](intervention-queue.spec.md) (web council / sidecar),
[`console-pargen.spec.md`](console-pargen.spec.md) (console surface contract).
Everything below lives in `spec/archive/` with its disposition.

## Absorbed into the canon (mechanisms live in code, spec text historical)

- `tdleaf.spec.md` — TDLeaf(λ), :Backed-bootstrap:, :Ramp-filter:, :Confirmed-crown:,
  :Informative-patience:, :Trivium-anneal:, :Graded-opponents: — all folded into the recipe.
- `knightcap-full.spec.md` — KC-faithful mode (donor fidelity); the replicate-first pillar.
- `parallel-generation.spec.md` — PARGEN native self-play batches (Merge 9).
- `eval-features.spec.md` — the 809 donor feature set (Merge 6).
- `rust-search.spec.md` — rsearch engine (Merge 8); v3.5 LMR/aspiration honestly falsified
  (tree already skeletal — LESSONS #21); depth is an inference converter.

## Superseded

- `pathfinding.spec.md` — :Out-of-bounds: exploration framing → superseded by the
  expectations rubric + council protocol.
- `q-learning.spec.md`, `actor-critic.spec.md` — Merge 2/3 lineage; AC dispositioned at the
  actor-sharpness wall; Q-learning evolved into the canon recipe.
- `decision-time-planning.spec.md`, `self-improvement-loop.spec.md`, `environment.spec.md`,
  `observability.spec.md` — early-era scaffolding, superseded by the console + canon.

## Prior archive (pre-campaign era, already under `spec/archive/`)

`annealing-schedule` `chess-rl` `dynamic-difficulty` `elo-calibration` `elo-measurement`
`entrypoint` `learned-model` `linear-value-rl` `nnue-eval` `prior-evaluator`
`rl-categorization` `search-mcts` `self-play-leela` `teacher-distillation`
`terminal-interface` `training-loop` `value-target` — the exploratory era that the
replicate-first pivot ended. `nnue-eval` may be revived by the queued capacity arm
(Merge 10 / purist representation queue).

## Archived UNTESTED (operator call, 2026-07-12 — all effort to Merge 14)
- Merge 10 (incremental eval / efficiency) — spec'd in LESSONS #21 context, never built.
- Merge 12 (policy head, spec/policy-head.spec.md) — spec committed, demoted by council
  round #7 (no linear-capacity precedent), never implemented.
- Merge 13 (Giraffe-lineage MLP capacity) — council-ordered, never implemented.
These remain re-openable; specs/ledger entries preserved. Active line: Merge 14 (GRPO + Elo
surprise, spec/relative-reward.spec.md).
- Merge 14 (rating-surprise + graded, spec/relative-reward.spec.md) — mechanics IMPLEMENTED
  and smoke-passed, arm never run; superseded by Merge 15 (spec/admixture-replay.spec.md)
  which includes its machinery. Archived as an arm 2026-07-12.
- Merge 15 (admixture replay, spec/admixture-replay.spec.md) — mechanics implemented; its
  3-trial study COMPLETED (best 892 = proven parms + replay_t 1.0, strongest clean-regime
  trial to date); arm superseded by Merge 16 full stack (spec/full-stack.spec.md) which
  includes all its machinery + the magic deck. Archived 2026-07-12.
- `relay.spec.md` — restart-wave driver; verdict logged closed/refuted in-file. Moved to
  archive/ 2026-07-25 (file had stayed in the active directory past its own closure).
- `policy-head.spec.md` (Merge 12) — spec committed, demoted by council round #7 (no
  linear-capacity precedent), never implemented; see "Archived UNTESTED" entry below.
  Moved to archive/ 2026-07-25 alongside its disposition record.

## Feature-screen ledger (screening tier, do-not-retry without new conditions)
Everything below screened against a matched fresh-seed control on the duel ruler
and PARKED (band spans zero) or CONFIRMED. Retrying any parked item requires an
explicitly finer instrument (operator H2H_CAP raise) or a changed variant —
never a plain re-run.
- tpst threat/guard planes (+1536d) — TIE −10 (−38..+17), 2026-07-14.
- hpst hanging planes (+768d) — TIE +15 (−13..+43), 2026-07-14; composition
  amaph (amap⊕hanging) also flat — hanging adds nothing over coverage.
- **amap attack maps (+128d) — CONFIRMED +51/+72 two seeds, 2026-07-15 → CHAMPION.**
- backup-λ (mellowmax soft backup, operator concept) — tie solo (+0), stack
  unconfirmed at seed-2 (pooled spans zero), 2026-07-15; mechanisms 0-for-7.
- dmap destination maps (mobility): alone −35 (−130..+61) TIE; **on top of amap
  +0 (−95..+95) DEAD TIE**, 2026-07-15 evening — mobility rides inside coverage
  (~90% bit overlap). Re-open variant only: dmap XOR amap (pushes+blocked bits).
- E5 per-piece mobility COUNTS — the compressed form; superseded by the dmap
  screen result (the map version tied ⇒ counts stay parked).
- kamap (kc-809 ⊕ amap, 937d) — TIE −42 (−138..+54), 2026-07-16. The
  merge-of-winners cell (future_exploration #1): KnightCap terms are
  attack-table-derived ⇒ mostly overlap + dilution. Re-open only at a higher
  training spend if :Arm7b-data-probe: supports H_data.
- cmap graded attacker counts (/4, same 128 dims as amap) — **LOSS −147
  (−251..−44), signed kill**, 2026-07-16. Declared confound: /4 scale
  attenuation vs amap's 1.0 bits; re-open variant only: unit-scaled counts.
- amaps (amap ⊕ E6 protected/threatened scalars, 24d) — TIE −28 (−123..+68),
  2026-07-16. The dilution-proof hpst retry still adds nothing; covered-count
  term pre-dropped (linear-span argument).
- dmap XOR amap re-basing — CLOSED WITHOUT A LANE, 2026-07-16: span-equivalent
  to amap⊕dmap for a linear net; amapd already measured that space (+0).

## Search-arm screen (pre-registered 2026-07-16, spec/search-arms.spec.md)
Decision-time search shapes over the FROZEN champion evaluator (amap-897); Optuna
screens (30-min cap each, H2H Elo-diff objective vs the d2 duel lens) → round-robin
of arm-bests → winner vs Optuna'd baseline. Four studies:
- `hyb` λ-tree-backup (hard mean↔minimax blend IN THE SEARCH TREE) + ab:<d> leaf —
  declared DISTINCT from the parked backup-λ entry above (that was training-time
  mellowmax; this is inference-time hard blend = changed variant under the re-open rule).
- `puc` value-PUCT rematch with champion leaf (prior PUCT ran a weak tower net only).
- `ucv` UCB-V variance-adaptive selection (no variance term exists anywhere in repo
  search code — first test).
- `baseline` the incumbent beam mover's own knobs (depth/width/tau) — no champion-config
  Optuna study existed before this (trivium knobs were tuned on kc and transferred).
Verdicts (2026-07-16 evening, full detail data/h2h_verdict_sa_campaign.md):
- **hyb — CONFIRMED at screen tier**: swept the RR (2-0) and beat the Optuna'd
  baseline +338 (95% +100..+576, band excludes zero) despite conceding time to
  baseline's d3w4. Winner = the pre-registered prior (λb=0.5, ab2 leaf, c=1.5, tp=0.2).
  Mechanism receipt: native ab2 leaves ≈ free (1.33 vs 1.21 s/mv @64 sims).
  NOT yet ladder-placed — claims rung vs SF pending operator go.
- puc — mildly positive vs the d2 lens (+89 best @16g); consistent with the PUCT
  parity disposition; superseded by hyb (same engine, λb=0/vf-leaf is hyb's subspace).
- ucv — PARKED: all trials negative (best −66 @16g), lost RR 0-2; variance bonus is
  budget the 16-sim regime can't afford. Re-open condition: ≥10× sims budget.
- baseline — secondary find: the duel lens itself improves narrower+deeper
  (beam d3w4 τ0.033 = +338 over the standard d2w8 lens) — instrument note.

Ladder disposition (2026-07-17 morning): **hyb PARKED at full budget — class gap,
not shape defect.** At the champion's d9 think budget (4.06 s/mv → 206 sims,
:Budget-match: probe): 60g deep rung **1314 (95% 1224..1404)** vs SF@1320 — CI
ceiling ~320 below the 1724 floor; H2H vs the champion's native d9 mover
**0-16, −597 (95% −597..−248)**, band excludes zero. 200g claims run CANCELLED
(would only tighten a CI on a decided question — operator time-budget law).
Standing lesson re-confirmed at a new point: search SHAPE wins within an
implementation class (hyb swept every same-budget Python mover incl. the tuned
incumbent), but implementation class beats shape — native sound depth converts
wall-clock ~13x better than the Python tree (206 sims vs d9 in the same 4 s).
Re-open condition: port :Lambda-tree-backup: + :AB-leaf: INTO the native rsearch
engine (~100x sims at equal wall) — new lever, operator-gated.
:Lambda-target-training: screen (2026-07-17 night, operator idea "mcts to reduce
training time") — PARKED at one study: paired 3-trial side-by-side (identical
TPE seed -> identical per-trial hyperparams; only target source differs),
control beam targets 586/586/710 @61m vs λ-tree targets (lb0.5, 16 sims)
587/783/587 @105m. Verdict: **+72% wall per epoch, paired diffs +1/+197/−123 =
noise at elo12** — no training-speed win; the cost is the Python tree, so the
question FOLDS INTO the native-port lever (tree targets ~free there) rather
than earning its own lane. σ-adaptive λ schedule (dispersion-driven, spec'd)
waits on the same port. Studies qlearn_elo_a988f596 (control) / _b5dfd3b4 (arm).
Mechanism stays shipped + flag-gated OFF (QLEARN_MCTS_LAMBDA=0 default,
gates passed, :Backed-bootstrap: honored).

SIMS-SCALING PROBE (2026-07-17 evening, pre-registered go/no-go, operator-run):
hyb vs its own λ=0/vf ablation at EQUAL sims, same champion evaluator, 20g/rung:
**+269 @16 → +338 @64 → +512 @256, all bands exclude zero, monotone GROWTH** —
the UCT-consistency decay hypothesis is refuted in-range; the shape edge
strengthens with scale. Verdict: **native port GO-justified** (caveat: 256 sims
is still ~10-40x below native operating range — the probe licenses the effort,
not the outcome). data/probe_sims_scaling.md.

Screen-validity disposition (2026-07-17, operator-ordered calibration; LADDER
FREEZE honored — y = recorded receipts only, x = 20g d2-lens screen duels):
**the d2-lens screen does NOT predict full-run strength — Spearman ρ = −0.70
(n=5: champion, kc1670, triv1540, vol1484, hyb)**; data/screen_vs_full.png/.json.
Mechanism: (a) depth-blindness — champion/kc1670/triv1540 all tie the lens at d2
(bands straddle zero) while their full-run gap is 184+ Elo (H_depth, previously
logged 2026-07-15, now quantified); (b) the screen rewards d2-transferable tricks
(hyb's search shape, vol's d2-suited eval) that do NOT survive depth. n=5 ⇒ ρ not
significant (p≈0.19) — the claim is "no evidence of predictive value + likely
inversion," not a proven anti-correlation. VALID remaining use: within-class
ranking at matched vehicle+budget (hyb>puc>ucv was instrument-consistent).
Contender screening at the FULL vehicle (small-n d9 duels / multi-anchor MLE at
20g) is the untested candidate replacement — ladder-frozen, operator-gated.
Companion scatter (operator ask, same day): Optuna-era best Elo (short-training
tier, from models/qlearn_optuna.db + qlearn_results.jsonl, era-mapped by param
signature + dates) vs full-run claims — **ρ = −0.30 (n=5), no rank signal**;
data/optuna_vs_full.png/.json (mapping caveats inside). Joint disposition:
neither cheap tier (short training OR shallow vehicle) rank-predicts full-scale
outcomes; screens are within-era knob-pickers and within-class mechanism
rankers, nothing more. All from existing receipts — ladder freeze honored.

CLOSING VERDICT (operator insight, 2026-07-17 afternoon): the cross-model
early-vs-late scatter is structurally invalid — **each model is its own
dynamical system with its own screen→full transfer coefficient**, so
between-model regression confounds model identity with screen score
(between/within-group problem; Simpson structure). Within one model the
trajectory IS monotone and early does predict late (amap crown series
758→1107→1511→champion); across models it cannot. Standing rules: (1) compare
early-vs-late ONLY along one model's own trajectory; (2) compare MODELS only at
matched late-stage conditions (H2H at deployment vehicle); (3) the scatter's
residual use = a map of per-model transfer coefficients (vertical distance from
the diagonal), not a predictor. Confirming instance: hyb's screen-tier Elo
measured directly vs SF = **938 (684..1112) @20g** vs 1704 implied by
duel-diff conversion — instrument-to-instrument transfer fails across mover
types too. Stats record: cross-model rho −0.30/−0.60, exact p 0.68/0.24,
underpowered (n>=20 pairs for 80% power); plots data/optuna_vs_full*.png.
FINAL CORRECTION (all-measured x, 2026-07-17 evening): with genuine same-currency
screen runs for all six winners (amap screen study best **710**, seeded from the
original arm5 amap seed; hyb SF-measured **938**), the apparent inversion
EVAPORATES: **rho = −0.09, exact p = 0.92** — screen and full-run Elo are
ORTHOGONAL cross-model, exactly as the per-model-dynamics verdict predicts. The
earlier negative rho was an artifact of the era-proxy and cross-instrument x's.
Per-model transfer lifts (the real payload): amap +1014, kc1670 +705, zca +560,
triv +462, vol +414, hyb +376. data/optuna_vs_full_final.png/.json.

SEARCH-DISTILL Phase A (:Distill-control:, 2026-07-21): CONFIRMED screen-grade.
One step of purity-compliant expert iteration — refit the amap-897 linear eval to its OWN
d5 native-minimax labels — beats the 1878 champion on TWO controlled instruments: head2head
d2-beam +798 (1.000/50g, champ-vs-champ ctl 0.517) AND fair native deterministic diverse-
opening duel d6 +176 (0.733, band excl 0.5; d4 +35 ns) — advantage grows with depth. Falsifies
the pre-registered linear-fixed-point risk. Two instrument traps recorded (spec/search-distill
.spec.md): (1) :Saturation-cut: raw ridge on ±1 labels blows eval scale 16x → native-unplayable,
fix = drop |y|>0.90 + ridge~100; (2) :No-native-duel-ruler: play_games handicaps the exploring
side only (champ-vs-itself=0.000), NOT a fair duel ruler. Artifacts: models/champion_distillA2.pt,
experiments/distill_linear.py, data/h2h_verdict_distillA.md. Next gate = native d9 anchor ladder
(operator-launched). Phase B (NNUE, same labels) gated; corrected-linear residual (atanh-RMS 0.11)
caps expected nonlinear headroom.

SEARCH-DISTILL FINAL (2026-07-22): CONFIRMED WIN — search-distillation expert iteration beats the
1878 TDLeaf champion by ~+240 Elo, native-d9 head-to-head (A2 +232, A3 +221, A4 +267; all bands
exclude 0.5; instrument control champ-vs-champ 0.490). Champion candidate = models/champion_distillA2.pt
(single distillation step captures the gain; iteration converges; SWA-average dilutes). Recipe +
promotion command in spec/search-distill.spec.md; full campaign data/search_distill_campaign.md.
KEY INSTRUMENT LESSON: the vs-SF anchor ladder DRAW-FLOODS strong agents (120-ply cap, no adjudication
-> 0.5 at upper anchors) and understated this to a false 20g "tie" (A2 1917 vs champ 1878 = +39 ladder,
but +232 head-to-head). For agents near/above the top anchor, the direct native head-to-head is the
true ruler, NOT the ladder — and NOT head2head (a d2-beam lens) NOR play_games (handicaps the explorer).

BACKLOG SWEEP P1-P7 (2026-07-22, ≤2 cores, all verdict-first):
- P1 deeper-teacher labels: NO — d5 absorption gave +232 vs the old champion, but d5->d7 collapsed to
  +47 (band incl 0.5, 30g); diminishing returns, linear eval search-consistent by d7.
- P2 halfKP nonlinear capacity (nnue_own on d5 labels): NO — lost 0.050 (-512 Elo, band excl 0.5) to
  the linear champion at matched d4 search; capacity inherits the teacher ceiling, does not extract more.
- P3 native NNUE Rust port: GATED OUT — won't native-port an eval that loses to the linear.
  => EVAL AXIS CLOSED: the search-distilled LINEAR amap champion is the endpoint (deeper labels AND
     more capacity both fail). Strength = search (alpha-beta), not eval class or label depth.
- P4 absolute Elo (adjudication + SF@2500 anchor): 1840 (1721..1958). Adjudication confirmed the
  draw-flood was real (champion 0-0-12 decisive vs SF@2500, previously scored 0.5) yet the ABSOLUTE
  number is consistent with the prior 1878 (overlapping) — proving the vs-SF ladder gives a rough
  absolute but is BLIND to the +240 head-to-head gain over the prior champion. H2H is the eval-improvement ruler.
- P5 random-bar/shadow feature audit: amap-897 is highly redundant — only 86/897 features beat the
  noise null; coverage-map features survive at ~10x the PST rate (53/128 vs 33/769), which is WHY amap
  beat pst-alone. Diagnostic only (correlation caveat: dropped=redundant, not useless).
- P6 search-arms MCTS×minimax-λ: NO/parked (prior campaign 0-16 vs native-d9 champion at full budget;
  native alpha-beta converts wall-clock ~13x better). Champion's alpha-beta is the winning search.
- P7 Prism: not a strength lever — init/transfer dominated by warm-start (own data); Mod-Wheel
  regularizer fits only the played-buffer anti-forgetting lane.
NET: champion (distillA2, search-distilled linear amap, ~1840 absolute / +240 H2H over predecessor)
stands as the endpoint on every axis tested. Next real levers would need MORE COMPUTE (deeper-than-
deployment labeling at scale) or a different modality, neither budget-feasible at ≤2 cores tonight.

ARCHITECTURE DISPOSITION — target network / critic / Dyna-Q (operator query, 2026-07-25):
- **No DQN-style target network, correctly absent**: the ":Dis-gamma:" "frozen" value
  (schedules.spec.md) is the native alpha-beta search's own returned leaf value (`gv`, stored
  at generation time), NOT a lagging/EMA copy of the net (grep of qlearn.py confirms no
  target_net/polyak/soft_update). Target nets exist to fix a self-referential bootstrap
  (Q(s,a) trained against Q(s',a') from the same shifting net); this project's targets are
  λ-returns anchored to actual game outcomes + search values, so that instability mode does
  not arise here. Adding one would solve a problem this architecture doesn't have.
- **No critic, correctly absent, already tried and closed**: actor-critic.spec.md's disposition
  reasoning explicitly REJECTS the textbook deadly-triad explanation — AC was closed via
  :Honest-expectation:, a pre-registered prediction that a pooled AC objective lands in the
  same ~650 Elo band as plain Q-learning because the actor stays a 1-ply reactive policy while
  search dominates strength. Later confirmed by P6 above (search, not eval/policy class, is
  the lever). No critic head exists in current code.
- **Dyna-Q — moot for this domain**: Dyna-Q's value-add is planning with a LEARNED model when
  the true model is unknown; chess's transition model is exact and known, so planning with the
  known model already IS the alpha-beta search (actor-critic.spec.md states this directly).
  MCTS-style rollout-planning was tried and lost to native alpha-beta ~13x on wall-clock (P6) —
  tested and closed, not an open gap.
- Bonus: experience replay DOES exist (uniform-sampled deque buffer + a second layered
  temperature-controlled elite-resampling mechanism, :Admixture-replay:, flag-gated off by
  default) — the one textbook DQN component genuinely present.
- Standing next lever (unexecuted, not part of this query but adjacent): the sims-scaling probe
  (2026-07-17 evening, above) already GO-justified a native rsearch port of `hyb`
  (λ-tree-backup + ab-leaf) for ~100x sims at equal wall-clock — derived, proven at screen tier,
  never executed, operator-gated.

NATIVE HYB PORT (2026-07-25): the standing lever above is now BUILT, not just derived.
`rsearch4::HybSearcher` (rsearch/src/lib.rs) implements :Lambda-tree-backup: + :AB-leaf: as a
Rust arena-tree PUCT search reusing the ALREADY-NATIVE `Eval` (amap-897 linear score) for
expansion values and the already-native negamax for the AB-leaf depth-d backed leaf — no new
eval code, only a new tree/selection layer over existing infrastructure. Exposed to Python as
`HybSearcher(weights, bias, seed).choose(fen, sims, lambda_backup, c_puct, t_prior, leaf_depth,
tau) -> (best_uci, gv, leaf_fen, predicted_reply_uci, nodes)`; wired into head2head.py's mover
dispatch as `san:<k=v,...>:<ckpt.pt>` (sits alongside the existing Python-tree `sa:` mover).
Gate results (bounded smoke, NOT the full pre-registered battery):
- G1 sign-convention: PASS (white-up-a-queen +0.32, black-up-a-queen -0.41 on the champion's
  own raw weights).
- G2 regression/self-consistency: PASS — lb=0/0.5/1.0 sweep across 3 FENs produces smoothly
  varying gv and, on the queen-endgame FEN, an actual DIVERGENT top move (lb=0.0 -> a1e5,
  lb=1.0 -> a1g7), confirming the M-blend is live and load-bearing, not a no-op.
- G3 thesis: PASS — 3-FEN battery (rook_fork_trap, queen_endgame, midgame_tactic) at lb=0.0 vs
  lb=1.0, sims=8 (low-sims regime where the mean/minimax gap should bite hardest): 1/3 FENs
  shows genuine top-move divergence (queen_endgame: lb=0.0 -> a1e5, lb=1.0 -> a1g7, gv shifts
  0.561->0.563), confirming the M-blend is mechanistically live, not a dead code path; the other
  2/3 correctly show NO divergence (most positions don't have a live mean/minimax gap — expected,
  not a failure). Deterministic across 5 seeds (move choice at tau=0 doesn't consume rng, as
  designed) — seed variation was the wrong axis to probe divergence on, the FEN choice is.
- G4 (UCB-V sanity): N/A — hyb's own arm space never uses ucbv (search-arms.spec.md); not
  ported natively.
- G5 smoke: PASS — 4-game live duel through `chessdq/head2head.py`, no crashes, no illegal moves.

SCREEN-TIER VERDICT (2026-07-25, :Tournament-and-verdict: step 2 — winner vs reference,
pre-registered winning params lb=0.5/leaf=ab2/c=1.5/tp=0.2 carried over from the ORIGINAL
Python-tree hyb screen, 2026-07-16): **native `hyb` CONFIRMED at screen tier vs the reference
beam mover, 50g, +275 Elo (95% CI +150..+401), band EXCLUDES ZERO, 66% decisive.** Same
direction and comparable magnitude to the original Python-tree result (+338, 2026-07-16),
now running ~13x cheaper per the same wall-clock-conversion advantage the champion's own
native alpha-beta already demonstrated over its Python predecessor (P6 above) — i.e. the
native port didn't just preserve the Python-tree's edge, it inherits the implementation-class
speedup that was the entire motivation for porting.
BUDGET-MATCHED FINAL VERDICT (2026-07-25, same session): ran the actual budget-matched probe
rather than deferring it. Calibration: champion's native d9 ≈1.07s/move on a representative
midgame FEN; native hyb (lb=0.5, leaf=ab2) reaches that same wall-clock at sims≈2000
(experiments/duel_hyb_native_d9.py, worker core experiments/_duelcore_hyb.py — mirrors
distill_iterate.py's `duel()` pattern, torch-free pool workers). Result, 6 deterministic
diverse-opening games (alternated colors): **native hyb LOSES 0-6 to the champion's own
native d9 mover, score 0.000 (95% CI 0.000..0.390), band excludes 0.5.** Per the operator's
own time-budget law ("200g claims run CANCELLED — would only tighten a CI on a decided
question"), no further games were run: 0-6 in the SAME direction and class as the prior
Python-tree finding (0-16, −597 Elo, 2026-07-17 ladder disposition above) is already decided,
not noise.
**This closes the open question, and refutes the re-open condition's own hypothesis**: the
2026-07-17 disposition speculated the ~13x class gap was a Python-vs-native wall-clock
artifact and that porting :Lambda-tree-backup: + :AB-leaf: INTO the native engine (done this
session) would let sims-scaling reach parity. It did not — even fully native, the PUCT-tree
loses to plain alpha-beta at matched wall-clock. The real mechanism: alpha-beta's pruning
explores exponentially fewer irrelevant nodes at a given depth, while every hyb "sim" pays a
fixed per-node cost (an ab2 leaf call is itself a shallow negamax) regardless of whether that
branch was ever going to matter — the inefficiency is structural to the tree-search paradigm
at this eval/domain, not a Python-implementation tax. This is a stronger, more specific
finding than the original P6/ladder disposition had available.
**FINAL PROMOTION DECISION: hyb (Python or native) stays PARKED, not promoted.** The
champion's own native alpha-beta d9 remains the production search — confirmed correct by
this session's own evidence, not merely inherited from the prior campaign. The native
`HybSearcher` engine and `san:` mover ARE kept (tested, gated, screen-confirmed +275 Elo
infrastructure) for a genuinely different re-open condition: :Lambda-target-training:
(mcts-as-training-target, spec/search-arms.spec.md) operates at the sims-matched screen
budget where hyb wins, not the wall-clock-matched deployment budget where it loses — that
lane is unaffected by this verdict and remains its own open question, separately gated.
