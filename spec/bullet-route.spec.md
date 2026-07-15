# :Bullet-route: — self-play corpus → NNUE-small → deep search → claims (goal-1600)

Governing spec for the goal-1600 program (operator goal 2026-07-13: RL model self-plays
to 1600+ Elo). Drafted as RECONCILIATION: corpus_gen.py, nnue_eval.py, the vol1600
bullet arm, and the mlp volume lane shipped ahead of this spec (contract deviation,
operator-flagged); this document now governs; future changes are spec-first again.

## Pipeline order (operator-corrected, NEVER invert)

1. SCREEN — studies/duels at toy scale measure POTENTIAL only (3-trial Optuna,
   600-game duels). Screening can never produce a claims-grade net (studies top
   ~950–1200; the 1540 record came from a 26-epoch full run).
2. GRADUATE — the strongest screened candidate gets the full-scale run, launched by
   the agent under explicit delegation (task #32). Trainer-lane arm: 30 epochs ×
   1000 games. Bullet arm: 10× corpus + retrain (training is ~23 s; DATA is the
   commitment axis).
3. VIEW — the graduated run's final Elo is the operator's deliverable.
4. CLAIM — the claims ladder runs on the GRADUATED net: 60g deep rung → 200g claims
   vs SF@1320; goal closes at 95% CI floor ≥ 1600.

## Definitions

- **Corpus** = bullet text format `FEN | score | result`, one line per TDLeaf SEARCH
  LEAF from native self-play (rsearch4.play_games, both sides = generator net,
  d2/d2, ε=0.1, ply cap 160). score = white-relative cp from the generator's own
  backed search value via cp = round(800·atanh(v)) (declared scale 400:
  sigmoid(cp/400) ≡ (v+1)/2), clip ±3000. result = (z+1)/2, z White-absolute.
- **Filter** (council verdict, data/council.md): drop in-check leaves (measured 9.6%).
- **Arm 1 (vol1600)** = jw1912/bullet stock simple.rs recipe replicated whole:
  (768→128)×2 dual-perspective SCReLU, AdamW, SCALE 400, WDL 0.75, StepLR. Declared
  departures: our data; steps scaled so one superbatch ≈ one corpus epoch.
- **Arm 2 (control)** = bullet-linear on the same corpus (pending; verdicts for any
  bullet net are RELATIVE to this control — regime confound rule).
- **Arm 3 (kingbucket)** = factorised input buckets (bullet progression example 3 /
  factorised.rs) — parked until arms 1–2 read out.
- **Evaluator** = nnue_eval.NNUEEval over the checkpoint's raw.bin (f32, SavedFormat
  order l0w/l0b/l1w/l1b; Chess768 map, stm-relative dual accumulators, SCReLU
  = clamp(0,1)²; value() returns White-absolute sigmoid-input units).

## Contracts

- **Require**: generator net = purity-clean self-play lineage (volume net: kc-809
  linear, ZCA back-conversion identity-gated); corpus fields all self-generated
  (purity law — SF is opponent/anchor only, never a label source).
- **Guarantee**: every trained arm is measured by the validated H2H instrument
  (spec/h2h-instrument.spec.md gates G1/G2b/G3) before any absolute rung is bought.
- **Maintain**: <15-min interactive gates; anything longer is a background lane;
  replicate-before-invent for every bullet recipe change (stock first, single-variable
  departures, declared).
- **Assert (gates, all measured 2026-07-13 for arm 1)**:
  1. ZCA identity gate: whitened-net value == raw-weight value on ≥3 positions (PASS,
     models/zca.npz).
  2. Corpus gate: ≥3 result classes, legal FENs, sane cp spread, decisive rate
     reported (PASS: 1.34M positions, 54% decisive games).
  3. Evaluator convention gate: corpus-label correlation split by side-to-move, both
     halves r ≥ 0.5 (PASS: +0.74/+0.57); sign agreement ≥ 80% at |cp|>300 (PASS:
     98.3%). OOD hand batteries are NOT valid gates for nets trained on narrow
     self-play distributions (pinned: 3-queen battery false-alarmed).
  4. Duel gate: arm vs TEACHER (its label source) 600g — student ≥ teacher-band =
     distillation sound; student > teacher beyond band = capacity gain at this data
     scale. (In flight: h2h nnue_vs_teacher.)

## Known data caveats (carried, not hidden)

- Corpus result skew: White 9% / draw 63% / Black 28% (ε-handicapped agent vs clean
  opponent asymmetry) — the net's priors inherit it (startpos value ≈ −0.58 in
  sigmoid units). Rebalance is an arm-2+ dial, not a silent patch.
- Generator strength (≈1500) unisolated in literature — mitigated by teacher-relative
  and control-relative verdicts only.

## :MCTS-door: (operator-reopened 2026-07-13 — second route to depth, no Rust merge)

Depth for nonlinear nets WITHOUT rsearch integration: python PUCT over the net's value
function. The prior PUCT-parity result is NOT evidence against this door — it ran a
weak value net with a flat prior at low sims, Grill et al. 2020's exact degenerate case.
- **Prior (declared, hand-crafted — purity-lawful)**: P(a) = softmax over the children's
  1-ply values at temperature T_P=0.2. Motivated by Grill (the prior is load-bearing);
  NOT a trained policy head (that stays parked as H2).
- **Search**: standard PUCT, c_puct=1.5 (declared), batched leaf evals, visit-count
  move choice with the duel dither (softmax over visits, τ_v=0.02·N_sims).
- **Gate ladder (each <15 min or background)**: G-M1: PUCT(200 sims) vs d2-beam over
  the SAME eval, 100g duel — MCTS must convert the identical eval into MORE strength,
  else the door closes again with a number. G-M2 (if open): sims ladder 200→800.
  G-M3: 60g SF@1320 rung; then the standard claims ladder.
- Rollout-mixed leaf eval (Q8-endorsed) is a declared OPTIONAL dial, OFF at G-M1.

## Verdicts (2026-07-13/14 — screening tier CLOSED, decision made)

- Teacher duel: bullet arm 1 **−62 (−90..−34)** vs volume teacher at d2-beam → arm 1
  does not graduate at this corpus size.
- G-M1: PUCT-200 vs d2-beam, SAME nnue eval: **0.970 → +604 (+414..+794)** — the MCTS
  door works as search (operator's claim confirmed: selective deepening crushes the
  shallow beam).
- Fork smoke: PUCT-200+nnue vs **rsearch-d7+linear incumbent: 0/8** — python MCTS at
  200 sims cannot bridge the native engine's ~1000× node throughput; with the eval
  also −62, the fork closes. MCTS-door stays open as a future vehicle for nonlinear
  nets pending native/GPU-batched search (revisit-later bucket, operator-blessed).
- **GRADUATED-RUN DECISION** (task #32, operator-delegated): vehicle = rsearch d7
  (unbeaten); net = kc-809 linear (the ONLY rsearch-compatible arch — pst-769 is
  769-dim and cannot ride the native engine); config = the RECORD arm's recipe
  (1540 claims): kc+ZCA(models/zca.npz) linear, KC-faithful TDLeaf d2 native targets
  (RSEARCH_DEPTH=2), tuned trivium anneal (0.285,0.341,0.374 → 0.516,0.341,0.143 @
  0.481), α=3e-4, PARGEN native self-play (batch 200, 12 threads, ε=0.1, opp=frozen
  self), confirm-on, patience 4, 30×1000 games, seed = models/qlearn_wseed.pt
  (pristine whitened — provenance caveats carried exactly as the ledger recorded them
  for the record arm; env reconstructed from experiments.md 2026-07-11 entries, knobs
  not stated in prose take module defaults, RECONSTRUCTION declared).
  Deliverable: final Elo for operator view → claims ladder (60g d7 rung → 200g).

## :Replication-bridge: (operator-mandated 2026-07-14 — the pipeline must repeat 1670)

Operator finding: 1670 was produced half-in-console, half-in-session (seed step and
deep-measure step lived outside the Start button) — exploration without a replication
pipeline. Bridge (server.py, console):
1. **Seed-from-scratch**: a FRESH lineage ckpt is created by copying `seed_ckpt`
   (default models/qlearn_wseed.pt — the champion run's exact from-scratch entry point)
   instead of random init. Existing lineages resume untouched (ratchet mode).
2. **Auto-ladder**: when training exits, the banked `_best` automatically runs
   claims_rung.py at `post_rung_depth` (default 9) × `post_rung_games` (default 60) —
   the row lands in rl_trend, moving the ladder plot and the Champion tile with no
   manual step. (Watcher is a server daemon thread; a server restart mid-run drops it —
   re-run manually.)
3. Console defaults = full champion recipe + lineage `rep1`: the out-of-the-box button
   IS the from-scratch replication experiment (wseed → 30×1000 → bank → d9 rung).
4. **Crown live-rungs** (`crown_live_rung`, default on): during training, each newly
   banked confirmed crown (+0.5 bar or more) fires a ≤24-game deep rung tagged
   `live-rung (provisional)` — goal-scale progress visible on the ladder DURING the
   run. Provisional because SF's clock shares the CPU with the trainer; the idle-box
   post-run rung is the honest number.
5. **Auto-claims escalation** (`auto_claims_at`, default 1550; 0=off): if the post-run
   60g scout's Elo ≥ the threshold, the 200-game claims run fires automatically —
   the UI alone can mint a claims-grade 1600+ number (operator mandate 2026-07-14:
   "mimic these results in our training ui"). Tag `<lineage> d<depth> CLAIMS auto`.
6. **Auto-promote** (`auto_promote`, default OFF): only when the auto-claims row's CI
   FLOOR strictly exceeds every prior ≥200g floor on the ladder does the lineage best
   copy to models/champion.pt (logged). Default off — promotion is normally the
   operator's call; the checkbox is the delegation.

## Acceptance

Bullet route graduates to step 2 (10× corpus) only if a bullet arm beats BOTH its
teacher and the bullet-linear control beyond the duel band. Otherwise the graduated
run goes to the best trainer-lane candidate (mlp volume lane vs volume-net incumbent).

## Parked: :Backup-temperature: (operator concept 2026-07-14 — lambda between backup operators)

Operator insight: blend minimax (max-backup) and MCTS (mean-backup) with a lambda, like
eligibility traces blend n-step estimators. Canonical form: SOFT BACKUP — log-sum-exp /
power-mean with temperature tau_b (tau_b->0 = minimax, ->inf = MC mean). Published:
Power-UCT (Dam et al. IJCAI 2020), MENTS (Xiao et al. 2019). In-repo motivation: the
TDLeaf maximization-bias collapse (max-backup amplifies eval error — measured); the
trivium/Q8 blends (house pattern: anneal trust dials).
- CHEAP test (pre-registered, fire on operator go after the SME-feature verdicts):
  organ study — training TARGETS computed with tau_b-annealed soft-backup over the d2
  beam values (mean-ish early, max-ish late), single knob, standard 3-trial protocol +
  duel-ruler verdict vs the max-backup control. Organ base rate 0/5 — expectations set.
- EXPENSIVE half (play-time hybrid backup on sparse trees): parked behind the standing
  bar — must beat native d9 minimax at equal compute; requires native-side work either
  way (the 1000x python/native throughput gap dominates any operator gain).

## :Operator-ideas-ledger: (2026-07-14 — every concept documented, statused, testable)

| # | Idea (operator phrasing) | Spec home | Status |
|---|---|---|---|
| 1 | Scores are FEATURES not distillation (hand evals = declared feature defs) | this file :SME-features: framing | doctrine, adopted |
| 2 | Threat/guard piece-square interactions (highlight engine as encoder) | qlearn/encoders `tpst` | duel RUNNING |
| 3 | Hanging conjunction (threatened AND unguarded) as input plane | encoders `hpst` | duel queued |
| 4 | Attack maps = learned mobility/space/center (no hand square-weights) | encoders `amap` | duel queued |
| 5 | Defender DEPTH (first line + who is behind: battery/x-ray support) | arm-D, pre-registered | parked on 2-4 verdicts |
| 6 | Quantized float counts (attacker/defender counts, not binary) | arm-D | parked on 2-4 verdicts |
| 7 | %-of-board-held scalar | arm-D | parked on 2-4 verdicts |
| 8 | QLoRA-style small dense adapter over the sparse base | capacity follow-on | parked (post-linear screen) |
| 9 | Threat features = 1-ply knowledge injected at EVERY search node | consequence clause, :SME-features: | automatic if 2-4 win + native port |
| 10 | Better tactical leaf eval can unlock MCTS (trap-blindness fix) | :MCTS-door: re-entry conditions | parked (needs native/GPU speed too) |
| 11 | Lambda BETWEEN search operators (minimax<->MCTS backup blend) | :Backup-temperature: (Power-UCT/MENTS) | parked, organ test pre-registered |
| 12 | Sufficiency over best (Elo bars + cost-ranked vehicles) | bakeoff tie->cost rule | doctrine, standing |
| 13 | Dynamic difficulty at player+1sigma (regret EMA + variance) | dynamic-difficulty :Sigma-offset: | LIVE (gates passed) |
| 14 | Depth ladder push (d9 "is fine") | claims ladder | RECEIPTED: 1572->1670 |

Rule: an idea enters this ledger when voiced, gets a spec home + pre-registered test
before any claim, and its status moves ONLY on measured verdicts (the honest-asterisk
register applies here too — "winner" is a status this table assigns, not a prior).

## :Played-buffer: (operator 2026-07-14 — idea #15, learn from human games)

KnightCap-canonical (the donor engine trained on FICS humans): the human is an OPPONENT
rung (same legal class as the SF ladder — shapes states, never labels); every training
label is self-generated (own d2 search values + outcome, proven trivium blend 0.285/
0.341/0.374, lambda 0.7). Pipeline, all operator-reachable (entrypoint law):
1. Finished games auto-archive to data/human_games/ (terminal_board, PGN + result +
   colors) — archive is part of Play, zero clicks.
2. Menu option 5 "Learn from my games" -> human_replay.py: fine-tunes a champion COPY
   (models/champion_hb.pt) on the buffer. Gate PASSED on 3 varied synthetic games
   (win/loss/draw; loss decreases; candidate written).
3. Promotion is DUEL-GATED, never automatic: champion_hb must beat champion on the
   ruler (600g), then the ladder. Honest scale note: single games are noise vs the
   30k-game corpus — value accrues over tens of games; human games are distribution
   coverage self-play cannot generate (the hole-coverage lever).
Also: :Sigma-offset: dial exposed in Difficulty (operator: +1sigma felt soft — the
sigma is now a prompt, 1.5-2 = harder tracking).
