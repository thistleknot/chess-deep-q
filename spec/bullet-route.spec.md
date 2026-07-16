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

## :Thermal-guard: (operator 2026-07-15 — mid-claims-run thermal reboot)

Every hours-long lane self-caps at entry: `thermal.engage()` = half the logical
cores (affinity, inherited by children incl. Stockfish) + below-normal priority +
matched torch thread pool. Wired into `claims_rung.py`, `head2head.py`,
`qlearn.py` mains. Override: `CHESS_THERMAL_FRACTION` (0..1), `CHESS_THERMAL_OFF=1`.
Guarantee: the operator's foreground use always preempts a lane; a lane never
holds 100% of the package for hours. The guard is operational only — it must
never change results, only wall-clock.

## :Factorial-tournament: (operator 2026-07-15 — "sport like tournaments")

Operator design: test lanes SEPARATELY then in COMBINATION — a 2x2 factorial
{base pst, amap features} x {max backup, soft backup tau_hi=0.5}, all SEED=1 s200
matched protocol, settled by ROUND-ROBIN (every net duels every net, 50g/pairing
per the games-cap law, standings table = the verdict artifact). Cells: base+max =
arm5_pst_b, base+soft = arm5_sb_b (tie head-to-head vs base+max), amap+max =
arm5_amap_b (confirmed winner), amap+soft = arm5_amapsb_b (TRAINING). amaph
(composition) joins the bracket when its verdict posts. Key question: does the
operator's backup-lambda pay ON TOP of the confirmed features (interaction),
even though it tied on the weak base? Caveat declared: 50g pairings resolve ~±95;
standings read as trends, each net's pooled 4-match record (200g) tightens its
overall read.

VERDICT (2026-07-15): tournament winner amap+soft (seed-1: +108 (+8..+207) over
amap, band excludes zero) FAILED seed-2 confirmation: -35 (-130..+61), point-sign
flip; pooled 100g = +35 (-33..+103) spans zero => per pre-registration the stack
is UNCONFIRMED (likely seed-1 luck; mechanism base rate 0/7). amap alone stays
the only confirmed winner and the native-port payload. :Backup-temperature:
remains implemented + flags-off; reopening = explicit operator ask at a finer
H2H_CAP resolution.

## :Backup-temperature: (operator concept 2026-07-14 — lambda between backup operators; IMPLEMENTED 2026-07-15, cheap-test QUEUED)

Operator insight: blend minimax (max-backup) and MCTS (mean-backup) with a lambda, like
eligibility traces blend n-step estimators. Canonical form: SOFT BACKUP — log-sum-exp /
power-mean with temperature tau_b (tau_b->0 = minimax, ->inf = MC mean). Published:
Power-UCT (Dam et al. IJCAI 2020), MENTS (Xiao et al. 2019); the tau_b operator used is
MELLOWMAX (Asadi & Littman 2017 — mean at tau->inf without the log-n divergence). In-repo
motivation: the TDLeaf maximization-bias collapse (max-backup amplifies eval error —
measured); the trivium/Q8 blends (house pattern: anneal trust dials).
- IMPLEMENTATION: `QLEARN_SOFT_BACKUP=<tau_hi>` (0=OFF, flags-off-inert PROVEN on a
  3-position battery + 3-value-set mellowmax limit battery). gv in search_policy.
  search_move (python d2 beam TDLeaf path ONLY — asserts RSEARCH_DEPTH=0) becomes
  mellowmax over the BACKED root values; tau_b anneals tau_hi -> 0 on the shared
  :anneal:/WARMUP schedule (mean-ish early, max late — the trace analogy). Move
  CHOICE untouched; only the bootstrap value changes.
- CHEAP test VERDICT (2026-07-15): arm5_sb_b (SEED=1, s200, tau_hi=0.5) vs matched
  max-backup control arm5_pst_b — **50g score 0.500 -> +0 Elo (95% -95..+95): no
  detectable effect at the operator's 50g spend** (games-cap law; effects < ~±95
  invisible at this resolution). Trained stably, 96% decisive. CLOSED at one trial
  per pre-registration; re-opening at a finer resolution requires an explicit
  operator H2H_CAP raise. Organ base rate now 0/6.
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

## :Feature-autoresearch: (operator delegation 2026-07-14 — canon-sourced hypothesis queue)

Standing loop: when a verdict batch lands on the duel ruler, the next queued arm is
built (encoders.py + gate battery + fresh seed), trained at the matched protocol, and
dueled vs the control — WITHOUT a per-arm operator ask (explicit delegation for this
feature campaign; operator may halt anytime). Queue sourced from the canonical
inventory (chessprogramming.org/Evaluation) minus families already covered (material/
PST = base; threats/hanging = arms A/B; mobility/space/center = arm C subsumes):

| # | Arm | Encoder sketch | Canon note |
|---|---|---|---|
| E1 | pawn-structure planes | per-pawn-square: PASSED / isolated / doubled indicator planes | canon's top non-material family; passed pawns = the endgame lever |
| E2 | king-safety counts | pawn-shield count (0-3) + enemy attackers in king zone (quantized) | canon "attack units"; operator idea #6 (quantized floats) applied |
| E3 | piece-specific trio | rook-on-open/semi-open file per rook; bishop-pair flag; knight outposts | classic engine terms, all conjunctions |
| E4 | defender-depth | attackers-minus-defenders count per occupied square (incl. batteries) | operator idea #5, Giraffe-adjacent |
| E5 | per-piece mobility | quantized legal-move count per piece-square | trapped-pieces canon; Giraffe input class |

Rules: single concept per arm; gate battery >=3 varied positions before any training;
duel verdict vs the SAME fresh-seed control; band +-~30. STOP conditions: two
consecutive arms tie/lose -> pause and council before continuing; any arm WINS ->
freeze queue, compose winner(s) marginally, plan the native port (winners must ride
d9 to matter for claims). Honest base rate carried: features are 0-for-N vs raw
planes so far; arms A/B/C verdicts pending.

## Ledger idea #16: :Retreat: (operator 2026-07-14 — coverage-delta move concept)

Definition (operator): a move that saves a threatened piece while NOT advancing net
coverage — measurable as delta(covered squares, protected piece-points) <= 0 with no
material change. Placements:
1. STATE ingredients = linear spans of arms A (guard planes -> protected material with
   LEARNED values) and C (attack maps -> coverage count) — already under duel; no new
   arm needed while those are live.
2. **E6 coverage-scalars (queued, PRIORITY-FIRST if A/B/C lose on dilution):** the
   compressed projection — per-side covered-square count + per-piece-type protected and
   threatened counts (~24 quantized floats vs ~1500 planes). Dilution-proof form of the
   same information (kpst lesson applied).
3. Move-level retreat DETECTOR: lawful as a UI/coaching cue (label the human move next
   to the regret readout) and as a search move-ordering prior (behavior only). BANNED
   as reward shaping (S&B chess clause: rewarding subgoals teaches subgoal-hacking —
   sometimes the retreat IS the best move; the value function prices that itself).

## Ledger idea #17: :Move-trichotomy: (operator 2026-07-14 — completes #16 into a partition)

Every move classes as ADVANCE / NEUTRAL("draw") / RETREAT via the sign of the #16
coverage+material delta; two binary flags encode all three (operator: "two classes
covers all 3 states"; operator also pre-applied the no-reward rule unprompted).
Placements: (1) FREE derivative of E6 scalars (sign of delta — one line); (2) coaching
UI cue + played-buffer analytics (move-class mix vs regret); (3) search move-ordering
prior (behavior only). Declared caveat: as a STATE-value feature it is Markov-redundant
(the position screens off the last move; canon cousin = tempo bonus) — if ever tested,
a 2-bit E6 add-on with expectations set low. NEVER reward (operator + S&B agree).

### Grounding check for #16/#17 (web, refutation-first, 2026-07-14)

- #16 coverage maps = canon "space advantage" (controlled-square counts; SF classical
  computed piece-activity/space intermediates); published analysis lens: "Statistical
  analysis of chess games: space control and tipping points" (arXiv 2304.11425) —
  VALIDATED precedent, E6 = space terms with learned prices.
- #17 advance-before-retreat move ordering is EXISTING engine practice (move-ordering
  canon: advancing generated before retreating; captures/killers/history taxonomy);
  coaching-classifier cousins exist (eval-delta based "brilliant/blunder" tools) —
  coverage-basis variant appears untried. Markov-redundancy caveat unchanged.

## :SME-features: batch-1 verdicts (2026-07-14, duel ruler, 600g each vs fresh-seed control)

- tpst threat/guard planes (+1536d): **-10 (-38..+17) TIE** — closed at this screen
  (declared caveat: d2 duels subsume 1-ply threat knowledge; deep-lens re-screen =
  optional follow-up).
- hpst hanging planes (+768d): **+15 (-13..+43) TIE, leaning** — held for the
  composition arm.
- **amap attack maps (+128d): CONFIRMED 2026-07-15 — seed 0: +51 (+23..+79); seed 1:
  +72 (+44..+101); two independent seeds, both bands exclude zero (pooled ~+60). The
  FIRST CONFIRMED feature win of the campaign** (operator's coverage-of-territory
  concept as learnable maps). Batch is monotone in information-per-dimension (dilution
  theory holds). Fired per pre-registration: (1) :Composition-arm: `amaph` encoder
  (amap 897d ⊕ hanging 768d = 1665d, encoders.py, prefix/suffix + positive-control
  battery PASSED) — arm5_amaph VERDICT 2026-07-15: **50g score 0.600 -> +70
  (95% -27..+167) vs the confirmed amap winner — leaning positive, band spans zero
  at the games-cap spend; joins the :Factorial-tournament: bracket where its pooled
  record adds resolution**; (2) native
  port planning so amap can ride the d9 claims engine — the path to a features-equipped
  champion challenger. E-queue unfreezes per win-rule (E6 coverage-scalars next).

## :Dmap-screen: (operator 2026-07-15 evening — "test those features out now just like we did before")

Feature: **dmap destination maps** — per-square CAN-MOVE-INTO bits per side
(operator's per-piece move-availability concept; E5 was its count compression).
Definition (PSEUDO-legal, declared): non-pawn destinations = attacks minus own
men; pawn destinations = captures onto enemy men + single/double pushes through
empty squares; NO en passant, NO castling, pins/check ignored (same
legality-blindness as amap's is_attacked_by — and a free native port later).

Pre-registration (exact arm5 protocol, SEED=1 matched, seeds torch.manual_seed(20)):
- Arms: `arm6_amap` (897, fresh control = current-best encoding), `arm6_dmap`
  (897: pst-769 ⊕ dmap-128), `arm6_amapd` (1025: amap ⊕ dmap composition).
- Train: `qlearn.py 200 1`, KC_FAITHFUL=1 OPP=graded EPOCH_ELO_GAMES=12
  CONFIRM=1 PATIENCE=99, python-beam targets (RSEARCH_DEPTH=0), via lane.py.
- Duels (50g law, sharded): D1 `arm6_dmap` vs `arm6_amap` (is mobility ≥
  coverage?); D2 `arm6_amapd` vs `arm6_amap` (does mobility ADD to the best?).
- Verdict rule: band excludes zero → seed-confirm fires (SEED=2 pair) before
  any native-port/champion-recipe step; band spans zero → no detectable effect
  at the 50g spend, park (screens kill, never confirm).
- Encoder gate battery (pre-launch, ≥3 varied FENs): startpos push/knight/no-
  rank-1 dests; blocked single+double pushes; pawn-capture-needs-enemy; black
  mirror; amapd prefix==amap / suffix==dmap positive control.

VERDICT (2026-07-15 evening, both 50g sharded, input SHAs on the verdict lines):
- D1 dmap vs amap: **−35 (95% −130..+61), 98% decisive — TIE, leaning under**;
  mobility alone does not beat coverage.
- D2 amapd vs amap: **+0 (95% −95..+95), 92% decisive — DEAD TIE**; mobility
  adds nothing detectable ON TOP of coverage at the 50g spend.
Both bands span zero → per pre-registration NO seed-confirm fires; dmap PARKS.
Consistent with dilution theory + prior evidence: non-pawn destinations ≈
attacked squares (high overlap with amap bits), E5 counts were already flat,
amaph (hanging⊕amap) was flat — amap appears to already carry the coverage
signal. Re-open only at a finer resolution (explicit operator H2H_CAP raise)
or with the divergent-bits-only variant (dmap XOR amap: pushes/blockers only,
~2×64 sparse) as a fresh E-queue entry.
Incident note: the serialized launcher's queued duplicate dmap lane slipped the
conflict guard (its twin was already 'done') and was killed pre-write;
arm6_dmap_best restored from the confirmed lane-12 save. Guard gap logged:
conflict check compares against RUNNING rows only — queued duplicate launches
of COMPLETED work are not refused. Mitigation: never queue launches behind a
blocking `lane.py run`; one lane = one launcher process.

## :Lane-registry: (2026-07-15 — operator: "serious code quality problems")

Root cause of the zombie incident: chains triggered by process-name greps +
duels reading mutable shared ckpts + no run ownership. Fix: `lane.py` (sqlite
data/lanes.db) is the ONE owner of background runs — `lane.py run --tag --cores
[--after id|tag] [--inputs] [--outputs] -- cmd` registers, guards conflicts
(overlapping outputs or cores vs RUNNING rows => REFUSED), waits on REGISTRY
state (never process greps), pins cores, marks done/failed. `ls` / `eta`
(measured from the run's own log pace — quoted ETAs must come from here) /
`reap [--kill]` (dead rows marked; unregistered qlearn/head2head/claims
processes flagged). Rules: (1) first action after any reboot/session start =
`lane.py reap` (server.py boot does it automatically); (2) the PowerShell
wait-loop chain pattern is RETIRED; (3) satisfies :Replication-requirement: —
lane knobs live in the registry + console Lanes tile (/api/lanes), not session
commands. Verified: conflict battery 3/3, reap, snapshot isolation, measured
eta, tile live+idle.

## :Native-amap-port: (2026-07-15 — goal: "champion built on my features")

The confirmed amap encoder now rides the native engine: rsearch4 v3.7-amap accepts
897-dim weights (mode = weight length; [769:833] white-attacked / [833:897]
black-attacked square bits, computed from the attack unions the eval already built
for mobility/hung — near-zero cost). PROVEN: python/native parity 0.00e+00 on a
5-position battery incl. castling-rights/endgame/tactical FENs; kc-809 path
regression-clean (diff < 1e-8); d5 search 0.01s. Plumbing: corpus_gen.raw_weights
passes amap ckpts (unwhitened — no 897 ZCA exists, declared) straight through;
qlearn sync_rsearch/PARGEN accept ENC=amap (smoke: native d2 targets + 2
parallel self-play games); console enc selector gained amap (replication law).
GRADUATED RUN amap1600 (lane 8): champion recipe exact (KC-faithful TDLeaf
RSEARCH-d2 native targets, trivium anneal 0.285,0.341,0.374 -> 0.516,0.341,0.143
@ 0.481, alpha 3e-4, warmup 0.4, PARGEN 200, confirm patience 4, 30x1000, fresh
amap seed) with declared departures: enc=amap unwhitened, pargen_threads 8
(thermal law). Chained: lane amap1600_rung = d9 60g scout on the banked best.
Challenge gate: promotion still requires a claims-grade floor above the
champion's 1605 — the games spend for that final gate is an explicit operator
decision under the 50-game law.

### :Native-amap-port: VERDICT — PROMOTED 2026-07-15

amap1600 d9 scout: 49W-1D-0L over 50 games (serial protocol; floor 1724) — clears
the 1605 gate under the operator games-cap law. models/champion.pt = the amap-897
net (prior kc champion backed up: models/champion_backup_kc1670.pt); play-menu
label updated; load+move verified. The +10 pooled finisher games merge into the
trend row asynchronously (can only confirm at this score). GOAL CLOSED: the
champion is built on the operator's features. Follow-up defect: human_replay.py
asserts enc=="kc" — port the played-buffer lane before next use.

## :Arm7-screen: (2026-07-16 — operator: "8 hours... think triz + six hats, design experiments, tournament style")

Meta-read driving the design: representation beats algorithm (features 1-for-4
confirmed — amap; mechanisms 0-for-7), and every ADD-ON tested on top of amap
so far has tied at the 50g resolution (amaph leaning, amapd dead, dmap under).
So: three FEATURE arms, chosen by TRIZ move, all screened against the standing
arm6_amap control (same protocol, same seed — reused, not retrained):

| Arm | Dims | TRIZ move | Question |
|---|---|---|---|
| `arm7_kamap` (kc-809 ⊕ amap-128) | 937 | merging | K1: do the KnightCap hand terms (king safety, pawn structure — the OLD champion's edge) add on top of coverage? Highest prior: both blocks independently confirmed; lowest expected bit-overlap of any candidate. future_exploration #1. |
| `arm7_cmap` (pst-769 ⊕ count-maps-128) | 897 | local quality | C1: do graded attacker COUNTS beat binary coverage at EQUAL dims (info-per-dim, ledger idea #6)? count = popcount(attackers)/4, declared. |
| `arm7_amaps` (amap-897 ⊕ E6-scalars-24) | 921 | segmentation/compression | S1: does the compressed protected/threatened signal add on top of coverage? E6 per :Retreat: — per-side × per-piece-type PROTECTED and THREATENED counts (/8, 24 floats). The dilution-proof retry of the leaning hpst arm. Covered-square COUNT deliberately DROPPED: it is a linear sum of amap bits — inside a linear net's span by construction, adds nothing. |

Door NOT taken, with the argument logged: dmap-XOR-amap re-basing is span-
equivalent to amap⊕dmap for a linear model — amapd (+0 dead tie) already
measured that function space; do-not-retry holds without a new instrument.

Pre-registration (exact arm5/arm6 protocol, SEED=1 matched):
- Control: `models/arm6_amap_best.pt` (trained 2026-07-15 at this exact
  protocol) — no retrain; duels are seed- and protocol-matched.
- Seeds: torch.manual_seed(20), fresh ValueNet('linear', 64, nin), bundle
  {state_dict, arch, enc, zca:False, cum_games:0} → models/arm7_<name>.pt.
- Train: `qlearn.py 200 1` via lane.py; KC_FAITHFUL=1 OPP=graded ELO_GAMES=0
  EPOCH_ELO_GAMES=12 CONFIRM=1 PATIENCE=99 SEED=1 RSEARCH_DEPTH=0
  (python-beam targets); one lane = one launcher process (lane-14 lesson);
  cores 0-2 / 3-5 / 6-8 (thermal law, 9/12).
- Duels (50g law, sharded H2H_SHARDS=6, sequential on cores 0-5):
  K1 kamap vs arm6_amap; C1 cmap vs arm6_amap; S1 amaps vs arm6_amap.
- Verdict rule (unchanged): band excludes zero → SEED=2 confirm pair fires
  before anything else; band spans zero → park, ledger, do-not-retry.
- Gate battery per encoder (≥3 varied FENs) before any training: kamap
  prefix==kc/suffix==amap; cmap bit-consistency with amap (count>0 ⇔ bit)
  + hand-counted startpos values + black mirror; amaps prefix==amap +
  hand-counted startpos scalars (8/8 pawns protected → 1.0, rooks 0,
  threatened all 0) + mirror.

### :Arm7-screen: VERDICT (2026-07-16, all 50g sharded, input SHAs on verdict lines)

- K1 kamap vs amap: **−42 (95% −138..+54), 100% decisive — TIE, leaning under.**
  The merge of the two confirmed winners does not beat amap alone. Council's
  pre-logged caution stands: KnightCap terms are attack-table-derived, so the
  kc-809 block is largely overlap + 809 dims of dilution at the 200g spend.
- C1 cmap vs amap: **−147 (95% −251..−44) — LOSS, band excludes zero (under).**
  Graded attacker counts LOSE to binary bits at equal dims. Declared confound:
  /4 quantization shrinks single-attacker signal to 0.25 vs amap's 1.0 at the
  same α — kill is (no-extra-info OR scale-attenuation); parks either way.
- S1 amaps vs amap: **−28 (95% −123..+68), 100% decisive — TIE.** The
  dilution-proof E6 scalars (24 dims) still add nothing detectable.

All three park per pre-registration. Feature-add-on failures on top of amap now
number SIX (amaph, dmap, amapd, kamap, cmap, amaps). :Feature-autoresearch:
STOP condition fired → council. Council read: the common factor is the SCREEN,
not the six features — H_data: 200 training games under-trains any encoder
larger than the control; the screen resolves only amap-sized (~+60) effects.
Wave 2 = instrument probe, pre-registered below.

## :Arm7b-data-probe: (2026-07-16 — instrument experiment; never writes rl_trend)

Question: does the 200-game screen spend HIDE feature wins? Single factor
changed vs :Arm7-screen:: training games 200 → 600 (`qlearn.py 600 1`), all
else identical (SEED=1, same seeds re-used from models/qlearn_<enc>_seed.pt,
fresh arm7b checkpoints).
- Lanes: `arm7b_amap` (fresh 600g control), `arm7b_kamap`, `arm7b_amaps`
  (the two wave-1 ties; cmap excluded — its kill was signed, not a tie).
- Duels (50g law): kamap600 vs amap600; amaps600 vs amap600.
- Readout: if a tie band MOVES materially positive at 600g, H_data is
  supported — every prior 200g park gains an asterisk and future screens move
  to the 600g spend; if bands stay ~0, H_data is refuted for these features
  and the parks are clean.
- Also: round-robin among the wave-1 arms (kamap/cmap/amaps pairings, 50g
  each) to complete the operator's tournament-standings artifact; pooled
  150g/net records read as trends only.

### :Arm7-screen: round-robin standings (2026-07-16, 50g/pairing, trends only)

| # | Net | Pooled /3 matches | Notes |
|---|---|---|---|
| 1 | amap (control) | 1.80 | holds the bracket |
| 2 | kamap | 1.69 | beat cmap +108 (+8..+207), amaps +70 (−27..+167) |
| 3 | amaps | 1.48 | beat cmap +85-lean |
| 4 | cmap | 1.03 | lost all three pairings — signed loser |

Fully transitive ordering; bracket agrees with the control-duels. arm7b
600g-probe lanes (26/27/28) launched on the same seeds.
