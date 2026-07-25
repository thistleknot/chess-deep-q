# Search-arms screen — decision-time search over the frozen champion evaluator

Layer: Behavioral (ordered search/backup logic) + Structural (mover spec, Optuna driver).
Status: pre-registered 2026-07-16 (operator directive: "make sure we test it if we haven't"
+ 3-parallel-lanes / Optuna-tournament / vs-baseline protocol).

## :Search-arm-screen:

Three decision-time mechanisms, one engine (`chessdq/puct_value.py`), evaluator FROZEN at
the champion (amap-897 linear, `models/champion.pt`, floor 1724). Nothing retrains; this
screens whether a different *search shape* extracts more strength from the same value
function than the incumbent alpha-beta / d2-beam movers.

| Arm | Mechanism | Prior evidence |
|---|---|---|
| `hyb` | :Lambda-tree-backup: + :AB-leaf: | **CLOSED 2026-07-25.** Native port built + gated + screen-CONFIRMED (`rsearch4.HybSearcher`) — all gates pass, 50g vs reference +275 Elo (band excl 0) at sims-matched budget. Budget-matched vs champion's native d9: LOSES 0-6 (band excl 0.5) — reconfirms the prior Python-tree class-gap finding, now proven NOT a Python-vs-native artifact (the native port still loses at matched wall-clock). PARKED, not promoted; champion's own native d9 stays the production mover. Engine kept for the separate :Lambda-target-training: (sims-matched) lane. |
| `puc` | plain value-PUCT, champion leaf | PUCT ran only with the weak tower net (parity) — rematch |
| `ucv` | :UCBV-selection: | UNTESTED (no variance term anywhere in repo search code) |

**Distinction from the parked backup-λ ledger entry** (`dispositioned.md` Feature-screen
ledger): that arm was mellowmax SOFT backup applied to the TRAINING bootstrap target.
:Lambda-tree-backup: is a HARD mean↔minimax blend applied to the SEARCH tree's node
values at inference. Different placement, different operator — a changed variant under
the ledger's re-open rule, not a plain re-run.

## :Lambda-tree-backup:

Node value used by selection becomes `V = (1-λb)·mean + λb·minimax`:

```text
Maintain per node (White-absolute, matching puct_value.py convention):
  N, W            # visit count, summed backup values (mean stats — unchanged)
  M               # minimax value of the subtree
Initialize:
  child M  <- its expansion-batch value (terminal exact; else vf 1-ply value)
  node  M  <- leaf evaluation at expansion
Update (on backup path, after mean stats):
  M(node) = max over children M   if node.board.turn == WHITE
          = min over children M   if BLACK
  (only expanded/initialized children participate; all children are initialized
   at parent expansion, so M is total)
Selection Q:
  q_eff = (1-λb) * W/N + λb * M
```

- Require: `0.0 <= lambda_backup <= 1.0`.
- Guarantee: `lambda_backup=0.0` reproduces the current engine's move choice
  bit-for-bit (regression gate G2).
- Guarantee: `lambda_backup=1.0` at a trap position (one refutation reply) scores the
  trap move by its refutation, not by the average reply (thesis gate G3).

## :AB-leaf:

`leaf="ab:<d>"` replaces the vf 1-ply backed leaf value with the native searcher's
depth-d backed value: `v_leaf = tanh(rsearch4.Searcher(w,b).search(fen, d)[1])`,
White-absolute, same tanh scale as the vf leaf. Priors still come from the vf
expansion batch (cheap). d ∈ {1,2,3}.

- Require: inner ckpt is amap/kc linear (raw_weights-compatible); position non-terminal.
- Assert (gate G1): searcher score sign agrees with the Python vf on White-winning /
  Black-winning / equal positions (3+ FENs) BEFORE any lane launches.

## :UCBV-selection:

`selection="ucbv"` (UCB-V, Audibert–Munos–Szepesvári 2009 — variance-adaptive bonus;
local canon RL-Intro 199 confirms the count-based bonus form and that c must be tuned):

```text
Maintain per child: N, W, W2 (sum of squared backup values, mover-perspective)
Seed each child with its expansion value as one pseudo-sample:
  n_eff = N + 1;  mean = (sgn*init + sgn-adjusted W)/n_eff;  var from W2 analog
Score(child) = mean + sqrt(2 * var * c * ln(Np) / n_eff) + 3 * c * ln(Np) / n_eff
```

The pseudo-sample seeding replaces UCB's forced first-visit sweep (fatal at chess
branching × low sims) — the expansion batch already measured every child once.

- Guarantee (gate G4): with equal means, the higher-variance child receives more visits.

## :SA-mover: (head2head wiring)

New mover spec: `sa:<params>:<ckpt.pt>` where params is comma-separated k=v:
`sel=puct|ucbv, lb=<float>, leaf=vf|ab1|ab2|ab3, sims=<int>, c=<float>, tp=<float>`.
Evaluator = `tanh(X @ w_raw + b_raw)` over `encoders.get(enc)` features, enc read from
the ckpt (`amap` for the champion) — NOT the `kcz:` path (its encoder is kc-809; the
champion is 897-dim, dimension mismatch, verified 2026-07-16). Root dither semantics
identical to the existing `puct:` spec (repetition-aware visit argmax).

## :Budget-match:

sims is an infrastructure CONTROL, never an Optuna dimension. Calibrated once per
(arm, leaf-depth) so median move time matches the reference mover within ±25% on 5
fixed midgame FENs; cached in `data/search_sims_budget.json`.

## :Screen-instrument:

- Reference mover (fixed, all studies): `enc:amap:models/champion.pt` — the standard
  d2-width-8 beam @ τ=0.02 duel lens over the champion evaluator.
- Objective: H2H Elo diff (A=arm, B=reference) at H2H_CAP=20 explore tier, H2H_SHARDS=2.
- Driver: `experiments/tune_search.py` — TPE, `sqlite:///models/search_optuna.db`,
  one study per arm (fingerprinted space), literature priors enqueued as trial 0
  (c_puct 1.5 Grill 2020; λb 0.5; UCB-V c 1.0 Audibert 2009), wall cap 30 min/study.
- Search spaces: hyb {lb[0,1], c[0.8,3.0], leaf∈{ab1,ab2,ab3}, tp[0.1,0.5]};
  puc {c[0.8,3.0], tp[0.1,0.5]}; ucv {c[0.5,2.0], tp[0.1,0.5]};
  baseline {depth∈{2,3}, width∈{4,8,12}, tau[0.02,0.25]} via :Beam-mover-spec:
  `beam:<tau>:<width>:<depth>:<inner>` (new, parameterizes the existing default policy).
- Cores: 3 lanes × 2 cores (lane.py pairs 0,1 / 2,3 / 4,5); baseline study on the first
  freed pair. Thermal guard inherited.

## :Tournament-and-verdict:

1. Round-robin of the 3 arm-bests, 50g/pairing (H2H_CAP=50) → standings winner.
2. Winner vs baseline-best, 50g. Gate: 95% band excludes zero.
3. Verdicts (either way) → `spec/dispositioned.md` search-arm ledger +
   `data/h2h_verdict_sa_<arm>.md`. Full pooled-ladder claims rung vs SF only on a
   passed gate AND explicit operator go.

## :Lambda-target-training: (added 2026-07-17 — operator: "mcts can help in
## reducing training time")

The λ-tree engine as the TDLeaf TARGET/behavior source inside qlearn.py, testing
whether λ-blended backed targets steepen Elo-per-wall-clock at the screen tier.
- Flags: `QLEARN_MCTS_LAMBDA` (float λb, 0 = OFF = engine untouched),
  `QLEARN_MCTS_SIMS` (default 16 ≈ d2-beam wall parity). Study identity gains a
  `|mcts-lb<λ>-s<sims>|v1` REGIME token — arm studies can never resume controls.
- `puct_choose(board, vf, enc, sims, rng, tau, lambda_backup)` returns the
  TDLeaf 4-tuple `(move, pv_leaf_board, gv, predicted_reply)`:
  - Require: non-terminal board; sims >= 1.
  - Guarantee (:Backed-bootstrap: honored): gv's minimax component maxes over
    EXPANDED children only — unvisited 1-ply inits are excluded (the S&B 015
    poison class); gv = (1-λb)·root_mean + λb·minimax(expanded children M),
    White-absolute.
  - Behavior: τ>0 → softmax over mover-perspective child blended values at
    temperature τ (same units/schedule semantics as the beam); τ≤0 → the
    engine's repetition-aware visit argmax.
  - pv_leaf_board: descend the chosen child by max visits while expanded —
    the PV-leaf analog; predicted_reply: chosen child's max-visit reply or None.
- greedy_move / Elo measurement UNCHANGED (d2 beam) — both arms measured on the
  same ruler; only target/behavior generation differs.
- σ-adaptive λ (dispersion-driven, spec'd in-chat) is the follow-on variant,
  gated on the fixed-λ arm showing wall-clock gains without collapse.

## Acceptance gates (all BEFORE lane launch; ≥3 varied inputs each)

- G1 sign-convention: native searcher vs Python vf agreement on 3 known-sign FENs.
- G2 regression: `lb=0, leaf=vf, sel=puct` move choice identical to pre-change engine
  on 3 fixed FENs (same rng seed, same sims).
- G3 thesis: a trap-position battery where lb=1.0 diverges from lb=0.0 toward the
  refutation-aware choice (mechanism-level M-propagation unit check + ≥1 divergent FEN).
- G4 UCB-V sanity: variance-sensitive visit allocation on a controlled toy.
- G5 smoke: one 4-game duel per arm spec through head2head.py end-to-end.
