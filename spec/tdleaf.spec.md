# Merge 5 — TDLeaf(λ): learn on the search's principal-variation leaves

Grounding: [TDLeaf(λ), Baxter/Tridgell/Weaver 1999](https://arxiv.org/abs/cs/9901001) +
[KnightCap](https://arxiv.org/abs/cs/9901002) (1650→2150 Elo in 308 games);
[Giraffe, Lai 2015](https://arxiv.org/abs/1509.01549) (same algorithm, single machine, ~IM).
Diagnosis it fixes: Merge 4's search adds ~+50–100 Elo at inference but the weights never see
search-backed values — vanilla TD(λ) trains V toward 1-ply-greedy λ-returns at raw afterstates.
TDLeaf substitutes, for every visited position, the **minimax value and its PV leaf**: the TD
error is computed between successive *search* values and the gradient is taken at the *leaf*
whose static eval produced them. One change to the target/state pair; the trace machinery is
untouched.

## :PV-leaf: (search_policy.py)

- `move_values(board, value_fn, width, qext, want_leaves=False)` — when `want_leaves`, ALSO
  return `leaves`: per root move, the 769-dim encoding of the position whose value became
  `vals[i]`.
  - Unexpanded / terminal root move → the root afterstate encoding itself.
  - Expanded root → the reply (or qext recapture) position whose value survived the
    min/max propagation, including standing-pat: if static beats every recapture, the leaf is
    the reply position, not the recapture.
- `move_values_d3(..., want_leaves=False)` — same contract; a beamed move's leaf is the leaf
  of the depth-2 reply that survived the opponent's min/max at the reply node.
- `search_move(..., return_leaf=False)` — when `return_leaf`, return
  `(move, leaf_of_chosen_move, greedy_search_value)`.
- :Backed-bootstrap: (SPEC REPAIR 2026-07-10, S&B skill 015 maximization-bias): with
  `want_leaves`, `move_values`/`move_values_d3` ALSO return `backed[i]` — True iff `vals[i]`
  is exact (terminal) or depth-backed (expanded/beamed with replies). `greedy_search_value`
  = max σ·vals over **backed moves only** (global argmax fallback iff none backed).
  Root cause of the measured universal training decline: width-8 of ~30 moves leaves ~75% of
  root values as optimistic 1-ply fallbacks; max over them systematically inflates the TDLeaf
  bootstrap → V trained toward inflation → runaway optimism → every arm (PST/kc, fresh/seeded,
  self/graded) degraded monotonically. Behavior/move choice is UNCHANGED — only the bootstrap.
- Guarantee: with `want_leaves`, `value_fn(leaf[i]) == vals[i]` for every non-terminal-
  propagated move (terminal-propagated leaves carry the terminal position; its value is exact
  and its gradient contribution is legitimately ~0 through tanh saturation).

## :TDLeaf-mode: (qlearn.py)

- New knob `QLEARN_TDLEAF` (default 0). When 1:
  - Generation plays with the SEARCH policy (depth `QLEARN_SEARCH_DEPTH`, default 2, width
    `QLEARN_SEARCH_WIDTH`) regardless of `QLEARN_BEHAVIOR` — TDLeaf is meaningless without
    search values.
  - `choose()` returns `(move, PV-leaf encoding, greedy SEARCH value)`: `xs[k]` stored for
    training is the **PV leaf**, `gvs[k]` the **minimax root value** — both replacing their
    1-ply counterparts. `build_targets` is unchanged (same λ-return over rewards + bootstrap),
    which IS the forward-view TDLeaf update.
  - Everything else — buffer, batching, anneals, adaptive λ, anchor ratchet, freeze-epoch,
    patience, Elo objective — unchanged.
- `greedy_move` under TDLEAF measures with the same search depth/width (consistent with
  BEHAVIOR=search semantics).

## :Console: (server.py)

- `TrainReq.tdleaf: bool = False` → env `QLEARN_TDLEAF`; `search_depth: int = 2` →
  `QLEARN_SEARCH_DEPTH`; checkbox + field in the form.
- Ladder panel: rows lacking finite `elo_lo`/`elo_hi` are excluded from the BEST-rung tile
  (fixes the spurious degenerate "1129 (1129..1129)").

## :Tuning: (tune_qlearn.py) — TDLeaf manifold re-tune

Measured (validation run, 2026-07-09): at the OLD manifold's tuned params (α=0.0084, λ=0.53,
γ=0.978) TDLeaf self-play DAMAGES the champion net within ~200 games (best-visited epoch shut
out 0-60 vs SF@1320 while the untouched champion holds ~880–900 at the same depth). Formula
change ⇒ Optuna reset+rerun (standing rule); the space itself is re-grounded:

- `QLEARN_TDLEAF=1` in the tuner's env selects the TDLeaf regime — part of the study identity
  (REGIME gains `tdleaf-leafgen|resume-champ`, PROTO gains `-tdleaf-d{depth}w{width}`; new
  fingerprint = new study, old studies untouched).
- α range re-grounded DOWN to [1e-4, 3e-3] log (leaf targets are minimax-selected → larger
  magnitude/variance than 1-ply targets; the mlp precedent applies). Other ranges unchanged.
- Trials are seeded FROM the champion (KnightCap/Giraffe: TDLeaf needs a sane initial eval;
  fresh-net trials would sit flat at the floor = the S1 flat-objective failure mode): each
  trial copies a bar-stripped champion seed (`models/qlearn_tdleaf_seed.pt`, no `strength`
  key) to the trial's ckpt + `_best` and runs `QLEARN_RESUME=1`. Identical start ⇒ hermetic.
- Infrastructure controls passed fixed as ever: sample 200, batch 20, ≤3 epochs, patience 2,
  elo 20, `QLEARN_PROXY_GAMES=6`, `QLEARN_DEV=cpu`, depth 2 / width 8 generation.

## :Ramp-filter: (donor transplant from KnightCap td.c — the blunder filter)

Source: `td.c` lines 158–207 + `RAMP(x) ((x)<0?(x):RAMP_FACTOR*(x))`, `RAMP_FACTOR 0`
(their live local.h): temporal differences on UNPREDICTED opponent moves keep only the
unfavorable part — favorable surprises (opponent blunders) are zeroed. Rationale (their
paper + measured here): learning from opponent mistakes teaches the eval to EXPECT mistakes;
on a weak-rung ladder the flood of unearned positive TDs caps learning at the opponents'
blunder level — the "parity but no climb" mechanism.

- `QLEARN_RAMP` (default 0; 1 = filter on). TDLEAF mode only.
- :PV-leaf: extension: with `want_leaves`, move_values/d3 ALSO return `preds[i]` — the
  opponent reply that survived the min/max for move i (None for unexpanded/terminal);
  `search_move(return_leaf=True)` returns `(move, leaf, gv, pred_reply)`.
- `play_game` records, per agent decision t>0, whether the opponent's ACTUAL move since the
  agent's previous decision equals the previous step's `pred_reply` → `predicted[t]`.
- `build_targets` (RAMP on): forward-view λ-return over FILTERED residuals —
  `δ_t = γ·boot_{t+1} − gv_t` (terminal: `γ·z − gv_T`); if NOT predicted[t+1] and δ_t is
  favorable to the agent (sign-adjusted by agent color), δ_t := 0; targets
  `G_t = gv_t + Σ_k (γλ)^{k−t} δ_k`. Unfiltered, this is algebraically the same λ-return as
  today. Self-play: filter applies to both sides symmetrically (each side's xs are its own).
- Single-variable arm vs the kc-lineage incumbent; graded ladder unchanged, reach=0.

Same defect class as :Backed-bootstrap: (S&B skill 015), in the OUTER loop: epoch strength is
a ~24-game sample (σ ≈ ±5 strength ≈ ±150 Elo); `best_strength = max(samples)` crowns lucky
rolls (measured: crowned 12.08/15.58 snapshots re-measure at 862/884 ≈ parity). The bar then
drifts on luck, honest epochs revert against phantoms, and KnightCap-scale slow climbs
(~1 Elo/game) are undetectable.

- `QLEARN_CONFIRM` (default 1; 0 = legacy). Infrastructure control, never tuned.
- When an epoch's `strength > best_strength` and SF is live: play `EPOCH_ELO_GAMES` more SF
  games (`greedy_elo`) + one `evaluate_greedy(PROXY_GAMES)` pass; confirmed strength =
  `100·(ep_pts+conf_pts)/(ep_n+conf_n) + mean(proxy_epoch, proxy_conf)`; crown and store the
  CONFIRMED value in `_best.pt` only if it still beats the bar, else the epoch counts stale
  (NO revert — a high measure is not a collapse). Confirmation games also join the run pool
  (Optuna objective resolution).
- Cost: paid only on candidate bests. Guarantee: resume bars inherit confirmed values.

## :Trivium-anneal: (operator-proposed) — scheduled compound-target weights

Measured: static trivium ⅓/⅓/⅓ ignites fast, fades (E2); static 0.6/0.3/0.1 extends a
trained line (+30%, lane B). Hypothesis: the right OUTCOME weight is time-varying — high
early (unbiased MC anchor while V is weak), low late (search precision once V knows things).
- `QLEARN_TRIVIUM` = start triple "a,b,c"; `QLEARN_TRIVIUM_END` = end triple;
  `QLEARN_TRIVIUM_WARMUP` = e-folding fraction (reuses the shared :anneal: shape). Weights
  interpolate per training progress; END empty = static (back-compatible).
- Tuning (`QLEARN_TRIV_TUNE=1` in the tuner): dims c_start [0.1,0.6], c_end [0.0,0.2],
  b (search weight, time-fixed) [0.1,0.5], triv_warmup [0.1,0.8]; a = 1−b−c(t).
  New REGIME suffix `|triv-anneal|v1`. Never hand-picked — Optuna per the standing law.

## :Graded-opponents: (qlearn.py) — KnightCap's actual headline

[cs/9901002](https://arxiv.org/abs/cs/9901002): the 1650→2150 climb came from ONLINE play vs
graded opponents (FICS matchmaking ≈ always near 50% score); their self-play TDLeaf variant was
substantially weaker. Generation therefore supports an opponent ladder:

- `QLEARN_OPP` = `self` (default, unchanged) | `graded`.
- Ladder, weak→strong: `random` → `heuristic` (1-ply PST) → SF `Skill Level` 0 → 2 → 5 → 10 →
  SF `UCI_Elo` 1320. One persistent SF instance, reconfigured on rung change; SF unavailable ⇒
  ladder truncates to the first two rungs (never crashes generation).
- Matchmaking (`:Matchmaking:`): per-game score (1/0.5/0) into a sliding window of the last 20
  games ON the current rung; window full and mean > 0.6 ⇒ move up one rung (reset window),
  mean < 0.4 ⇒ move down. Start rung: `heuristic`. Rung changes are logged and the rung index
  is emitted in the metrics row (`opp_rung`).
- `play_game` (graded): the agent plays ONE side (color alternates per game); `choose()` runs
  only on agent turns, so xs/gvs hold only agent decisions — `build_targets` is unchanged (its
  step is then one full move, γ applies per agent decision; White-absolute values/z unchanged).
- Composes with TDLEAF (leaf states + minimax targets) and with plain 1-ply targets; orthogonal
  knob. Part of the Optuna study identity when tuning (PROTO gains `-opp{ladder}`).

## Acceptance

1. `py_compile` clean on all three files.
2. Tactical position: chosen move's PV leaf differs from its root afterstate, and
   `value_fn(leaf) == vals[chosen]` (float32 tolerance).
3. 1-epoch CPU smoke (epoch_games=8, d2 w8): finite loss, checkpoint written, metrics rows.
4. Ladder: post-training 60-game rung appended to data/rl_trend.jsonl and visible in the
   console panel; success = raw greedy ≥1000 on a later 200-game run.
