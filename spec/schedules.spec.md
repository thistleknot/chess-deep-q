# :Schedules: — the dial panel (LR + horizon/mix scheduling), Merge 18

Operator direction: handle learning rate alongside discount/horizon scheduling; **no
complexity for its own sake** — if the standard is linear-in-timestep, so be it; a dial
keyed on N signals is a *linear model over hand-crafted schedule features*
`dial_k = base·(1 + w·φ_k)`, so build ONE mechanism and let multi-dimensionality arrive
only by composing bake-off SURVIVORS.

## The panel (state after this merge)

| Dial | Schedule | Reacts to |
|---|---|---|
| τ (behavior temp) | anneal over `WARMUP` + `STALE_REHEAT` ×(1+k·stale) | progress + crown staleness |
| λ (return horizon) | anneal over `LAMBDA_WARMUP` + `adapt_lambda()` (ON by default) | progress + prev-epoch TD-error sdev |
| trivium a/b/c mix | anneal over `TRIV_WARMUP` **+ :Phase-mix: (per-position)** | progress + game phase |
| γ (discount) | constant (Optuna `decay`) **+ :Dis-gamma: (per-position)** | search↔net disagreement |
| α (LR) | **:LR-warmup: linear ramp** | timestep only (Adam covers reactivity) |

## Signal audit (the five candidates, dispositioned)

- **loss variance → λ**: ALREADY EXISTS (`ADAPTIVE_LAMBDA`, :Variance-adaptive-λ:) — not duplicated.
- **ply / unmoved pieces → phase**: accepted as ONE phase axis; implemented as TOTAL
  material on board (sum of both colors, kings excluded — derivable from the encoding
  already in hand at target-build time; unmoved-pieces would need new plumbing for the
  same axis). Neutral by construction: does not leak who is winning.
- **signed material advantage**: REJECTED as a schedule key — it is the value function's
  own dominant feature; keying the horizon on it truncates credit exactly on
  sacrifice/compensation lines (the depth-amplifies-holes wall would deepen).
- **ML loss level**: REJECTED (direction ambiguity — lengthen or shorten on plateau is
  a guess; the variance form already covers the well-posed half).

## Organs (each: one env knob, default 0 = OFF; bakeoff.spec.md protocol, 3-trial study)

### :LR-warmup: — `QLEARN_LR_WARMUP` (fraction of training; 0 = constant α)
- lr(t) = α · clamp(frac/LR_WARMUP, 0.05, 1.0), frac = (cum+played)/total — linear ramp
  with a declared 5% floor (lr=0 on chunk 1 would waste it), then hold at α.
- Hook: `opt.param_groups[*]["lr"]` once per chunk (SGD-faithful and Adam paths alike).
- Rationale: early targets are noise (untrained V, short λ); full-size steps imprint it.
- Optuna dim `lr_warmup[0.0,0.5]` (`QLEARN_LRWU_TUNE=1`).

### :Phase-mix: — `QLEARN_PHASE_MIX` (mix modulation gain; 0 = flat mix)
- p_k = min(1, total_material(x_k)/78) ∈ [0,1] (1 = full board/opening).
- c_k = c·(1 + PM·(p_k − 0.5)) clipped to [0, b+c]; b_k = (a+b+c) − a − c_k (b absorbs
  the complement; mix mass preserved). Opening leans OUTCOME (the "γ>1 early" intent as
  safe target reweighting — no contraction risk), endgame leans SEARCH.
- Require: raw piece planes at x[:768] — ENC ∈ {pst, kc, kx} and NO ZCA wrap; violating
  configs fail fast at startup (never a silent no-op).
- Optuna dim `phase_mix[0.0,1.0]` (`QLEARN_PHASE_TUNE=1`).

### :Dis-gamma: — `QLEARN_DIS_GAMMA` (γ modulation gain; 0 = constant γ)
- d_k = |gv_k − V_net(x_k)| (search value vs the net's own static value — provenance-pure).
- γ_k = γ·(1 − DG·min(d_k, 1)) used per-step in the λ-return recursion (both the TDLeaf
  ramp path and the plain path). γ_k ∈ (0, γ] always — state-dependent discounting
  (generalized-MDP / GVF termination form, White 2017): hot positions shorten credit,
  quiet positions keep it long. Quiescence logic promoted into the target schedule.
- DDQN composition: disagreement = |live V − frozen search value| (the same two
  quantities DDQN already computes).
- Cost: one batched no-grad forward per game when ON.
- Optuna dim `dis_gamma[0.0,0.8]` (`QLEARN_DISG_TUNE=1`).

## Horizon accounting (the reasoning frame — operator-derived, = truncated λ-returns)

Contribution of a reward n plies out is (γλ)^n ⇒ effective credit horizon
n_eff = ln(ε)/ln(γλ). γ is horizon-relative (γ ≈ 1 − 1/n_expected), NEVER ≥ 1 (episodic
would allow γ=1; we keep γ<1 as a variance choice). Proven parms γ≈0.977, λ≈0.77 ⇒
γλ≈0.75 ⇒ n_eff(1%) ≈ 16 plies — the tuner's own answer to "how far should credit see,"
found inside a search space spanning ~6–1000 plies. DECIDED: trust the tuned horizon; no
range changes. :Dis-gamma: is per-position n_eff control — hot positions pull the horizon
in, quiet ones leave it long.

## Contracts

- **Require**: flags default 0; :Phase-mix: requires raw planes (fail-fast guard).
- **Guarantee**: all flags at 0 ⇒ target math is IDENTICAL to pre-merge (the guarded
  branches are the only additions; flags-off smoke must reproduce baseline losses).
- **Maintain**: a+b_k+c_k = a+b+c per position; γ_k ∈ (0, γ]; provenance purity (every
  schedule input is our own play/net — no external signal).
- **Assert**: smoke prints per-chunk `sched | lr … | mean_p … | mean_gk …` when any
  flag is ON.

## Composition rule

Composition = enabling multiple surviving features in the same linear form — a CONFIG,
never new code, and only after each survives its solo study (±100 tie rule applies).

## Parked (documented, not built)

meta-gradient γ/λ (Xu/van Hasselt/Silver 2018 — principled, heavy; revisit if the
single-feature organs survive) · ply-oscillating γ (no evidence base) · loss-LEVEL-reactive
λ (see signal audit).

## Gate

NO bake3 study launches until the capacity verdict (B0 vs mlpb, termination clause)
lands and survives, AND the operator says go — new studies are a user decision.
