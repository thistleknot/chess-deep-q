# Merge 7 — full KnightCap training fidelity (the donor recipe, whole)

Directive: implement their system IN FULL before considering any homegrown machinery.
Source of truth = the donor CODE (scratchpad clone of github.com/tridge/KnightCap: td.c,
knightcap.h, local.h, large_coeffs.h), not our paraphrase of the paper. Our own speculative
mechanisms (λ/τ asymptotic anneals, variance-adaptive λ, buffer-epoch windows, feature
compression from archived specs) are EXCLUDED from this mode — they were stabs in the dark;
the donor recipe is measured (+500 Elo / 300 games).

## :Faithful-mode: `QLEARN_KC_FAITHFUL=1` (implies TDLEAF)

Every clause cites the donor source:

1. **RAMP blunder filter** (td.c:163, RAMP_FACTOR=0 in local.h) — spec/tdleaf.spec.md
   :Ramp-filter:. Favorable TDs on unpredicted opponent moves are zeroed.
2. **Online per-GAME updates, no replay buffer** (td.c td_update called at game end;
   backward-view eligibility over the game's own leaf gradients, td.c:229-237): after each
   game, one plain-SGD step per coefficient from that game's filtered TD sum. No buffer, no
   minibatches, no TRAIN_STEPS, no Adam moments.
3. **λ = 0.7 fixed** (knightcap.h TD_LAMBDA), **γ = 1** (absent from their code) — no
   annealing, no variance adaptation.
4. **tanh calibration** (local.h EVAL_SCALE: one pawn ⇒ V=0.25) — DROPPED WITH NOTE, like
   item 8: rescaling the champion PST ×18 to hit it amplifies per-square weight noise until
   material responses invert (measured: +1 pawn evaluated −0.68 lower). Substitute that
   preserves the donor's intent: donor feature inits are expressed in the champion's own
   learned pawn unit, so the donor's feature-to-material RATIOS hold at the champion's scale.
5. **Greedy behavior** (KnightCap always played its search move): τ pinned at floor;
   exploration comes from opponent variety, not softmax dithering.
6. **Opponents at and above own strength** (their FICS pool): graded ladder + reach games
   (:Opponent-diet:, QLEARN_OPP_REACH=0.25).
7. **Donor-informed init**: our 40 donor features initialized from large_coeffs.h values,
   converted: w_i = (coeff_i / PAWN_VALUE) × atanh(0.25) ÷ feature_normalization_i, sign
   flipped for Black-side features; PST planes keep the champion embed. Their trained values
   are a legitimate init for THEIR features (operator-approved; distinct from engine-label
   distillation).
8. **Draw target** (td.c: tanh(EVAL_SCALE·DRAW_VALUE), DRAW_VALUE=-10) ≈ -0.00026 →
   negligible at our scale; documented as consciously dropped.
9. **Update magnitude**: their TD_ALPHA=10/EVAL_SCALE in coefficient units has no exact
   analog under normalized features; α remains a knob (plain SGD), grounded at the S&B rule
   (3e-4) with a stability probe; translation uncertainty NOTED, not hidden.

OUT OF SCOPE (Merge 8 if the recipe demands it): porting all 577 eval features/game-phase
staging (eval.c is 1888 lines of C); full-width deep alpha-beta target search (their MTD(f)
d4+) — the one recipe line we knowingly run shallower (beam d2), kept LAST because of cost.

## Acceptance

1. Unit: RAMP filter zeroes exactly the favorable-unpredicted residuals on a constructed
   game trace; unfiltered targets equal today's λ-returns bit-for-bit.
2. Donor-init net: 1-pawn-up test positions evaluate ≈ +0.25; hung-queen positions strongly
   negative for the side to move that hung it.
3. Smoke: 8-game faithful run — per-game update path (buffer bypassed), finite deltas.
4. Arm: faithful mode vs the kc incumbent, graded+reach, ≥ 3 epochs; success = confirmed
   crowns trending up / rung climbs; then 60g + 200g rungs.
