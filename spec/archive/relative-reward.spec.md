# Merge 14 — Relative-rating rewards: the terminal signal, re-founded

Operator hypothesis (2026-07-12): the terminal reward is the defect — replace per-game
checkmate z with relative rating over a series ("its own dynamic embedding with a single
scalar, relative to its peers"). Research verdicts (council round #8): the idea is published
practice (ERRL, arXiv 2409.03301 — Elo ratings of trajectories AS rewards; AlphaStar's
league = rating-aware opponent sampling), and its mechanically correct form is **rating
surprise** — Elo's own update term `K·(outcome − expected)`, an advantage whose baseline is
the peer-relative rating embedding.

The operator's key correction, folded in: under rating surprise **a draw is no longer zero
reward** — drawing a stronger opponent pays positive surprise, drawing a weaker one negative
— so the reward change itself manufactures learning signal from non-terminating games. The
only regime it cannot help is pure MIRROR self-play (symmetry ⇒ expected 0.5 ⇒ draw = zero
surprise under any scheme), which is why the primary arm pairs the new reward with the
GRADED LADDER, and mate-dense starts serve only the mirror-play share.

## :Rating-surprise: (mechanics, both arms)

- The trivium's outcome term uses `z_rel = z − E[z | ratings]` instead of raw z.
- `E[z | ratings]` = Bradley-Terry expectation `1/(1+10^((R_opp − R_agent)/400))` mapped to
  the z scale; rung ratings from the ladder, **SF@1320 as the fixed anchor** in every pool.
- Mirror self-play batches (no opponent rating variance): group-relative fallback
  `z_rel = z − mean(z over the batch)` — GRPO-lite; batch is the baseline.
- Agent rating updated online per game (Elo K-update) from the same anchored pool.

## Guards (from the refutation searches — these are requirements, not advice)

1. :Anchor-pinned: the SF@1320 anchor participates in every rating pool (closed self-play
   pools drift/inflate; the anchor pins the gauge).
2. :Diverse-pool: matchmaking preserves opponent diversity — never a single-opponent pool
   (Elo is who-plays-who dependent under intransitivity).
3. :No-self-reference: the agent's rating never updates from games where both sides' ratings
   are its own (mirror games use the group-relative fallback, not rating updates).
4. GRPO saturation caveat stands: if all outcomes in a batch are identical, z_rel ≡ 0 —
   logged per batch (`surprise_var` metric) so saturation is visible, never silent.

## Arm B (PRIMARY) — :Rating-surprise: + graded ladder

- Canon recipe, OPP=graded (rungs random → heuristic → SF skills → SF@1320), outcome term =
  z_rel. Single variable vs the raw-z graded control.
- Pre-registered: (a) |surprise| > 0.1 in >50% of generation games from epoch 1 (the
  operator's "probabilities naturally increase" claim, made testable); (b) at matched games,
  surprise-arm learning curve ≥ raw-z graded control; falsified ⇒ revert, trivium keeps raw z.

## Arm A (complementary) — :Density-curriculum: for the mirror share

- Exploring starts from mate-dense/reduced positions for an annealed fraction of self-play
  generation games. :Provenance: the position generator must derive starts by RULES from
  random/own play (piece-removal from reached positions) — no curated human/tablebase data.
- Port note: `QLEARN_CURRICULUM` is wired to the AC path; the q-path port needs a start-
  position hook in qlearn generation (native path: a start-fen parameter in `play_games`).
- Pre-registered: generation ply-cap draw rate <50% (from ~97%); outcome variance non-zero;
  pure-seed first-5-epoch trajectory ≥ the self-whitened baseline (crown 8.83).

## Sequencing

Merge 13 (capacity, council round #7) owns the CEILING; this merge owns the SIGNAL. They
compose later (capacity net under surprise rewards) but never stack in one arm. Operator
picks the run order; all runs operator-started.

## Acceptance

1. Smoke (Arm B): surprise values zero-mean across a matched batch; `surprise_var` metric
   emitted; anchor present in the pool object.
2. Smoke (Arm A): 200-game run shows draw-rate reduction + finite outcome variance.
3. Both arms' falsification lines pre-registered here BEFORE launch; ledger + findings.md
   updated with whichever verdict lands (F14 candidate: density/surprise as rate factors).
