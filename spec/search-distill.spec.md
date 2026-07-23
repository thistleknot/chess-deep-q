# Search-bootstrapped self-distillation (:Search-teacher:)

**Status:** ACTIVE ARM (pre-registered 2026-07-21, before any run)

## Lesson under test

The bitter-lesson-consistent move at this compute scale is not tabula-rasa RL (data-starved)
and not more hand-derived features (dispositioned plateau): it is **expert iteration** —
use the existing native alpha-beta (`rsearch4`) as the *teacher*, labeling self-play
positions with deep backed values, and fit the eval offline against those labels.
Own-search labels are purity-compliant per spec/pathfind-population.spec.md
("no engine-label distillation ever (own-search self-distillation is compliant)");
the existing `experiments/train_nnue.py` Stockfish-label path is the non-compliant
teacher route and is NOT used here.

Rejected mechanisms (pre-registered reasoning, not outcomes):
- Online TDLeaf continuation as the vehicle for representation change: self-reinforcing
  targets; KnightCap required diverse opponents to progress under TD. Offline
  distillation decouples generation from fitting.
- Full halfKP (~10M features): needs orders of magnitude more labeled positions than
  this project can generate. The repo's HalfKP-lite (2560, `chessdq/nnue_model.py`)
  is the correctly-scaled variant.

## Phases and gates

- **Phase A (:Distill-control:)** — refit the SAME amap-897 linear eval against its own
  depth-d search labels (`experiments/distill_linear.py`). Drop-in native ckpt
  (`enc=amap, arch=linear`). Gate: H2H 50g vs `models/champion.pt` at the standard
  d2 duel lens; Wilson band excludes zero. Pre-registered risk: TDLeaf may already
  sit near this fixed point — a null here is a cheap, legitimate verdict and caps
  Phase B expectations.
- **Phase B** — HalfKP-lite NNUE (existing arch, 2560→128→32→1) trained on the SAME
  own-search label set (swap label source in `train_nnue.py`, SF path untouched).
  Gate: beats the Phase-A linear at equal search budget by ≥ +50 Elo (pre-registered
  threshold; below that, native Rust port is not worth the engineering).
- **Phase C** — native `NnueSearcher` in `rsearch/src/lib.rs`, mirroring
  `IncrementalNNUE` (reference impl + `experiments/test_accumulator.py` battery).
  Only after B clears its gate; full-scale ladder runs surfaced to operator first.

## Protocol constants

- Position source: `rsearch4.play_games` self-play from champion weights, ε/τ
  exploration for diversity; PV-leaf FENs are the regression targets (quiescence-resolved).
- Labels: `Searcher.search(fen, d)` White-tanh value; fit in atanh space (native eval
  is linear pre-tanh). Label depth chosen by a bounded timing probe so a full labeling
  pass stays ≤ ~30 min on 6 threads.
- Every phase verdict (win or park) → spec/dispositioned.md + data/ note, per repo law.

## Falsification

The arm is parked if Phase A is null AND Phase A residuals show no structure a
nonlinear eval could exploit (residual-vs-feature analysis flat), or if Phase B
fails its +50 gate.

## Phase A VERDICT — CONFIRMED (screen-grade), 2026-07-21

CONFIRMED on two independent self-controlled instruments (data/h2h_verdict_distillA.md):
head2head canonical d2-beam ruler **+798 (1.000/50g, champ-vs-champ control 0.517)**;
fair native deterministic diverse-opening duel at deployment depth **d6 +176 (0.733, band
excludes 0.5)**, d4 +35 (ns) — native advantage GROWS with depth. Pre-registered fixed-point
risk falsified. Artifact: models/champion_distillA2.pt.

Two instrument traps caught (both would have flipped the verdict):
- :Saturation-cut: — raw ridge on ±1-saturated self-play labels blows the eval scale 16x
  (native-unplayable). Drop |label|>0.90 + ridge~100 → champion-scale, corr +0.39.
- :No-native-duel-ruler: — `rsearch4.play_games` applies exploration to the AGENT side only
  (training generator, not a fair duel); champ-vs-itself scored 0.000. Fair native eval
  comparison requires deterministic play from diverse openings, role-symmetric.

## FINAL VERDICT (2026-07-22) — CONFIRMED WIN, +~240 Elo over champion

Native-d9 head-to-head (the true ruler; the vs-SF ladder draw-floods strong agents and
understated this to a false "tie") — instrument control champ-vs-champ = 0.490 (band incl 0.5):
distillA2 **+232**, distillA3 **+221**, distillA4 **+267** vs the 1878 champion, all bands exclude
0.5. ROBUST across iterates; the FIRST distillation step captures the gain (A2≈A3≈A4, iteration
converges). SWA weight-average dilutes (rejected). Champion candidate = **distillA2**.

Mechanism (answers the bitter-lesson question): the teacher is eval + d5 minimax search, which
strictly exceeds the static eval; distilling it back makes the static eval ~one lookahead stronger,
and at d9 deployment that converts to strength. Search generates the signal; the eval learns it.

Phase B NNUE (capacity, same labels): trained (sign_acc 0.97) but inherits the teacher's
compression and is bounded by the same teacher ceiling; not measured to native deployment (needs a
Rust port) — not pursued because more capacity cannot exceed the teacher it imitates.

### Promotion (operator-run; reversible)
    UTC=$(date -u +%Y%m%dT%H%M%SZ)
    cp -L models/champion.pt "models/champion_backup_${UTC}_predistill.pt"   # backup current
    cp models/champion_distillA2.pt "$(readlink -f models/champion.pt)"       # promote
    python -c "import chessdq.agents as a; print(a.make_agent('champion')[0])" # verify loads
    # REVERT: cp "models/champion_backup_${UTC}_predistill.pt" "$(readlink -f models/champion.pt)"
