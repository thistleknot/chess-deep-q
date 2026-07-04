"""Concatenate the spec set into one copy-pasteable file for review/handoff.

Emits spec_bundle.md: the README index, then the single entry point (chess-rl), then every
spec in dependency (reading) order, each under a clear header with its source path. Order follows
the import DAG so a reader meets each concept before it is used.

Usage: python bundle_specs.py [output_path]
"""

import os
import sys

SPEC_DIR = "spec"

REVIEWER_CONTEXT = """## For reviewers — live decisions & open questions

**Where the implementation actually is (measured, honest):**
- Learned net currently ~920 Elo vs a real Stockfish anchor (SF 18; UCI_Elo floor 1320). It is
  *undertrained*, not broken — see blocker 1.
- The repo's pure-Python alpha-beta engine (hand eval): n=30 with CIs, pst measures ~1428 at a real
  0.3s/move [CI 1306-1584] and ~1672 at fixed-depth-3 (unbounded time). The old "4-0 -> ~1720" was
  n=4 noise. It is the baseline + a candidate teacher, not the learned model.
- Only Stage 1 (Stockfish distillation) has run. Stage 2/3 (λ-return refinement, self-play) are
  authored in spec but gated behind an unmet 1200-Elo gate and have not executed.

**Goal (ambition):** frontier strength, **2600+**, honestly measured; plus an **Elo↔temperature
proxy** so ONE strong model can be dialed to any point on the human rating curve (see
dynamic-difficulty + elo-measurement).

**Open questions where SME input is most valuable:**
1. **Training throughput.** The residual tower runs ~1 step/s at batch 512 and ~12 s/step at batch
   2048 on a Max-Q laptop GPU (superlinear — thermal/throttle). A 200 s train phase buys tens of
   gradient steps, not thousands. Is a residual tower the wrong net for this hardware? CPU training
   (batched CPU hit ~18k samples/s for the tiny net), a smaller net, or gradient accumulation?
2. **Search ceiling. ANSWERED (n=30, this iteration): search depth reached within the time budget
   dominates.** At a fixed 0.3s/move the evals rank strictly by per-node SPEED, not accuracy:
   pst 1428 >> linear 1200 > gbdt 735 >> hybrid(pst+residual) -280 (0/30) — anything that slows the
   per-node eval collapses search depth and strength. The learned leaf value is a dead end here; the
   fast smooth hand eval searched deeper wins. Using the net as a move-ordering PRIOR orders ~8%
   fewer nodes at ply<=1 but costs ~9.6ms/call (112% of a 0.3s budget) -> not repaid at time control
   until a cheaper policy exists. So: *search*, not the net, is the binding constraint on this
   hardware; the net earns its place only as a cheap ordering/window signal, not a leaf eval.
3. **Teacher & data.** To exceed 1720 toward 2600+, Stockfish must be the teacher (the 1720 engine
   caps too low). Distill SF eval + best move (dense, low-variance signal) vs learn from SF
   self-play games (in-distribution, outcome-grounded) vs both? Teacher depth vs data volume under
   the throughput constraint?
4. **RL-rule soundness.** Value = search-bootstrapped λ-return (tree-backup, off-policy-safe);
   policy = MCTS visit-count distillation (expert iteration). Is the classification and the
   off-policy-safety argument sound? See `rl-categorization.spec.md` and `value-target.spec.md`.
"""

# Reading order: index, entry point, then the numbered dependency order from README.
ORDER = [
    "README.md",
    "chess-rl.spec.md",          # single entry point (root of the import DAG)
    "entrypoint.spec.md",        # the top-level main.py contract (Play/Train/Measure/Difficulty)
    "elo-measurement.spec.md",
    "annealing-schedule.spec.md",
    "prior-evaluator.spec.md",
    "learned-model.spec.md",
    "teacher-distillation.spec.md",
    "nnue-eval.spec.md",
    "search-mcts.spec.md",
    "value-target.spec.md",
    "linear-value-rl.spec.md",
    "self-play-leela.spec.md",
    "training-loop.spec.md",
    "dynamic-difficulty.spec.md",
    "elo-calibration.spec.md",
    "rl-categorization.spec.md",
]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "spec_bundle.md"

    # Include any spec files not explicitly ordered (so nothing is silently dropped).
    present = [f for f in os.listdir(SPEC_DIR) if f.endswith(".md")]
    ordered = [f for f in ORDER if f in present]
    extras = sorted(f for f in present if f not in ORDER)
    if extras:
        ordered += extras

    parts = []
    parts.append("# chess-deep-q — full spec bundle\n")
    parts.append("Spec-driven development: this is the authoritative source of truth; code traces "
                 "to it. Single entry point is `chess-rl.spec.md` (its imports transitively close "
                 "the whole set). Files below are in dependency reading order.\n")
    parts.append(REVIEWER_CONTEXT)
    parts.append("## Contents\n")
    for i, f in enumerate(ordered, 1):
        parts.append(f"{i}. `{f}`")
    parts.append("")

    for f in ordered:
        path = os.path.join(SPEC_DIR, f)
        with open(path, encoding="utf-8") as fh:
            body = fh.read().rstrip()
        parts.append("\n\n" + "=" * 80)
        parts.append(f"FILE: {path}")
        parts.append("=" * 80 + "\n")
        parts.append(body)

    text = "\n".join(parts) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    lines = text.count("\n") + 1
    print(f"Wrote {out_path}: {len(ordered)} specs, {len(text)} chars, {lines} lines.")
    if extras:
        print(f"(Included unordered extras at end: {', '.join(extras)})")


if __name__ == "__main__":
    main()
