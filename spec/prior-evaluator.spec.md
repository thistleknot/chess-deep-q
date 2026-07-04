---
description: 'The fixed heuristic prior, and the prior lineage it starts: heuristic -> distilled teacher -> learned net'
import:
  - elo-measurement
  - annealing-schedule
---

***definitions***

- :Prior-evaluator: is the fixed, hand-authored scoring of a chess position from material, mobility, king safety, pawn structure, space and coordination; it is the bootstrap knowledge the learner starts from and must eventually surpass.
- :White-absolute-frame: is the sign convention where a positive score favors White regardless of side to move; it is the single frame shared by prior, learned value, and reward.
- :Move-category-weights: is the prior's search bias — a mapping from tactical move classes (captures, checks, threats, development) to sampling weights that decide which moves a search even considers.
- :Prior-lineage: is the ordered succession of priors — hand heuristic, then distilled teacher-net snapshot, then current learned net — where each member bootstraps the next, each handoff is annealed and gated by :Measured-elo:, and every earlier member remains a fallback for the one after it. The teacher snapshot *supersedes* the heuristic as the operative prior once it measures stronger against the :Elo-anchor:; the heuristic is never deleted.

***implementation reqs***

- The :Prior-evaluator: lives in `evaluation.py` and is never trained; no gradient ever updates it.
- Constant: PIECE_VALUES and the per-feature evaluation weights — domain rules of the prior, developer-tuned.

***test reqs***

- A position winning for White and its mirror winning for Black, to assert frame symmetry.

***functional specs***

- The :Prior-evaluator: must score every non-terminal position in the :White-absolute-frame:.
  - Given a position better for White, Then its score should be positive; Given the color-mirrored position, Then its score should be negative of the first.
- The :Prior-evaluator: must stay fixed across all of training.
  - Given any training progress, When the evaluator is queried, Then it returns the same score for the same position (no learned drift). (Maintain: prior is a constant reference, not a moving target.)
- The prior's influence must be annealable by callers, not by mutating the prior itself.
  - Search leaves and shaped reward should scale prior contribution via the :Annealing-schedule:; the prior remains whole and callers down-weight it.
- :Move-category-weights: must be softenable before sampling so the prior's search bias can decay with progress.
  - Given rising :Prior-bias-temperature:, When categories are sampled, Then the effective weights should flatten toward uniform, widening the move set the learner may discover. (This is the guard against prompt.md's "prior lock-in".)
- The :Prior-lineage: handoff must be measured, not assumed.
  - Given the distilled teacher snapshot measures stronger than the heuristic against the :Elo-anchor:, Then the teacher snapshot becomes the operative prior for leaf blending and shaped reward.
- Lineage fallback must step one member back, never to random.
  - Given net inference fails at any lineage level, When a value is needed, Then evaluation falls back one lineage step (learned → teacher → heuristic); the heuristic is the terminal fallback.
