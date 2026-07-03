---
description: 'Stage 1 — supervised distillation of a Stockfish teacher into the dual-head net, under a 5-minute cumulative run contract'
import:
  - elo-measurement
  - annealing-schedule
  - prior-evaluator
  - learned-model
---

***definitions***

- :Teacher: is Stockfish analysing at a fixed shallow depth — depth-8 measures ≈2200+ strength at ≈180 labels/s per engine process. The repo's Python alpha-beta (`engine.py`, ~1720 measured) is explicitly NOT a labelling teacher (3–8 labels/s), though it remains the strength milestone a distilled-and-searched net should approach.
- :Distillation-label: is one supervised example: (FEN, value target per the :Value-target-convention:, policy target = softmax over the MultiPV candidates' centipawns at temperature POLICY_TARGET_TEMP with zero mass elsewhere).
- :Cumulative-dataset: is the append-only labelled set on disk (`data/distill_sf.jsonl`; ~29k depth-8 rows already exist), deduplicated by (FEN, depth), from which every run trains — each ≤5-minute run climbs from the accumulated total rather than starting over.
- :Process-separated-labeling: is the requirement that labelling engines run in separate OS processes (multiprocessing workers, or distinct label-then-train phases) — never as threads sharing the trainer's Python GIL.
- :Run-contract: is the unit of operation: one ≤5-minute run = append labels + train + measure :Measured-elo: + append the (dataset-size, Elo) point to a persistent trend log.
- :Position-source: is where the positions to be labelled come from: **strong-game trajectories** — self-play games by the :Teacher: begun from a few random opening plies (for diversity), from which positions are sampled — NOT uniform random-playout positions. Random-playout positions are largely off-distribution (positions no strong player reaches), so labelling them wastes capacity; in-distribution positions make the distilled value and policy targets relevant to real play, and give the learner a better off-policy dataset to bootstrap from.
- :Trajectory-sampling: is how a :Position-source: game advances at each move: a temperature-softmax sample over the :Teacher:'s MultiPV top-K candidates (scored by centipawns), rather than always its single best move. The MultiPV analyse is already computed for the policy target, so sampling adds NO search cost and does not deepen the teacher (respects LABELS_PER_SEC_FLOOR). Temperature is annealed high→low across the game (more diverse in the opening, near-best later) so trajectories stay near the strong-play manifold — hot sampling throughout would drift positions off-distribution and forfeit the in-distribution benefit.
- :Search-visited-positions: is a required slice of the :Position-source: drawn from the (often non-quiet) distribution the SEARCH will actually query at inference — sampled interior/leaf nodes of the teacher's search trees, or trajectory positions perturbed by 1–2 plies of sampled captures — not game-trajectory positions alone. Rationale (the q8 blowup): a learned eval's error is DISTRIBUTION-DEPENDENT — the net collapsed on off-distribution capture-resolved positions (−1300 Elo) while the hand heuristic degraded gracefully (−400), because material terms still work off-distribution but learned features do not. Since the eval lives inside deep search, it MUST be trained on the mid-tactical, non-quiet positions the search visits. This is the NNUE data-generation principle (search-visited positions labeled at low depth) and applies to BOTH the NN and GBDT evals; strong-game trajectories alone underrepresent it.

- :Dataset-curation: (OPTIONAL, deferred until the 1200 gate clears) is an importance+diversity filter on which labelled positions enter the :Cumulative-dataset:, borrowing temperature+MMR from LLM decoding: :Trajectory-sampling: gives move-level diversity, while an MMR-style selector keeps a position only if it is both *informative* — high teacher-vs-current-net value disagreement (hard-example / uncertainty sampling) — and *novel* — dissimilar (cosine over the net's trunk embedding) to positions already kept. Rationale: under the throughput constraint only a few gradient steps run per cycle, so each stored position should be maximally informative and non-redundant. It is a sample-efficiency refinement, OFF until rung 1 (1200) clears, to avoid premature complexity. Note: kept positions' trunk embeddings go stale as the net trains — either re-embed the kept set at curation time or accept the drift.

***implementation reqs***

- `distill_sf.py` owns Stage 1; labels come only from the :Teacher:.
- Constant: TEACHER_DEPTH, MULTIPV_K, POLICY_TARGET_TEMP, LABELS_PER_SEC_FLOOR (≥100/s per engine process) — developer-tuned distillation rules.
- Constant: TRAJECTORY_TEMP_OPENING / TRAJECTORY_TEMP_LATE and TRAJECTORY_OPENING_PLIES — the annealed :Trajectory-sampling: temperature bounds and the ply count over which it decays.
- Constant: CURATION_ENABLED (default false until the 1200 gate) and CURATION_MMR_LAMBDA — the :Dataset-curation: on/off flag and its relevance-vs-diversity balance.
- The trend log persists across runs; it is the Stage-1 observability artifact.

***test reqs***

- A fixed FEN with a known Stockfish depth-8 centipawn score, pinning the tanh(cp/400) convention and its sign.
- A GIL regression check: concurrent labelling + training throughput must stay within a small factor of the separate-process baselines (≈180 labels/s per engine; SGD steps unimpeded).

***functional specs***

- Labelling and training must never share a Python interpreter.
  - Given labelling runs concurrently with torch training, Then labellers occupy separate OS processes; If in-process threads are used instead, Then the run is non-conformant. (Pins the measured failure: threaded labellers + trainer yielded 58 labels/s, ~80 SGD steps in 8 minutes, and *worse* validation than a prior shorter run.)
- Policy targets must be soft where affordable, one-hot as fallback.
  - Given MultiPV output for a position, Then the policy target is softmax(cp_i / POLICY_TARGET_TEMP) over the K candidates with zero mass elsewhere.
  - Given MultiPV is unavailable or drops throughput below LABELS_PER_SEC_FLOOR, Then the one-hot best-move target is the fallback.
- The :Cumulative-dataset: must accumulate, deduplicate, and never regress.
  - Given a new label whose FEN already exists at >= its depth, Then it is not appended.
  - Given a completed run, Then dataset row count and :Measured-elo: are appended to the trend log.
- Stagnation must be surfaced, never annealed past.
  - Given no Elo improvement over M consecutive runs within a gate segment, Then escalation (more data, deeper teacher, more capacity) is flagged to the user — :Gated-progress: never advances to paper over stagnation.
- Positions must come from :Position-source: strong-game trajectories, not uniform random playouts.
  - Given a game generated by the :Teacher: from random opening plies, When positions are sampled from it, Then those in-distribution positions are labelled and appended to the :Cumulative-dataset:.
- :Position-source: games must advance by :Trajectory-sampling:, at no extra search cost.
  - Given the teacher's MultiPV top-K for a position, When the trajectory's next move is chosen, Then it is a temperature-softmax sample over those K candidates — no analyse beyond the one already run for the label.
  - Given opening plies, Then the sampling temperature is TRAJECTORY_TEMP_OPENING (diverse); Given plies past TRAJECTORY_OPENING_PLIES, Then it decays toward TRAJECTORY_TEMP_LATE (near-best), keeping positions near the strong-play manifold.
- The labelled set must include :Search-visited-positions:, not game-trajectory positions alone.
  - Given the eval will be queried inside deep search, When the dataset is built, Then a slice of positions is drawn from search-visited (non-quiet) states — teacher search-tree nodes or trajectory positions perturbed by 1–2 sampled-capture plies — and labelled at the teacher depth.
  - Given an eval trained only on quiet trajectory positions, Then it MUST be expected to degrade on the non-quiet positions search visits (distribution-dependent error); this applies to the NN and GBDT evals equally.
- :Trajectory-sampling: must not weaken the labels.
  - Given any sampled trajectory move, Then each position's value and policy LABEL is still the teacher's full TEACHER_DEPTH eval / MultiPV, independent of which move was sampled to continue the game.
- :Dataset-curation:, when enabled, must select on informativeness AND novelty, never informativeness alone.
  - Given a candidate labelled position and CURATION_ENABLED, Then it is kept only if its MMR score (relevance = teacher-vs-net value disagreement, diversity = 1 − max cosine similarity to kept positions, balanced by CURATION_MMR_LAMBDA) exceeds threshold.
  - Given CURATION_ENABLED is false (the default until the 1200 gate), Then all in-distribution labelled positions are kept (exact-FEN dedup only).
- Stage-1 exit is gated, not scheduled.
  - Given the distilled net (with its conformant search profile) clears the 1200 :Elo-gate: and then the 1600 :Elo-gate:, Then the next stage may begin; Given neither gate has cleared, Then self-play training MUST NOT start.
