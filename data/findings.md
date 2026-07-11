# Validated findings — TRUE POSITIVES ONLY (standing council input)

Rule of inclusion: measured with a comparator, reproduced, donor-replicated, or predicted-
then-observed. No hopes, no failure dispositions (those live in LESSONS.md / experiments.md).
Council rounds draw proposals from the **open questions** below; challenging a finding
requires new evidence.

---

**F1. The trivium compound target with a tuned annealed mix beats matched controls.**
Grade: CONTROLLED + REPRODUCED. Evidence: matched-depth rung 1237 vs 1166 un-annealed;
survived two 3-trial re-probes (999/1027 vs 998–1018; 822 vs 716–730); carried three
lineages (volume→1484 claims, triv→1540 claims, populations).
Open question: is a time-varying outcome→search mix near modern best practice, or is there
a stronger scheduled-target family (distributional, uncertainty-weighted)?
Query seeds: "mixed value targets outcome bootstrap weighting limitations" ·
"scheduled auxiliary target weighting reinforcement learning negative results" ·
"KataGo value target composition ablation".

**F2. Feature whitening ≈ 2.5× game efficiency; raw-space conditioning (cond ~4300) stalls
learning.** Grade: CONTROLLED both directions (whitened arm matched faithful on ¼ games;
the only unwhitened run ever — the clean baseline — was the flattest run ever).
Open question: is there a provenance-clean equivalent (random-play ZCA, online/natural
gradient, Adam-class per-feature scaling) that removes the trained-corpus dependency?
Query seeds: "natural gradient TD learning linear function approximation" ·
"feature whitening reinforcement learning drawbacks nonstationary" ·
"Adam vs SGD linear value function conditioning".

**F3. Teacher depth beyond soundness pays nothing: sound d2 targets == d4 at half clock.**
Grade: CONTROLLED (E-series adjudication).
Open question: what makes a shallow target "sound," and can soundness be detected online?
Query seeds: "shallow search training targets as good as deep chess TD" ·
"target depth ablation TDLeaf" · "when do deeper bootstrap targets hurt".

**F4. Noise-gated decision rules are load-bearing: confirmed crowns kill phantom bests;
informative patience prevents noise-deaths.** Grade: REPRODUCED (caught 36.12→31.61 and
others; arms now outlive the old 1,200-game ceiling).
Open question: optimal confirmation game-count vs decision error rate (sequential testing?).
Query seeds: "sequential probability ratio test model selection noisy evaluations" ·
"early stopping false positives noisy validation RL".

**F5. Maximization bias is a defect CLASS; bootstrapping only from BACKED values fixed a
universal 4-arm collapse (490→862).** Grade: CONTROLLED, S&B-predicted (skills 013–015).
Open question: remaining bias sources (softmax off-best records? PARGEN exact-min targets?).
Query seeds: "maximization bias function approximation remedies double estimator" ·
"off-policy corrections TD leaf targets".

**F6. Replicating the donor's CODE (not paper) broke the plateau: +264 Elo in one day after
the final fidelity line (full-width quiescence search).** Grade: the campaign's central
process finding; DONOR-REPLICATED.
Open question: which modern donor (public code + weights) is the right next replication
target at our scale?
Query seeds: "minimal open source NNUE training pipeline reproduce" ·
"smallest chess engine reinforcement learning reproducible baseline".

**F7. Material initialization ≈ 6 generations of zero learning.** Grade: CONTROLLED (same
noise stream, single variable; mat pop matched zero's 6-gen peak in 1 gen). Consistent with
the donor paper's material-only start.
Open question: what is the NEXT cheapest declared-constant knowledge injection, and is any
of it worth more than the games it saves?
Query seeds: "piece square table priors impact learning speed" ·
"handcrafted initialization vs learned chess evaluation ablation".

**F8. Search is a play-time converter (+125/ply donor era; material+d7 = 1527) AND depth
rungs are meaningless without an untrained-seed control at the same depth — training can
LOWER deep-search strength (trained mat 1326 < untrained seed 1527, non-overlapping).**
Grade: CONTROLLED, both directions. (Corollary: SF@1320's deliberate blunders donate ~free
tactics to any 7-ply material engine.)
Open question: why did d2-target training damage d7 play — depth-amplified eval holes?
Is there a training term that preserves deep-search compatibility?
Query seeds: "evaluation function overfitting shallow search hurts deep search" ·
"depth pathology chess evaluation learning" · "search-eval co-adaptation".

**F9. Optimistic initialization washes out under linear FA.** Grade: PREDICTED-THEN-OBSERVED
(council forecast from the literature, confirmed by the zero run's flat early epochs).
Open question: durable directed exploration for linear evals (count-based per-feature?).
Query seeds: "count based exploration linear function approximation" ·
"optimism washout function approximation fixes".

**F10. Ordering beats regression at matched cost: a 2-minute outcome-logistic fit on 522
positions equaled the champion TD net — but flattens on unbalanced outcomes.** Grade:
CONTROLLED, both halves.
Open question: ranking losses (pairwise/listwise) as the trivium's regression replacement?
Query seeds: "ranking loss evaluation function chess Texel tuning limitations" ·
"pairwise preference learning value function games".

**F11. The native/torch split is the right architecture at this scale: Rust search 2M nps
(~2000× python) with 2e-9 eval parity; torch keeps the learning loop.** Grade: REPRODUCED
infrastructure (v2→v3.6 across five side-builds).
Open question: incremental/lazy eval headroom (Merge 10) — how much nps is left on the table?
Query seeds: "NNUE accumulator incremental update speedup measurements" ·
"lazy evaluation margin alpha beta risks".

**F12. In chess, searchless from-scratch learning flatlines — with exploration maxed, anchor
removed, and two seeds: zero froze (11.17), material froze (13.94), clean floor 756.**
Grade: CONTROLLED (the wall; three-way triangulated with the donor arm's purist ceiling).
Open question: which single addition unlocks it first — policy storage (ExIt head),
nonlinear capacity (hidden units), or conditioning (clean whitening)?
Query seeds: "expert iteration small linear policy network results" ·
"minimum network capacity chess evaluation learning" ·
"TD-Gammon hidden units necessity linear baseline comparison".

**F13. Self-play volume pays only downstream of a sane eval: 15k games → 1484 claims on the
donor lineage; 13k games → flat from zero.** Grade: CONTROLLED contrast.
Open question: the volume × eval-quality interaction — is there a curriculum that makes
early volume useful from scratch?
Query seeds: "self-play curriculum from random initialization board games" ·
"when does more self-play data stop helping".
