# Model architectures

Every selectable agent (`chessdq/agents.py :: make_agent`) described in one format:
**what it is · substrate/arch · search · training · status**. The CHAMPION (the default) is
led by *what actually bought its strength*; the rest match that shape.

---

## CHAMPION (default) — linear eval, search-distilled, native alpha-beta

*Led by what actually bought the gain, not by the substrate.*

### 1. The improvement engine: search generates the signal, the eval learns it (the bitter-lesson part)
The Elo came from **compute, not human knowledge**. The move that produced **+240**: have the eval
**search its own positions deeper (d5)** than it was trained at (d2), then **re-fit the eval to
reproduce those deeper search values**. Search + learning did all the work — no feature designed, no
heuristic tuned; you could regenerate the gain with nothing but more CPU. A general method (search)
leveraging computation to teach the model, zero human insight injected.

### 2. Search — the compute multiplier
**Alpha-beta (minimax + pruning + quiescence), native Rust (`rsearch4`), depth 9 at play.** Used two
ways: at **train time** to generate the teaching signal (d2→d5 targets), at **play time** to convert
eval quality into strength. (P6: beats the MCTS λ-blend 0-16 — depth is the lever.)

### 3. Learning — two stages, both compute-driven
- **Stage 1 — Trivium (bootstrap from nothing).** Online TDLeaf(λ) self-play, compound target
  `G = a·λ-return + b·search-value + c·outcome`, weights annealed on an Optuna schedule (λ≈0.7
  KC-faithful, γ=1.0, d2 targets, ZCA whitening, RAMP filter, outcome-anchor annealing). λ substitutes
  for depth; the d2 glance keeps targets sound; outcome anchors early and fades with self-play volume.
  Pure self-play, no human labels. → the **1878 base**.
- **Stage 2 — Search-distillation (the +240).** Freeze the base as a teacher, label its self-play
  positions with its **own d5 search**, re-fit. **Search-value only** (no λ-return, no outcome — no
  trajectory, no games; just "reproduce the deeper search"). A frozen-teacher target network at
  infinite lag. → the **champion** (`models/champion.pt`, `distill: search-teacher-A`).

### 4. The value substrate (where the learned knowledge is stored)
A **linear** state-value `V(s) = tanh(w·φ(board) + b)` over **amap-897** features (pst-769
piece-square planes ⊕ 128 attack-coverage maps). This is the one place we did **not** follow the
bitter lesson — it's hand-engineered and non-scaling: P1 (d5→d7 = +47) and P2 (halfKP lost −512)
showed it caps out, exactly as the lesson warns. Kept because at this data scale it still beats raw
representation. The vessel, not the engine — and the honest ceiling.

### 5. Excluded (all measured)
No MCTS (P6), no actor-critic (search *is* the policy-improvement operator), no DQN/target-net
(distillation subsumes it), no halfKP/NNUE (P2, lost).

### 6. Strength & deployment
**~1840 absolute Elo (1721..1958, adjudicated) / +240 head-to-head** over the trivium-only
predecessor. Two runtime lanes (auto-detected): **Lane 1** pure-Python alpha-beta (`PyChampionAgent`,
pip-only, no Rust); **Lane 2** native `rsearch4` (full-speed d9 + difficulty dial).

**One line:** *deeper search generated better training targets and the eval learned them (+240, pure
compute) — the bitter lesson adopted, stored in a hand-feature linear substrate that then capped.*

---

## net+PUCT — AlphaZero-style deep net + MCTS

- **What it is:** a deep residual net with value+policy heads, played through PUCT Monte-Carlo tree search.
- **Substrate/arch:** `ChessResNet` (`resnet_model.py`) — 18-plane board tensor → residual tower
  (`RES_BLOCKS` × `RES_FILTERS`) → **dual heads**: value (Flatten→Linear→ReLU→Linear(1)→**tanh**) and
  policy (Flatten→Linear→move logits). AlphaZero's shape.
- **Search:** PUCT MCTS at inference (`puct_selfplay.py`), visit-count-averaged backup.
- **Training:** tabula-rasa self-play (AlphaStar-hybrid PUCT).
- **Status:** **parked.** Reached only *parity* with a 1-ply heuristic; tabula-rasa plateaus. Strength
  here is search, and MCTS's soft backup loses to alpha-beta on this low-variance eval (see CHAMPION P6).
  `models/tower_puct.pt`.

## alpha-beta engine — classical, no learning

- **What it is:** the baseline classical engine — hand-coded eval in the same alpha-beta search.
- **Substrate/arch:** `pst_eval` (`engine.py`) — material values + piece-square tables (`_PST`), no
  parameters learned, no net.
- **Search:** `AlphaBetaEngine` (minimax + pruning + quiescence, optional `phi_widen`), time-limited.
- **Training:** none — the eval is hand-authored.
- **Status:** reference baseline. Pure "human knowledge, no compute-learning" — the anti-bitter-lesson
  control the champion is measured against.

## nnue — halfKP-lite net in alpha-beta

- **What it is:** a small NNUE-style net (Stockfish-shaped) as the eval inside phi-widen alpha-beta.
- **Substrate/arch:** `NNUENet` (`nnue_model.py`) — sparse **halfKP-lite** features
  (`king_bucket*(64*10) + piece_sq*10 + piece_type`, 4 king buckets, **2560** indices) →
  **EmbeddingBag(2560→128)** accumulator → clipped-ReLU → Linear(128,32) → ReLU → Linear(32,1) →
  white-absolute centipawns. Incrementally updatable (`IncrementalNNUE`).
- **Search:** `AlphaBetaEngine` with `make_incremental_nnue_eval`, `phi_widen=True`.
- **Training:** supervised distillation (`train_nnue.py`). NB: the shipped `nnue.pt` used **Stockfish**
  labels; an **own-search** relabel (`data/distill_own_d5.jsonl`) is the purity-compliant variant.
- **Status:** **NO (P2).** At matched search depth the halfKP NNUE lost **−512 Elo** to the linear
  champion — capacity inherits the teacher ceiling; native port not pursued. `models/nnue.pt`.

## beam — net eval under a Fibonacci beam

- **What it is:** a net eval selected through a fixed-width beam search (root-commit argmax).
- **Substrate/arch:** a net eval driven by `NetBeam` (`play_beam.py`).
- **Search:** Fibonacci-spaced beam width over a `(depth, total_ops)` budget (default d6/140 ops),
  commits the root argmax.
- **Training:** inherits its net; the beam is a search policy, not a learner.
- **Status:** deliverable/legacy mover; superseded by native alpha-beta for the champion lineage.

---

## Model files (quick map)
- `models/champion.pt` — the CHAMPION above (amap-897 linear, search-distilled; `enc=amap, arch=linear`).
- `models/champion_backup_*.pt` — timestamped pre-promotion backups (revert targets).
- `models/champion_distillA{2,3,4}.pt` — distillation-iteration evidence (A2 = promoted champion).
- `models/tower_puct.pt` — net+PUCT ResNet. `models/nnue.pt` — halfKP NNUE.

Full campaign evidence: `spec/dispositioned.md`, `docs/LESSONS.md`, `spec/trivium.spec.md`.
