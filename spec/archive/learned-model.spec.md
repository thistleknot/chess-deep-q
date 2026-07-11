---
description: 'The dual-head learned network (residual tower: value + policy) and the agent that trains it'
import:
  - annealing-schedule
  - prior-evaluator
---

***definitions***

- :Learned-value: is the network's scalar estimate of a position's worth in the :White-absolute-frame:, in [-1,1], updated by training and intended to eventually outrun every earlier :Prior-lineage: member.
- :Policy-head: is the network's distribution over a flat move index (from-square × to-square + promotion piece); it is trained by cross-entropy — on Stockfish MultiPV soft targets during distillation, and on MCTS root visit distributions during self-play — never by policy gradient.
- :Residual-tower: is the shared trunk feeding both heads: a small residual convolutional stack (on the order of 4–6 blocks × 64 filters) sized by the throughput budget below, not by a fixed architecture clause. (This supersedes the earlier "value-only is intentional" clause, which is deleted: the policy head is required for PUCT self-play.)
- :Input-planes: is the board encoding: 12 piece-placement planes plus a mandatory side-to-move plane (and castling/en-passant planes where implemented). The side-to-move plane became mandatory the moment a :Policy-head: exists — 12 placement planes cannot identify the mover, and a policy is meaningless without knowing whose move it is. The value output remains White-absolute regardless.
- :Encoding-packing: is the storage rule for the binary :Input-planes:: the encoded dataset and replay board tensors are 0/1 planes (several constant-fill), so they MUST be stored PACKED (uint8 or bit-packed) and unpacked to float only at batch assembly — fp32-on-disk is up to 32× wasteful (the ~217 MB encoded cache packs to single-digit MB). This is the memory reduction that COUNTS. WEIGHT quantization (NF4 / QLoRA-style) is redundant here: the tower is ~1 M params (~4 MB), so 4-bit saves nothing material, and the compute-precision win is already taken by AMP fp16 autocast. The binding constraints are batched-eval throughput (NET_MIN_BATCHED_EVALS_PER_SEC) and DATA storage, not weight memory — precision reduction is applied to DATA, never weights.
- :Value-target-convention: is the single scale shared by every stage: tanh(white_centipawns / 400), White-absolute, in [-1,1] — used identically by teacher labels, leaf blending, TD targets, and the terminal ±1 outcomes it must be commensurate with.
- :DQN-agent: is the `service` that owns the network(s), the target network, the replay buffer, and the training step; it coordinates learning but delegates move choice to search. (Name kept for compatibility; see rl-categorization.)
- :Replay-transition: is a stored (state, move, reward, next-state, done) tuple, the off-policy record of experience the learner samples from.
- :Terminal-reward: is the fixed game-outcome signal (+1 win / 0 draw / -1 loss) that the objective must ultimately serve.

***implementation reqs***

- Constant: GAMMA, LEARNING_RATE, BATCH_SIZE, REPLAY_CAPACITY, TARGET_UPDATE_INTERVAL — developer-tuned learning rules, centralized in `constants.py`.
- Constant: NET_MIN_BATCHED_EVALS_PER_SEC — the throughput budget any capacity change must respect. Empirical anchors (tiny 2-conv net, batch 256): ~18k evals/s CPU, ~122k evals/s GPU; single-position ~2–3 ms. Any tower change is conformant only if it still meets the constant.
- The network MUST expose a batch-evaluate API (one forward pass for a batch of positions); search code that loops single-position inference is non-conformant.
- Encoded board tensors and replay states MUST be stored packed (uint8/bit-packed per :Encoding-packing:) and unpacked to float only per batch; fp32-on-disk encodings are non-conformant. Weight quantization is out of scope (nets too small; AMP owns the compute-precision win).

***test reqs***

- A position one legal move from mate for each color, to assert terminal reward sign.
- A position and its color-mirror with side to move flipped, to assert the symmetry spec below.
- A throughput benchmark batch, to assert NET_MIN_BATCHED_EVALS_PER_SEC.

***functional specs***

- :Learned-value: must share the :White-absolute-frame: and :Value-target-convention: with every :Prior-lineage: member so all values are blendable at a search leaf.
  - Given a position winning for White, Then a trained :Learned-value: should be positive.
- Color symmetry must hold across both heads.
  - Given a position and its color-mirror with turn flipped, Then the value negates and the policy distribution maps through the mirror transform.
- Batched evaluation must be the only evaluation path used at scale.
  - Given a batch of B positions, Then evaluation is one forward pass (or ⌈B/max_batch⌉ passes), never B single-position calls.
  - Given the benchmark batch, Then measured throughput >= NET_MIN_BATCHED_EVALS_PER_SEC.
- :Terminal-reward: must be signed by the winner, not hardcoded.
  - Given White delivers checkmate, When the transition is stored, Then reward should be +1; Given Black delivers checkmate, Then reward should be -1. (This corrects a sign that previously taught winning positions as losses.)
- Intermediate reward must be **potential-based** (Ng 1999), so it cannot change the optimal policy: the shaping term is F = γ·Φ(s′) − Φ(s), where Φ(s) is the operative :Prior-lineage: member's normalized score, scaled by :Shaped-reward-weight:. Raw `weight × prior-score` shaping is non-conformant — it can alter the optimum the :Terminal-reward: defines.
  - Given the potential-based form, Then the optimal policy is provably invariant to the shaping and :Shaped-reward-weight: (the floor) is harmless, not merely small. (Guards prompt.md's "reward hacking" by construction, not by magnitude.)
  - Given late training, Then shaped magnitude is small relative to :Terminal-reward: regardless, so game result dominates the learning signal.
- The :DQN-agent: must learn off-policy from sampled :Replay-transition:s with a periodically-synced target network.
  - If the buffer holds fewer than BATCH_SIZE transitions, Then the training step is a no-op.
- Exploration must be structured (search-driven), so the agent must not expose a uniform-random epsilon path as its behavior policy.
  - The vestigial epsilon-greedy branch should be removed; move choice comes from search, not a coin flip.
  - The standard A2C entropy-bonus (c_e·H(π) in the policy loss) is DELIBERATELY ABSENT: the policy is trained by cross-entropy to MCTS visit counts (expert iteration), and exploration entropy is injected UPSTREAM as Dirichlet root noise + visit temperature (AlphaZero-style, see self-play-leela). So the missing H(π) term is a documented choice, not an omission; adding a policy-gradient entropy bonus would double-count the exploration the search targets already carry.
