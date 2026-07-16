# SOTA grounding notes (operator contract 2026-07-13: pull canon before designing)

Living synthesis of fetched canonical sources; every design below cites its source.
Rule: transplant the proven procedure whole; departures declared inline.

## Engine A/B testing (fishtest canon)

- GSPRT, α = β = 0.05, LLR accept/reject bounds ±2.94 = ln((1−β)/α); bounds (elo0,
  elo1) expressed in NORMALIZED Elo so test duration is book/draw-ratio independent.
  [source: official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html,
  chessprogramming.org/Sequential_Probability_Ratio_Test]
- Outcomes are modeled PENTANOMIALLY over game PAIRS (LL, LD, DD+WL, WD, WW) — the
  paired-opening design we already use; pentanomial accounting "yields substantial
  saving of testing resources".
- ADOPT: future H2H gates run SPRT with pentanomial pair outcomes instead of fixed-N
  (stops early on clear results). The charter's in-flight 600-game duels stay fixed-N
  (launched before this note; Wilson band remains valid, just less efficient).

## NNUE training loss (nnue-pytorch canon, docs/nnue.md)

- WDL space: `wdl = sigmoid(cp / scale)`; scale is ENGINE-SPECIFIC (Stockfish example
  410) — ours must be declared/fit to our own eval scale at export time.
- Target: `wdl_value = λ·sigmoid(eval/scale) + (1−λ)·game_result`; loss = MSE in wdl
  space; Stockfish measured good results with loss exponent 2.6 (|err|^2.6).
- Our purity mapping: eval field = OWN compound TDLeaf target (cp-mapped), λ = 1
  bullet-side (outcome already blended inside our target — avoids double-count; this
  matches the SME direction and the canon's λ semantics).
- **Feature factorization** "helps the net generalize" and matters mostly EARLY in
  training — the canon's one documented small-data-relevant lever; note for any
  king-bucket arm (factorized buckets share gradients with the unbucketed parent).
- NOTABLE ABSENCE: the canonical docs prescribe NO minimum data volume. The community
  10⁸-positions figure is practice lore, not doctrine — the volume question is
  legitimately empirical (what the charter's duels measure).

## bullet trainer I/O (jw1912/bullet docs/3-data.md)

- Simplest ingestion: text lines `<FEN> | <score> | <result>` — score white-relative
  centipawns, result white-relative 1.0/0.5/0.0 — converted to bulletformat;
  `DirectSequentialDataLoader` explicitly adequate for SMALL networks.
- Game-contiguous binpacks (viriformat most tooling) need noisy-position filters;
  bullet-utils does shuffle/interleave/convert.
- Our exporter is therefore ~trivial: emit the text format from self-play records
  (purity: all fields self-generated).
