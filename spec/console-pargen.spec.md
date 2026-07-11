# Console surface: full trivium/PARGEN recipe drivable from the UI

Why: every trivium-era lane was launched out-of-band with raw env vars because the console
form never exposed the Merge 7–9 knobs (they existed in `TrainReq` but had no inputs and the
JS never sent them) and `TrainReq` lacked PARGEN entirely — so the UI showed stale runs and
the operator's "Load Optuna best → Start" flow couldn't reproduce the winning recipe.

## :TrainReq-pargen: (server.py)

- `pargen: int = 0` → `QLEARN_PARGEN` (Merge 9 native parallel self-play batches; excludes
  `opp=graded` by construction), `pargen_eps: float = 0.1` → `QLEARN_PARGEN_EPS`,
  `pargen_threads: int = 12` → `QLEARN_PARGEN_THREADS`, `pargen_opp_d: int = 1` →
  `QLEARN_PARGEN_OPP_D`.

## :Form-v2: (server.py PAGE)

- New inputs wired into `startTrain()`: KC faithful (M7), RAMP filter (M7), confirmed crowns
  (default ON), native search depth (M8, 0=off), native module (default `rsearch3`), ZCA path,
  trivium start/end triples ("λ,search,outcome"), trivium warmup, parallel native gen (M9)
  + threads, proxy games/sample, lineage. `input[type=text]` styled like number inputs.
- `loadBest()`: when the served study has :Trivium-anneal: dims (`c_start`,`c_end`,`b_srch`,
  `triv_warmup`), compose the env triples (`a = 1 − b − c(t)`) into the trivium fields;
  status line names the study suffix so the operator can see WHICH study was loaded.

## :Tuner-pargen-identity: (tune_qlearn.py)

- `QLEARN_PARGEN` set ⇒ `REGIME += "|pargen|v1"` (native generation changes behavior/target
  machinery → new study fingerprint, old studies untouched) and PARGEN knobs join the trial
  env passthrough. First study under this identity: `qlearn_elo_df2508cf`
  (q-s200-b20-e3-elo20-linear-tdleaf-d2w8-enckc, 24 trials, launched 2026-07-11).

## Acceptance

1. `py_compile` clean on server.py + tune_qlearn.py.
2. `GET /` serves the new form; `POST /api/train/start` with the trivium recipe reaches the
   trainer env (header line shows TDLEAF/KC-FAITHFUL/RSEARCH; metrics carry the run).
3. Tuner started with PARGEN env prints `new study …` (not RESUMING) — fresh fingerprint.
4. Operator flow: "Load Optuna best" fills trivium fields from the new study → "Start
   training" launches the full recipe from the console (spec :User-drives-final-runs:).
