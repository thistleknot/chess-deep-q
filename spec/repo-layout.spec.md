# Repo layout spec — :Package-restructure: (2026-07-15, operator order)

## Why
Root held ~200 files (117 .py, 73 .log/.md, misc). Operator: "we need proper
folder structure, else no one is going to take this repo seriously."

## Target structure
```
chess-deep-q/
├── README.md  LICENSE  requirements.txt  future_exploration.md  .gitignore
├── app.py   main.py            # the two real entry points (console UI, terminal menu)
├── qlearn.py head2head.py claims_rung.py lane.py bakeoff.py human_replay.py
│                                # THIN SHIMS: forward to `python -m chessdq.<name>`
│                                # so every documented command keeps working verbatim
├── chessdq/                     # the live package: all imported library modules
│                                # + live CLIs (server, head2head, qlearn, ...)
├── experiments/                 # historical one-off arms/scripts (never imported by
│                                # live code); each carries a repo-root sys.path shim;
│                                # run from repo root
├── docs/                        # FINDINGS, RL_FINDINGS, GLOSSARY, LESSONS, ROLLBACK,
│                                # cliff_notes, codebase.md
├── logs/                        # all stray root *.log
└── spec/ data/ models/ rsearch/ engines/ reference/ archive/ legacy/
    images/ saved_games/ training_plots/ bullet_arms/ mlruns/   (unchanged)
```

## Classification rule (deterministic, from the import graph)
- Module imported by ANY other module → `chessdq/` (even if only dead scripts
  import it — experiments import `chessdq.*`, so they still resolve).
- Never-imported leaf reachable from the console/menu/infra → `chessdq/`
  (server, head2head, human_replay, bakeoff, finish10, ac_learn, watch_crowns,
  play_engine, play_puct).
- Every other leaf → `experiments/`.

## Contracts
- **Require**: all commands run from repo root (unchanged convention).
- **Guarantee**: every operator-documented invocation (`python app.py`,
  `python head2head.py enc:...`, `python lane.py run ...`, console buttons,
  menu options) behaves identically post-move. Checkpoints unaffected
  (state_dict bundles; verified `weights_only=True` loads).
- **Maintain**: package imports are ABSOLUTE (`from chessdq.engine import ...`)
  in both `chessdq/` and `experiments/` files, so moving a file between those
  dirs never changes its imports.
- **Assert** (verification battery, all must pass):
  1. `import` sweep over every `chessdq/*` module — zero failures.
  2. Champion loads and produces a legal move via the agents path.
  3. Server boots; `/` and `/api/lanes` respond; lane reap runs on boot.
  4. `python lane.py ls` and a shim invocation (`python head2head.py` usage
     path) work from root.
  5. `python -m compileall chessdq experiments` clean.

## Mechanics that made this safe
- multiprocessing spawn: live CLIs run as `python -m chessdq.X` (shims forward),
  so worker pickles resolve `chessdq.X._worker` in children.
- server.py: ROOT is the package's parent; spawned scripts use `-m chessdq.X`.
- app.py: uvicorn target `chessdq.server:app`.
- torch checkpoints: state_dict bundles only — no pickled class module paths.

## .gitignore additions
`models/*.pt` except `models/champion*.pt` (experiment checkpoints stay local);
`mlflow.db`, `mlruns/`.
