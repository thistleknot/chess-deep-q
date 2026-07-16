# experiments/ — historical one-off arms and scripts

Archived research scripts from the campaign (bake-offs, climb chains, probe
arms, measurement variants). None are imported by live code; the live package
is `chessdq/` and the console entry points are `app.py` / `main.py` at the
repo root.

- Run from the **repo root**: `python experiments/<name>.py ...`
  (each file carries a repo-root `sys.path` shim so `chessdq.*` imports resolve).
- These predate the :Package-restructure: (spec/repo-layout.spec.md) and are
  kept for provenance — the findings they produced live in `spec/` and `docs/`.
- Anything here that graduates back into active use moves to `chessdq/`
  unchanged (imports are absolute `chessdq.*` in both directories by design).
