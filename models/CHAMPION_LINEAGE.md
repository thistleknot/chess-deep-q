# Champion lineage — research bootstrap base

Every crowned/promoted net in this project's history, in order, each with its governing
spec and Elo receipt. Kept in `models/` (not archived) so future research can resume,
compare against, or re-distill from any rung without re-running the full campaign.

| Rung | File | Elo | Instrument | Spec | Commit |
|---|---|---|---|---|---|
| 1 | `champion_backup_kc1670.pt` | 1670 claims (95% 1605..1762) | 200g vs SF@1320, rsearch d9, zero losses | `spec/archive/knightcap-full.spec.md` | `77ff00c` |
| 2 | `champion_backup_grad1600.pt` | 1572 claims (95% 1516..1642) | 200g vs SF@1320, rsearch d7 | `spec/archive/q-learning.spec.md` | `42303a7` |
| 3 | `champion_backup_20260721T034404Z.pt` | 1878 (95% 1756..2000) — pre Lane-1 pip-only change | multi-anchor MLE | `spec/search-arms.spec.md`, `spec/trivium.spec.md` | `138e172` |
| 4 | `champion_backup_20260722T134534Z_predistill.pt` | 1878 (95% 1756..2000) — same net, backup taken immediately before distillation promotion | multi-anchor MLE; floor 1724 @d9 native amap-897 | `spec/search-arms.spec.md`, `spec/trivium.spec.md` | `9578615`, `5d85cad` |
| 5 | `champion_distillA2.pt` **(current champion.pt)** | ~1840 absolute (1721..1958); **+232 Elo H2H** over rung 4 | native-d9 head-to-head (the valid ruler above the top ladder anchor — see `spec/dispositioned.md` SEARCH-DISTILL FINAL) | `spec/search-distill.spec.md` | `1ea4417`, `e4b1895` |
| 6 | `champion_distillA3.pt` | same absolute tier; **+221 Elo H2H** over rung 4 | native-d9 head-to-head | `spec/search-distill.spec.md` | `1ea4417` |
| 7 | `champion_distillA4.pt` | same absolute tier; **+267 Elo H2H** over rung 4 | native-d9 head-to-head | `spec/search-distill.spec.md` | `1ea4417` |

Rungs 5/6/7 are three independent single-step distillation runs off rung 4's labels;
A2 was promoted (SWA-averaging across them diluted the gain — see `spec/dispositioned.md`
SEARCH-DISTILL FINAL). A3/A4 are kept as same-tier comparison points, not inferior rejects.

**Key instrument lesson** (see `spec/dispositioned.md`): the vs-SF anchor ladder
draw-floods agents at/above the top anchor and understates real gains — e.g. A2 measured
only +39 on the ladder but +232 on direct native head-to-head. For anything at or above
this tier, use the H2H instrument (`spec/h2h-instrument.spec.md`), not the ladder.

## Everything else in models/

~200 one-off experiment/probe/arm-screen checkpoints, none referenced by any code path,
moved to `models/archive/` (2026-07-25 cleanup) to declutter the working directory. They
remain on disk for provenance but are not part of the research bootstrap base above.
Files still at `models/` root beyond this lineage are live working state (`champion.pt`,
`champion_hb.pt`, Optuna `.db`/`.pt` pairs, encoder seeds) referenced directly by
`chessdq/` or `experiments/` scripts — left in place.
