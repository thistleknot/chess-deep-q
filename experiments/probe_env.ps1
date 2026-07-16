# Volume-curve probe shared env (Merge 20 gate) — EXACT bake4 trial-0 regime:
# proven parms, Adam+replay (KC_FAITHFUL=0), TDLeaf d2w8, graded ladder, pst enc.
# Dot-source, then set QLEARN_ARCH/QLEARN_CRELU/QLEARN_CKPT/QLEARN_METRICS/QLEARN_TAG per arm.
$env:QLEARN_GAMMA = "0.9900"; $env:QLEARN_ALPHA = "0.000300"; $env:QLEARN_LAMBDA = "0.8000"
$env:QLEARN_WARMUP = "0.4000"; $env:QLEARN_LAMBDA_WARMUP = "0.8000"; $env:QLEARN_TAU_FLOOR = "0.0500"
$env:QLEARN_TRIVIUM = "0.285,0.341,0.374"; $env:QLEARN_TRIVIUM_END = "0.516,0.341,0.143"
$env:QLEARN_TRIVIUM_WARMUP = "0.481"
$env:QLEARN_ENC = "pst"; $env:QLEARN_OPP = "graded"; $env:QLEARN_TDLEAF = "1"
$env:QLEARN_SEARCH_DEPTH = "2"; $env:QLEARN_SEARCH_WIDTH = "8"
$env:QLEARN_KC_FAITHFUL = "0"; $env:QLEARN_RAMP = "1"; $env:QLEARN_RSEARCH_DEPTH = "0"
$env:QLEARN_PARGEN = "0"; $env:QLEARN_FREEZE_EPOCH = "1"; $env:QLEARN_ADAPTIVE_LAMBDA = "1"
$env:QLEARN_BATCH_GAMES = "20"; $env:QLEARN_ELO_GAMES = "12"; $env:QLEARN_EPOCH_ELO_GAMES = "12"
$env:QLEARN_PROXY_GAMES = "6"; $env:QLEARN_SEED = "0"; $env:QLEARN_RESUME = "1"
$env:QLEARN_HIDDEN = "64"; $env:QLEARN_DEV = "cpu"
