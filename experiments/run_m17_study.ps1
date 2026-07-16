# Merge 17: waits for the nk completion's KILL-CHECK, then launches the mlp capacity study
while (-not (Select-String -Path "data\train.log" -Pattern "KILL-CHECK" -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
$env:QLEARN_ARCH_UNUSED="marker"
$env:QLEARN_ENC="kc"; $env:QLEARN_SURPRISE="1"; $env:QLEARN_GRPO="1"; $env:QLEARN_DDQN="1"; $env:QLEARN_REPLAY_TUNE="1"
$env:QLEARN_OPP="graded"; $env:QLEARN_TDLEAF="1"; $env:QLEARN_TRIV_TUNE="1"; $env:QLEARN_KC_FAITHFUL="1"; $env:QLEARN_RAMP="1"
$env:QLEARN_RSEARCH_DEPTH="0"; $env:QLEARN_PARGEN="0"; $env:QLEARN_CONFIRM="1"; $env:QLEARN_PROXY_GAMES="4"; $env:QLEARN_DEV="cpu"
python tune_qlearn.py 3 100 1 12 20 mlp 64 q > data\tune_m17.log 2> data\tune_m17.err
