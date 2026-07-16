import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Stage-1 run contract (spec/teacher-distillation): one <=5-minute cycle =
label (separate OS processes) -> merge/dedup into the cumulative dataset -> train the tower ->
measure Elo vs the real Stockfish anchor -> append the trend log.

Usage: python run_stage1.py [label_s] [train_s] [workers] [sf_depth] [games]
Defaults sized so label+train fits ~5 minutes; measurement follows immediately.
"""

import os
import sys
import json
import glob
import time
import random
import subprocess

import chess
import chess.engine
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True   # autotune convs for the fixed 8x8 shape

from chessdq.resnet_model import ChessResNet, encode18, move_index, POLICY_SIZE
from chessdq.elo_calibration import score_to_elo_diff
from chessdq import net_search

DATA = "data/distill_sf.jsonl"
MODEL = "models/tower.pth"
ENC = "models/encoded.pt"
TREND = "models/trend.jsonl"
SF_ANCHOR = 1320
POLICY_LOSS_WEIGHT = 0.5


def find_sf():
    return glob.glob("engines/**/stockfish*.exe", recursive=True)[0]


def phase_label(seconds, workers, depth):
    os.makedirs("data/shards", exist_ok=True)
    procs = []
    for i in range(workers):
        shard = f"data/shards/shard_{i}.jsonl"
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "chessdq.labeler_sf", shard, str(seconds), str(depth), str(i + 1)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    # Merge shards into the cumulative dataset, dedup by FEN.
    seen = set()
    if os.path.exists(DATA):
        with open(DATA) as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)[0])
                except Exception:
                    continue
    added = 0
    with open(DATA, "a") as out:
        for shard in glob.glob("data/shards/shard_*.jsonl"):
            with open(shard) as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row[0] in seen:
                        continue
                    seen.add(row[0])
                    out.write(json.dumps(row) + "\n")
                    added += 1
            os.remove(shard)
    return added, len(seen)


def load_dataset():
    """Rows: (fen, value, best_uci|None). Older rows may lack the move -> value-only."""
    rows = []
    with open(DATA) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            rows.append((row[0], float(row[1]), row[2] if len(row) > 2 else None))
    return rows


def encode_dataset(rows):
    """Incrementally encode the append-only dataset; cache tensors so each run only encodes
    the newly-added rows (dataset order is stable — dedup keeps first occurrence, appends at end)."""
    cached = None
    if os.path.exists(ENC):
        try:
            cached = torch.load(ENC)
        except Exception:
            cached = None
    start = cached["n"] if (cached and cached["n"] <= len(rows)) else 0
    if start == 0:
        cached = None
    t0 = time.time()
    newX, newy, newp = [], [], []
    for fen, v, best in rows[start:]:
        newX.append(encode18(chess.Board(fen)))
        newy.append(v)
        idx = -1
        if best:
            try:
                idx = move_index(chess.Move.from_uci(best))
            except Exception:
                idx = -1
        newp.append(idx)
    if cached:
        X = torch.cat([cached["X"], torch.stack(newX)]) if newX else cached["X"]
        y = torch.cat([cached["y"], torch.tensor(newy, dtype=torch.float32)]) if newy else cached["y"]
        pol = torch.cat([cached["pol"], torch.tensor(newp, dtype=torch.long)]) if newp else cached["pol"]
    else:
        X = torch.stack(newX)
        y = torch.tensor(newy, dtype=torch.float32)
        pol = torch.tensor(newp, dtype=torch.long)
    os.makedirs("models", exist_ok=True)
    torch.save({"n": len(rows), "X": X, "y": y, "pol": pol}, ENC)
    print(f"  encoded +{len(newX)} new (total {len(rows)}) in {time.time()-t0:.0f}s")
    return X, y, pol


def phase_train(seconds, device, batch=512):   # bs512 ~1s/step here; bs2048 was 12s/step (Max-Q throttle)
    rows = load_dataset()
    X, y, pol = encode_dataset(rows)

    net = ChessResNet().to(device)
    if os.path.exists(MODEL):
        try:
            net.load_state_dict(torch.load(MODEL, map_location=device)["state_dict"])
            print("  resumed tower weights")
        except Exception as e:
            print(f"  fresh tower ({e})")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Move the whole dataset to the device ONCE (index on-GPU each step; per-step host->device
    # copies were the ~1 step/s bottleneck). Falls back to CPU-resident + per-batch copy on OOM.
    n = len(rows)
    on_device = True
    try:
        Xd, yd, pd = X.to(device), y.to(device), pol.to(device)
    except RuntimeError:
        Xd, yd, pd, on_device = X, y, pol, False
        print("  dataset too large for device; using host-resident batches")
    perm = torch.randperm(n)
    vcount = min(2000, n // 10)
    val_idx = perm[:vcount].to(device)
    train_idx = perm[vcount:]
    Xv = Xd[val_idx] if on_device else X[perm[:vcount]].to(device)
    yv = yd[val_idx] if on_device else y[perm[:vcount]].to(device)

    use_amp = (device.type == "cuda")   # fp16 autocast ~2x on the Max-Q GPU (Q1 test)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    steps = 0
    start = time.time()
    while time.time() - start < seconds:
        sel = train_idx[torch.randint(len(train_idx), (batch,))]
        if on_device:
            sel = sel.to(device)
            xb, yb, pb = Xd[sel], yd[sel], pd[sel]
        else:
            xb, yb, pb = X[sel].to(device), y[sel].to(device), pol[sel].to(device)
        opt.zero_grad()
        with torch.autocast("cuda", enabled=use_amp):
            v, p = net(xb)
            loss = F.mse_loss(v.squeeze(1), yb)
            mask = pb >= 0
            if mask.any():
                loss = loss + POLICY_LOSS_WEIGHT * F.cross_entropy(p[mask], pb[mask])
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        steps += 1

    net.eval()
    with torch.no_grad():
        vp, _ = net(Xv)
        vmse = F.mse_loss(vp.squeeze(1), yv).item()
        sign = ((vp.squeeze(1) > 0) == (yv > 0)).float().mean().item()
    torch.save({"state_dict": net.state_dict()}, MODEL)
    print(f"  trained {steps} steps in {time.time()-start:.0f}s  val_mse {vmse:.4f} sign_acc {sign:.2f}")
    return net, vmse, sign, len(rows)


def phase_measure(net, device, games, sf_ms=50, cap=90, depth=2):
    sf = chess.engine.SimpleEngine.popen_uci(find_sf())
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": SF_ANCHOR})
    lim = chess.engine.Limit(time=sf_ms / 1000.0)
    total = 0.0
    for g in range(games):
        net_white = (g % 2 == 0)
        b = chess.Board()
        plies = 0
        while not b.is_game_over() and plies < cap:
            if (b.turn == chess.WHITE) == net_white:
                mv = net_search.best_move(b, net, device, depth=depth, beam=8, encode_fn=encode18)
            else:
                mv = sf.play(b, lim).move
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv)
            plies += 1
        if b.is_checkmate():
            ww = b.turn == chess.BLACK
            pts, res = (1.0 if ww == net_white else 0.0), ("1-0" if ww else "0-1")
        elif b.is_game_over():
            pts, res = 0.5, b.result()
        else:
            cp = sf.analyse(b, chess.engine.Limit(time=0.05))["score"].white().score(mate_score=100000)
            if cp is None or abs(cp) <= 150:
                pts, res = 0.5, "adj-draw"
            else:
                white_won = cp > 150
                pts, res = (1.0 if white_won == net_white else 0.0), ("adj-1-0" if white_won else "adj-0-1")
        total += pts
        print(f"  game {g+1}/{games}: net={'W' if net_white else 'B'} {res:9s} pts={pts:.1f} plies={plies}")
    sf.quit()
    score = total / games
    if 0 < score < 1:
        elo = SF_ANCHOR + score_to_elo_diff(score)
    else:
        elo = SF_ANCHOR + (400 if score >= 1 else -400)
    return score, elo


def main():
    label_s = float(sys.argv[1]) if len(sys.argv) > 1 else 120
    train_s = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    games = int(sys.argv[5]) if len(sys.argv) > 5 else 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_start = time.time()

    print(f"[label] {workers} workers x {label_s:.0f}s @ SF depth-{depth} (separate processes)...")
    added, total_rows = phase_label(label_s, workers, depth)
    print(f"  +{added} labels -> {total_rows} total\n")

    print(f"[train] tower for {train_s:.0f}s on {device}...")
    net, vmse, sign, rows = phase_train(train_s, device)
    core = time.time() - run_start
    print(f"  label+train wall: {core:.0f}s\n")

    if games <= 0:
        print("[measure] skipped (games=0)")
        return
    print(f"[measure] net-minimax d2 vs SF@{SF_ANCHOR}, {games} games (in-loop TREND HINT only)...")
    score, elo = phase_measure(net, device, games)
    # NON-AUTHORITATIVE: at n~6 the per-game SD (~0.13) is wider than a gate step, so this Elo is a
    # coin flip, not a measurement. It exists only to sketch the trend cheaply inside the run
    # contract. Gate decisions MUST use `python measure_sf.py <net> 30` (n>=30 with CIs); never gate
    # on this number (spec/elo-measurement.spec.md power rule).
    print(f"\nscore {score:.2f} -> HINT Elo ~{elo:.0f} (n={games}, NOT a gate measurement; "
          f"run measure_sf.py at n>=30 for a real anchor)")

    with open(TREND, "a") as fh:
        fh.write(json.dumps({"rows": rows, "added": added, "val_mse": round(vmse, 4),
                             "sign_acc": round(sign, 3), "score": score, "elo": round(elo),
                             "n": games, "label_s": label_s, "train_s": train_s,
                             "core_wall_s": round(core)}) + "\n")
    print(f"trend appended to {TREND}; run wall {time.time()-run_start:.0f}s")


if __name__ == "__main__":
    main()
