import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))
"""Train NNUE on uncertainty-enriched corpus to convergence, then measure.

Runs 200 epochs with cosine LR decay + weight decay. Saves best-by-val checkpoint.
After training, runs the standard 100g anchor ladder measurement.

Usage: python experiments/train_nnue_unc.py
  (no args — fully self-contained, writes logs to data/nnue_unc_train.log)
"""
import os
import sys
import json
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
import chess

from chessdq.nnue_model import NNUENet, features, make_nnue_eval, NNUE_ACC_DIM, NNUE_HIDDEN
from chessdq.gbdt_features import cp_from_tanh
from chessdq import augment

CORPUS = "data/distill_nnue_unc.jsonl"
OUT = "models/nnue_unc.pt"
LOG = "data/nnue_unc_train.log"
EPOCHS = 200
BATCH = 2048
LR_MAX = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
CP_CLIP = 2000.0
PATIENCE = 30  # early stop if val doesn't improve for this many epochs


def load_rows(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            elif isinstance(obj, list) and len(obj) >= 2:
                rows.append({"fen": obj[0], "value": obj[1]})
    return rows


def rows_to_xy(rows, do_augment):
    idxs, ys = [], []
    for r in rows:
        variants = [r]
        if do_augment:
            m = augment.mirror(r)
            if m is not None:
                variants.append(m)
            variants.append(augment.color_flip(r))
        for v in variants:
            cp = max(-CP_CLIP, min(CP_CLIP, cp_from_tanh(float(v["value"]))))
            idxs.append(features(chess.Board(v["fen"])))
            ys.append(cp)
    return idxs, np.array(ys, dtype=np.float32)


def batch_tensors(idx_lists, dev):
    offsets, flat = [], []
    for lst in idx_lists:
        offsets.append(len(flat))
        flat.extend(lst if lst else [0])
    return (torch.tensor(flat, dtype=torch.long, device=dev),
            torch.tensor(offsets, dtype=torch.long, device=dev))


def cosine_lr(epoch, total, lr_max, lr_min):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logfh = open(LOG, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logfh.write(msg + "\n")
        logfh.flush()

    log(f"NNUE uncertainty training: {EPOCHS} epochs, batch={BATCH}, lr={LR_MAX}->{LR_MIN} cosine")
    log(f"corpus: {CORPUS}, output: {OUT}, device: {dev}")

    rows = load_rows(CORPUS)
    n = len(rows)
    log(f"loaded {n} rows")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    nval = min(5000, n // 10)
    val_rows = [rows[i] for i in perm[:nval]]
    tr_rows = [rows[i] for i in perm[nval:]]

    log("preparing train (augmented) + val...")
    idxs_tr, y_tr = rows_to_xy(tr_rows, do_augment=True)
    idxs_val, y_val = rows_to_xy(val_rows, do_augment=False)
    ntr = len(y_tr)
    log(f"train: {ntr} (aug from {len(tr_rows)}) | val: {len(y_val)} | "
        f"cp range [{y_tr.min():.0f},{y_tr.max():.0f}]")

    net = NNUENet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    yv = torch.tensor(y_val, device=dev)
    vf, vo = batch_tensors(idxs_val, dev)

    best_val = float("inf")
    best_ep = 0
    t0 = time.time()

    for ep in range(EPOCHS):
        # Cosine LR
        lr = cosine_lr(ep, EPOCHS, LR_MAX, LR_MIN)
        for pg in opt.param_groups:
            pg["lr"] = lr

        net.train()
        order = rng.permutation(ntr)
        losses = []
        for s in range(0, ntr, BATCH):
            bi = order[s:s + BATCH]
            feats, offs = batch_tensors([idxs_tr[j] for j in bi], dev)
            tgt = torch.tensor(y_tr[bi], device=dev)
            opt.zero_grad()
            loss = F.mse_loss(net(feats, offs), tgt)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        net.eval()
        with torch.no_grad():
            vp = net(vf, vo)
            rmse = torch.sqrt(F.mse_loss(vp, yv)).item()
            sign = ((vp > 0) == (yv > 0)).float().mean().item()

        improved = rmse < best_val
        if improved:
            best_val = rmse
            best_ep = ep
            torch.save({"state_dict": net.state_dict(),
                        "acc_dim": NNUE_ACC_DIM, "hidden": NNUE_HIDDEN,
                        "epoch": ep, "val_rmse": rmse}, OUT)

        if ep % 5 == 0 or ep == EPOCHS - 1 or improved:
            elapsed = time.time() - t0
            log(f"ep {ep:4d}  lr={lr:.1e}  train_mse={np.mean(losses):.0f}  "
                f"val_rmse={rmse:.0f}cp  sign={sign:.3f}  "
                f"{'*BEST*' if improved else ''}  [{elapsed:.0f}s]")

        # Early stop
        if ep - best_ep >= PATIENCE:
            log(f"early stop at ep {ep} (no improvement for {PATIENCE} epochs, best={best_ep})")
            break

    # Reload best
    ck = torch.load(OUT, map_location=dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    log(f"\nbest checkpoint: ep {ck['epoch']}, val_rmse={ck['val_rmse']:.0f}cp")

    # Sanity
    ev = make_nnue_eval(net, dev)
    for name, fen in [("start", chess.STARTING_FEN),
                      ("white +Q", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                      ("black +Q", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")]:
        log(f"  {name}: {ev(chess.Board(fen)):+.0f} cp")

    log(f"\ntraining done in {time.time()-t0:.0f}s. saved {OUT}")
    log(f"next: python -m experiments.anchor_ladder nnue:{OUT} 0 0 nnue_unc")
    logfh.close()


if __name__ == "__main__":
    main()
