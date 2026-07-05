"""Train the NNUE eval (spec/nnue-eval.spec.md :NNUE-training:) — Stage-2 off-policy distillation.

Supervised MSE(V, SF centipawns) on the Stage-1 ENRICHED corpus (data/distill_v2.jsonl: in-distribution
trajectory positions, quiet-filtered, dict schema with tanh value). This IS the :Distillation-anchor:
at alpha=1 — the whole objective is the supervised tether; Stage 4 (Expert Iteration) later blends a
self-play TD-Leaf term against this same anchor so self-play cannot erode the distilled floor.

:Label-augmentation: — the training split is expanded ~3x by the two valid chess symmetries (horizontal
mirror when castling-free, color-flip with sign-negated value), the free variance-reduction lever on a
scarce corpus. Val is NOT augmented (clean held-out).

Architecture is HELD at v1 (128 acc / 32 hidden) so this run isolates the hypothesis under test: does
better DATA (enriched + more) move the eval off the v1 wall (d2=808 vs pst 1367), independent of arch.

Usage: python train_nnue.py [epochs] [batch] [corpus_path]
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import chess

from nnue_model import NNUENet, features, make_nnue_eval, NNUE_ACC_DIM, NNUE_HIDDEN
from gbdt_features import cp_from_tanh
import augment

DATA_V2 = "data/distill_v2.jsonl"       # Stage-1 enriched dict corpus (preferred)
DATA_TANH = "data/distill_sf.jsonl"     # legacy fallback ([fen, tanh_value] lists)
OUT = "models/nnue.pt"
CP_CLIP = 2000.0


def load_rows(path):
    """Load label rows as dicts {fen, value(tanh), ...}. Tolerates dict schema and legacy [fen, v]."""
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
                rows.append({"fen": obj[0], "value": obj[1]})   # legacy tanh value
    return rows


def rows_to_xy(rows, do_augment):
    """(feature-index lists, cp targets) from label rows; optionally add mirror/color-flip variants.

    Augmentation transforms the whole row (augment.py handles the value sign/frame), so converting each
    variant's tanh 'value' through cp_from_tanh yields the correctly-transformed cp target.
    """
    idxs, ys = [], []
    for r in rows:
        variants = [r]
        if do_augment:
            m = augment.mirror(r)          # None when the position has castling rights (not equivalent)
            if m is not None:
                variants.append(m)
            variants.append(augment.color_flip(r))
        for v in variants:
            cp = max(-CP_CLIP, min(CP_CLIP, cp_from_tanh(float(v["value"]))))
            idxs.append(features(chess.Board(v["fen"])))
            ys.append(cp)
    return idxs, np.array(ys, dtype=np.float32)


def batch_tensors(idx_lists, dev):
    """Flatten a list of feature-index lists into (feats, offsets) for EmbeddingBag."""
    offsets, flat = [], []
    for lst in idx_lists:
        offsets.append(len(flat))
        flat.extend(lst if lst else [0])   # empty -> a harmless index; near-empty boards only
    return (torch.tensor(flat, dtype=torch.long, device=dev),
            torch.tensor(offsets, dtype=torch.long, device=dev))


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    corpus = sys.argv[3] if len(sys.argv) > 3 else (DATA_V2 if os.path.exists(DATA_V2) else DATA_TANH)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = time.time()
    rows = load_rows(corpus)
    n = len(rows)
    print(f"loaded {n} rows from {corpus} in {time.time()-t:.0f}s")

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    nval = min(4000, n // 10)
    val_rows = [rows[i] for i in perm[:nval]]
    tr_rows = [rows[i] for i in perm[nval:]]

    idxs_tr, y_tr = rows_to_xy(tr_rows, do_augment=True)     # train: ~3x via symmetries
    idxs_val, y_val = rows_to_xy(val_rows, do_augment=False)  # val: clean held-out
    print(f"train {len(y_tr)} (aug from {len(tr_rows)}) | val {len(y_val)} | "
          f"cp range [{y_tr.min():.0f},{y_tr.max():.0f}] mean|cp| {np.abs(y_tr).mean():.0f} | {dev}")

    net = NNUENet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    yv = torch.tensor(y_val, device=dev)
    vf, vo = batch_tensors(idxs_val, dev)
    ntr = len(y_tr)

    for ep in range(epochs):
        net.train()
        order = rng.permutation(ntr)
        losses = []
        for s in range(0, ntr, batch):
            bi = order[s:s + batch]
            feats, offs = batch_tensors([idxs_tr[j] for j in bi], dev)
            tgt = torch.tensor(y_tr[bi], device=dev)
            opt.zero_grad()
            loss = F.mse_loss(net(feats, offs), tgt)
            loss.backward(); opt.step()
            losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            vp = net(vf, vo)
            rmse = torch.sqrt(F.mse_loss(vp, yv)).item()
            sign = ((vp > 0) == (yv > 0)).float().mean().item()
        if ep % 2 == 0 or ep == epochs - 1:
            print(f"ep {ep:3d}  train_mse {np.mean(losses):.0f}  val_rmse {rmse:.0f}cp  "
                  f"sign_acc {sign:.3f}", flush=True)

    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "acc_dim": NNUE_ACC_DIM, "hidden": NNUE_HIDDEN}, OUT)
    print(f"saved {OUT}")

    # Sanity: a white queen up must read clearly positive (~+900), not compressed.
    ev = make_nnue_eval(net, dev)
    for name, fen in [("start", chess.STARTING_FEN),
                      ("white up a queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")]:
        print(f"  {name}: {ev(chess.Board(fen)):+.0f} cp")


if __name__ == "__main__":
    main()
