"""Train the NNUE eval (spec/nnue-eval.spec.md :NNUE-training:) — v1 BOOTSTRAP target.

Supervised MSE(V, SF centipawns) on the Stockfish-distilled labels already on disk (the α=1 phase;
the RL phase later swaps the target to self-play λ-returns = TD-Leaf). Device-agnostic: trains on
CUDA if available. Outputs models/nnue.pt for AlphaBetaEngine(eval_fn=make_nnue_eval(...)).

Usage: python train_nnue.py [epochs] [batch]
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
import chess

from nnue_model import NNUENet, features, make_nnue_eval, NNUE_ACC_DIM, NNUE_HIDDEN

DATA_CP = "data/distill_cp.jsonl"
DATA_TANH = "data/distill_sf.jsonl"
OUT = "models/nnue.pt"
CP_CLIP = 2000.0


def load():
    """Return (index_lists, cp_targets). Raw cp labels preferred; recover from tanh if absent."""
    raw = os.path.exists(DATA_CP)
    src = DATA_CP if raw else DATA_TANH
    print(f"loading {'RAW cp' if raw else 'tanh (recovered)'} labels from {src}")
    from gbdt_features import cp_from_tanh
    idxs, ys = [], []
    with open(src) as fh:
        for line in fh:
            try:
                fen, v = json.loads(line)[:2]
            except Exception:
                continue
            idxs.append(features(chess.Board(fen)))
            cp = max(-CP_CLIP, min(CP_CLIP, float(v))) if raw else cp_from_tanh(v)
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = time.time()
    idxs, y = load()
    n = len(y)
    print(f"{n} positions in {time.time()-t:.0f}s; cp range [{y.min():.0f},{y.max():.0f}] "
          f"mean|cp| {np.abs(y).mean():.0f}; training on {dev}")

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    nval = min(4000, n // 10)
    val_i, tr_i = perm[:nval], perm[nval:]

    net = NNUENet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    yv = torch.tensor(y[val_i], device=dev)
    vf, vo = batch_tensors([idxs[i] for i in val_i], dev)

    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr_i))
        losses = []
        for s in range(0, len(order), batch):
            bi = [tr_i[j] for j in order[s:s + batch]]
            feats, offs = batch_tensors([idxs[i] for i in bi], dev)
            tgt = torch.tensor(y[bi], device=dev)
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
