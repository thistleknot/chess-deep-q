import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))
"""Derive improved model: uncertainty-MCTS generation + native d5 labeling.

v2: labels come from rsearch4 native d5 minimax (same quality as existing cache),
NOT from MC game outcomes (which diluted the fit in v1).

Pipeline:
  1. Build ensemble, create hybrid agent
  2. Play 200 games vs varied opponents (same as v1 — reuse positions)
  3. Harvest unique novel FENs
  4. Label with rsearch4.Searcher.search(fen, 5) — native d5, parallel
  5. Merge with existing 65k cache, refit ridge
  6. Measure on standard anchor ladder (native d9)

Usage: python -m experiments.uncertainty_mcts_derive_v2 [games=200] [label_depth=5] [threads=6]
"""
import importlib
import os
import sys
import time

import numpy as np
import torch
import chess

from chessdq.corpus_gen import raw_weights
from chessdq.encoders import get
from chessdq.uncertainty_mcts import UncertaintyMCTSAgent, build_ensemble
from experiments.distill_linear import (
    CHAMPION, CACHE, featurize, fit_ridge, save_ckpt, label
)

_ENC_FN, _NIN = get("amap")
OUT_CKPT = "models/champion_umcts_v2.pt"
PLY_CAP = 100


def play_games_varied(agent, n_games, ply_cap=PLY_CAP):
    """Play hybrid agent vs varied opponents. Returns list of (fen, game_idx)
    and outcomes."""
    from chessdq.measure_ladder import random_mover, heuristic_mover
    positions = []
    outcomes = []
    opponents = [random_mover, heuristic_mover, random_mover]

    for g in range(n_games):
        board = chess.Board()
        game_fens = []
        plies = 0
        agent_white = (g % 2 == 0)
        opp = opponents[g % len(opponents)]
        while not board.is_game_over(claim_draw=True) and plies < ply_cap:
            game_fens.append(board.fen())
            if (board.turn == chess.WHITE) == agent_white:
                mv = agent(board)
            else:
                mv = opp(board)
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
            plies += 1
        if board.is_checkmate():
            z = -1.0 if board.turn == chess.WHITE else 1.0
        else:
            z = 0.0
        outcomes.append(z)
        for fen in game_fens:
            positions.append((fen, g))
        if (g + 1) % 50 == 0:
            decisive = sum(1 for o in outcomes if o != 0)
            print(f"  game {g+1}/{n_games}: {len(positions)} positions, "
                  f"{decisive}/{g+1} decisive", flush=True)
    return positions, outcomes


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    label_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    print("=" * 60)
    print("UNCERTAINTY-MCTS DERIVE v2 — native d5 labels")
    print("=" * 60)
    t0 = time.time()

    # 1. Build ensemble + agent
    print("\n[1] Building ensemble and hybrid agent...", flush=True)
    w, b = raw_weights(CHAMPION)
    src_meta = torch.load(CHAMPION, map_location="cpu")
    W_ens, B_ens = build_ensemble()
    agent = UncertaintyMCTSAgent(W_ens, B_ens, sigma_threshold=0.010,
                                  sims=32, tau=0.15, c_puct=1.5)

    # 2. Generate games
    print(f"\n[2] Playing {n_games} games (hybrid vs varied opponents)...", flush=True)
    t1 = time.time()
    positions, outcomes = play_games_varied(agent, n_games)
    decisive = sum(1 for o in outcomes if o != 0)
    print(f"  done in {time.time()-t1:.0f}s: {decisive}/{n_games} decisive, "
          f"{len(positions)} total positions", flush=True)

    # 3. Deduplicate, filter novel FENs
    print(f"\n[3] Deduplicating + filtering novel positions...", flush=True)
    z = np.load(CACHE, allow_pickle=True)
    cached_fens = set(z["fens"])
    seen = set()
    novel_fens = []
    for fen, _gidx in positions:
        key = " ".join(fen.split()[:4])
        if key not in seen and fen not in cached_fens:
            seen.add(key)
            if any(True for _ in chess.Board(fen).legal_moves):
                novel_fens.append(fen)
    print(f"  {len(novel_fens)} novel non-terminal positions (not in 65k cache)")

    # 4. Label with native d5 minimax (rsearch4)
    print(f"\n[4] Labeling {len(novel_fens)} positions at d{label_depth} "
          f"(native, {threads} threads)...", flush=True)
    t2 = time.time()
    new_y = label(novel_fens, w, b, label_depth, threads)
    print(f"  labeled in {time.time()-t2:.0f}s, range [{new_y.min():.3f}, {new_y.max():.3f}]")

    # 5. Merge + refit
    print(f"\n[5] Merging with existing cache and refitting...", flush=True)
    cached_fens_list = list(z["fens"])
    cached_y = z["y"]
    all_fens = cached_fens_list + novel_fens
    all_y = np.concatenate([cached_y, new_y])
    print(f"  merged corpus: {len(all_fens)} positions "
          f"({len(cached_fens_list)} cached + {len(novel_fens)} novel)")
    X_all = featurize(all_fens)
    w_new, b_new, rms = fit_ridge(X_all, all_y, 100.0)
    print(f"  fit RMS (atanh) = {rms:.4f}  ||w||={np.linalg.norm(w_new):.3f}")
    save_ckpt(w_new, b_new, OUT_CKPT, src_meta)
    print(f"  saved {OUT_CKPT}")

    # 6. Measure on standard ladder
    print(f"\n[6] Running standard anchor ladder (native d9, 100g)...", flush=True)
    print(f"  python -m experiments.anchor_ladder {OUT_CKPT} 9 0 umcts_v2")
    total = time.time() - t0
    print(f"\nGeneration + labeling + refit: {total/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
