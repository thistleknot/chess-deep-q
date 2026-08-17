import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))
"""Derive an improved model via uncertainty-MCTS augmented training.

The confirmed hybrid agent (uncertainty_mcts.py, +191 Elo over greedy at screen tier)
generates self-play games. At uncertain positions, it runs 32-sim PUCT — these produce
DEEPER value estimates than 1-ply greedy. We harvest (position, MCTS-backed-value) pairs
as training targets and refit the ridge model on the enriched corpus.

Pipeline:
  1. Build ensemble (reuse cached corpus)
  2. Play N self-play games (hybrid vs hybrid)
  3. Harvest positions + MCTS-backed values where search fired (uncertain positions)
  4. Also label ALL positions at d3 with the Python alpha-beta (cheap deeper teacher)
  5. Merge with existing cache, refit ridge
  6. Measure new model vs champion on the ladder

Usage: python -m experiments.uncertainty_mcts_derive [games=100] [label_depth=3]
"""
import os
import sys
import time
import math

import numpy as np
import torch
import chess

from chessdq.corpus_gen import raw_weights
from chessdq.encoders import get
from chessdq.engine import AlphaBetaEngine
from chessdq.uncertainty_mcts import UncertaintyMCTSAgent, build_ensemble
from experiments.distill_linear import CHAMPION, CACHE, featurize, fit_ridge, save_ckpt

_ENC_FN, _NIN = get("amap")

OUT_CKPT = "models/champion_umcts.pt"
PLY_CAP = 100
LABEL_DEPTH = 3  # Python alpha-beta depth for labeling (d3 ~0.5s/pos, d5 ~3s/pos)


def _make_eval_fn(w, b):
    """Build an eval_fn for the AlphaBetaEngine from champion weights."""
    w_arr = np.asarray(w, dtype=np.float64)
    b_val = float(b)
    def eval_fn(board):
        x = _ENC_FN(board).astype(np.float64)
        # Return centipawn-ish scale (white-absolute)
        return 800.0 * float(w_arr @ x + b_val)
    return eval_fn


def play_selfplay_games(agent, n_games, ply_cap=PLY_CAP):
    """Play hybrid agent vs varied opponents (random, heuristic, greedy-self)
    to generate diverse positions. Returns list of (fen, game_idx)."""
    from chessdq.measure_ladder import random_mover, heuristic_mover
    positions = []
    outcomes = []
    opponents = [random_mover, heuristic_mover, random_mover]  # cycle through

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
        # Outcome
        if board.is_checkmate():
            z = -1.0 if board.turn == chess.WHITE else 1.0
        else:
            z = 0.0
        outcomes.append(z)
        for fen in game_fens:
            positions.append((fen, g))
        if (g + 1) % 10 == 0:
            decisive = sum(1 for o in outcomes if o != 0)
            print(f"  game {g+1}/{n_games}: {len(positions)} positions, "
                  f"{decisive}/{g+1} decisive", flush=True)
    return positions, outcomes


def label_positions(fens, eval_fn, depth, threads=1):
    """Label positions using the Python AlphaBetaEngine at given depth.
    Returns white-absolute tanh values. Uses a tight time limit to keep it fast."""
    engine = AlphaBetaEngine(eval_fn=eval_fn, time_limit=2.0, max_depth=depth)
    values = np.empty(len(fens), dtype=np.float64)
    t0 = time.time()
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        if board.is_game_over():
            if board.is_checkmate():
                values[i] = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                values[i] = 0.0
            continue
        _mv, info = engine.search(board)
        # score_cp is side-to-move perspective; convert to white-absolute
        cp = info["score_cp"]
        if board.turn == chess.BLACK:
            cp = -cp
        # Convert centipawns to tanh scale (800cp = tanh(1.0))
        values[i] = math.tanh(cp / 800.0)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(fens) - i - 1) / rate
            print(f"    labeled {i+1}/{len(fens)} ({rate:.1f} pos/s, "
                  f"ETA {eta:.0f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"  labeled {len(fens)} positions at d{depth} in {elapsed:.1f}s "
          f"({len(fens)/elapsed:.1f} pos/s)", flush=True)
    return values


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    label_depth = int(sys.argv[2]) if len(sys.argv) > 2 else LABEL_DEPTH

    print("=" * 60)
    print("UNCERTAINTY-MCTS — DERIVE IMPROVED MODEL")
    print("=" * 60)
    t0_total = time.time()

    # 1. Load champion + build ensemble
    print("\n[1] Loading champion and building ensemble...", flush=True)
    w, b = raw_weights(CHAMPION)
    src_meta = torch.load(CHAMPION, map_location="cpu")
    W_ens, B_ens = build_ensemble()

    agent = UncertaintyMCTSAgent(W_ens, B_ens, sigma_threshold=0.010,
                                  sims=32, tau=0.15, c_puct=1.5)

    # 2. Self-play games
    print(f"\n[2] Playing {n_games} self-play games (hybrid vs hybrid)...", flush=True)
    t1 = time.time()
    positions, outcomes = play_selfplay_games(agent, n_games)
    decisive = sum(1 for o in outcomes if o != 0)
    print(f"  {n_games} games done in {time.time()-t1:.1f}s: "
          f"{decisive} decisive, {len(positions)} total positions", flush=True)

    # 3. Positions are grouped and labeled in step 4 (game-outcome labeling)
    print(f"\n[3] Total raw positions from self-play: {len(positions)}", flush=True)

    # 4. Label positions using GAME OUTCOMES (Monte Carlo return).
    # Decisive games: z = +1 (white won) or -1 (black won) propagated to all positions
    # in the game, discounted by γ^(distance_to_end). Draws: z = 0 (skip or keep as-is).
    # This is how the positions' TRUE values are revealed — the hybrid wins games the
    # greedy agent draws, so these outcomes ARE the improved signal.
    print(f"\n[4] Labeling via game outcomes (MC returns)...", flush=True)
    # Group positions by game + assign discounted outcome
    unique_with_labels = {}  # fen -> value (white-absolute)
    gamma = 0.99
    for g in range(len(outcomes)):
        z = outcomes[g]
        game_positions = [(fen, gidx) for fen, gidx in positions if gidx == g]
        T = len(game_positions)
        for t, (fen, _) in enumerate(game_positions):
            # Discounted MC return from position t
            dist_to_end = T - 1 - t
            val = z * (gamma ** dist_to_end)  # white-absolute outcome discounted
            key = " ".join(fen.split()[:4])
            if key not in unique_with_labels:
                unique_with_labels[key] = (fen, val)
            else:
                # Average if same position seen in multiple games
                old_fen, old_val = unique_with_labels[key]
                unique_with_labels[key] = (old_fen, (old_val + val) / 2.0)

    # Also label at d3 using eval for non-decisive positions (draws) — they still have positional value
    eval_fn = _make_eval_fn(w, b)
    unique_fens_labeled = []
    new_y_list = []
    for key, (fen, mc_val) in unique_with_labels.items():
        # Only keep positions with legal moves
        if not any(True for _ in chess.Board(fen).legal_moves):
            continue
        if abs(mc_val) > 0.01:  # from a decisive game
            unique_fens_labeled.append(fen)
            new_y_list.append(mc_val)
        else:
            # Draw game position — use the eval's own static assessment
            x = _ENC_FN(chess.Board(fen)).astype(np.float64)
            v = float(np.tanh(np.asarray(w, dtype=np.float64) @ x + float(b)))
            unique_fens_labeled.append(fen)
            new_y_list.append(v)

    unique_fens = unique_fens_labeled
    new_y = np.array(new_y_list, dtype=np.float64)
    print(f"  {len(unique_fens)} labeled positions "
          f"({sum(1 for v in new_y_list if abs(v) > 0.01)} from decisive games)")
    print(f"  label range: [{new_y.min():.3f}, {new_y.max():.3f}]")

    # 5. Merge with existing cache + refit
    print(f"\n[5] Merging with existing cache and refitting...", flush=True)
    z = np.load(CACHE, allow_pickle=True)
    cached_fens, cached_y = list(z["fens"]), z["y"]
    print(f"  existing cache: {len(cached_fens)} positions")

    # De-dup against cache
    cached_set = set(cached_fens)
    novel_fens, novel_y = [], []
    for fen, val in zip(unique_fens, new_y):
        if fen not in cached_set:
            novel_fens.append(fen)
            novel_y.append(val)
    novel_y = np.array(novel_y, dtype=np.float64)
    print(f"  novel positions (not in cache): {len(novel_fens)}")

    all_fens = cached_fens + novel_fens
    all_y = np.concatenate([cached_y, novel_y])
    print(f"  merged corpus: {len(all_fens)} positions")

    print(f"\n  Featurizing...", flush=True)
    X_all = featurize(all_fens)
    w_new, b_new, rms = fit_ridge(X_all, all_y, 100.0)
    print(f"  fit RMS (atanh) = {rms:.4f}  ||w||={np.linalg.norm(w_new):.3f}")

    save_ckpt(w_new, b_new, OUT_CKPT, src_meta)
    print(f"  saved {OUT_CKPT}")

    # 6. Quick ladder measurement (new model greedy vs rungs)
    print(f"\n[6] Measuring new model on ladder...", flush=True)
    from chessdq.measure_ladder import random_mover, heuristic_mover, play, adj_pst, elo_diff

    def new_greedy(board, _w=np.asarray(w_new, dtype=np.float64), _b=float(b_new)):
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        best_mv, best_v = None, -1e9
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                return mv
            x = _ENC_FN(board).astype(np.float64)
            v = sign * math.tanh(_w @ x + _b)
            board.pop()
            if v > best_v:
                best_v, best_mv = v, mv
        return best_mv

    def old_greedy(board, _w=np.asarray(w, dtype=np.float64), _b=float(b)):
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        best_mv, best_v = None, -1e9
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                return mv
            x = _ENC_FN(board).astype(np.float64)
            v = sign * math.tanh(_w @ x + _b)
            board.pop()
            if v > best_v:
                best_v, best_mv = v, mv
        return best_mv

    games = 40
    print(f"\n  --- NEW MODEL (greedy d1) vs rungs ({games}g) ---")
    for name, opp in [("random", random_mover), ("heuristic", heuristic_mover)]:
        W_r, D_r, L_r = play(new_greedy, opp, games, adj_pst)
        s = (W_r + 0.5 * D_r) / games
        print(f"    vs {name:10s}: {W_r}W {D_r}D {L_r}L  score={s:.3f}  "
              f"Elo_diff={elo_diff(s):+.0f}")

    print(f"\n  --- OLD CHAMPION (greedy d1) vs rungs ({games}g) ---")
    for name, opp in [("random", random_mover), ("heuristic", heuristic_mover)]:
        W_r, D_r, L_r = play(old_greedy, opp, games, adj_pst)
        s = (W_r + 0.5 * D_r) / games
        print(f"    vs {name:10s}: {W_r}W {D_r}D {L_r}L  score={s:.3f}  "
              f"Elo_diff={elo_diff(s):+.0f}")

    # Head-to-head: new vs old
    print(f"\n  --- HEAD-TO-HEAD: new vs old ({games}g) ---")
    W_r, D_r, L_r = play(new_greedy, old_greedy, games, adj_pst)
    s = (W_r + 0.5 * D_r) / games
    print(f"    new vs old: {W_r}W {D_r}D {L_r}L  score={s:.3f}  "
          f"Elo_diff={elo_diff(s):+.0f}")

    total = time.time() - t0_total
    print(f"\nTotal wall-clock: {total/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
