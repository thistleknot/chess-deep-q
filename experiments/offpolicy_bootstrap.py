import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Off-policy bootstrap from the ~1672 alpha-beta engine (spec :Off-policy-handoff:, :Demo-share:=1).

The strong CPU engine generates frontier-quality trajectories; the resnet learns OFF-POLICY from
them — value target = the game OUTCOME z (committed RL knowledge, White-absolute), policy target =
the engine's cloned move (behavioural cloning of the strong player). This starts the net near the
engine's level (the fix for the d1 self-play failure: that bootstrapped from a WEAK net). Self-play
(selfplay.py) then refines beyond. Stockfish is only the :Ladder: gate.

Usage: python offpolicy_bootstrap.py [rounds] [games_per_round] [engine_sec_per_move]
"""
import random, sys, time
from collections import deque

import chess
import torch

from chessdq.resnet_model import ChessResNet, encode18, move_index
from chessdq.engine import AlphaBetaEngine, pst_eval
from chessdq.selfplay import _train, _ladder


def gen_game(engine, random_plies=4, cap=140):
    """Open with `random_plies` random moves (state-space coverage), then the engine plays both
    sides. Return (states, engine_move_idxs, value_targets=z) for the engine-played plies."""
    board = chess.Board()
    for _ in range(random_plies):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    states, pols, plies = [], [], 0
    while not board.is_game_over() and plies < cap:
        mv = engine.search(board)[0]
        if mv is None or mv not in board.legal_moves:
            break
        states.append(encode18(board))
        pols.append(move_index(mv))
        board.push(mv); plies += 1
    z = 0.0
    if board.is_checkmate():                         # side to move mated
        z = 1.0 if board.turn == chess.BLACK else -1.0
    return states, pols, [z] * len(states)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    tsec = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessResNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buffer = deque(maxlen=200000)
    engine = AlphaBetaEngine(eval_fn=pst_eval, time_limit=tsec)
    print(f"Off-policy bootstrap: ~1672 engine @ {tsec}s/move, {rounds} rounds x {games} games "
          f"(train on {dev})\n")
    for r in range(rounds):
        t0 = time.time()
        for _ in range(games):
            st, po, va = gen_game(engine)
            for s in zip(st, po, va):
                buffer.append(s)
        loss = _train(net, opt, buffer, dev, steps=300, batch=256, use_amp=(dev.type == "cuda"))
        lad = _ladder(net, dev, depth=1, games=12)
        print(f"round {r}: loss={loss:.3f}  vs_random={lad['random']['score']:.2f}  "
              f"vs_heuristic={lad['heuristic']['score']:.2f}  buf={len(buffer)} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        torch.save({"state_dict": net.state_dict()}, "models/tower_offpolicy.pt")
    print("saved models/tower_offpolicy.pt")


if __name__ == "__main__":
    main()
