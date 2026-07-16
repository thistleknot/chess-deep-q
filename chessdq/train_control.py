"""The :Stage-controller: (spec/training-loop.spec.md) — the staged, Elo-gated training entry, the
home the DQN `chess_ai.py` used to hold.

Default :Train-mode: approach is Stage-3 batched-PUCT self-play — the demonstrated climber (reached
parity with heuristic-1ply; RL_FINDINGS). Stage-1 Stockfish distillation is a valid optional
warm-start; the off-policy and linear paths were tried and plateaued (kept as scripts, not defaults).
Each trainer honors the :Run-contract: (checkpoint + resume).
"""

APPROACHES = ("puct", "selfplay")


def train(approach="puct", iters=8, games=96, playouts=None):
    """Run a training approach; return the per-iteration :Ladder: curve. Default = PUCT self-play."""
    if approach == "puct":
        from chessdq.puct_selfplay import train_puct_selfplay, PUCT_PLAYOUTS
        return train_puct_selfplay(iters, games, playouts or PUCT_PLAYOUTS)
    if approach == "selfplay":
        import torch
        from chessdq.resnet_model import ChessResNet
        from chessdq.selfplay import expert_iteration
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = ChessResNet().to(dev)
        curve = expert_iteration(net, dev, iters, games)
        torch.save({"state_dict": net.state_dict()}, "models/tower_selfplay.pt")
        print("saved models/tower_selfplay.pt")
        return curve
    raise ValueError(f"unknown training approach '{approach}' (use one of {APPROACHES})")
