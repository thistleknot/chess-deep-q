"""The experimental net-guided Fibonacci beam agent (models/tower.pth) — re-homed from the retired
`chess_ai.get_beam_move` (spec/entrypoint.spec.md :Selectable-agent: 'beam').

Experimental architecture path; the tower net is undertrained, so expect weak play. The strength knob
is the SEARCH BUDGET (depth, total_ops); the beam commits argmax (:Root-commit:).
"""
import os


def beam_mover(dev=None, depth=6, total_ops=140):
    """Return (label, move_fn). move_fn is None if models/tower.pth is missing."""
    import torch
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "models/tower.pth"
    if not os.path.exists(path):
        print(f"No {path} (residual tower) found — train it via run_stage1.py.")
        return "beam", None
    from chessdq.resnet_model import ChessResNet
    from chessdq.beam import NetBeam, schedule_for
    net = ChessResNet().to(dev)
    sd = torch.load(path, map_location=dev)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd); net.eval()

    def mv(board):
        return NetBeam(net, dev, schedule_for(depth, total_ops)).select(board)

    return f"beam(d{depth})", mv


def main():
    from chessdq.agents import play_loop
    label, mv = beam_mover()
    if mv is None:
        return
    human_white = (input("Play as white? (y/n, default y): ").strip().lower() or "y") == "y"
    play_loop(mv, label, human_white)


if __name__ == "__main__":
    main()
