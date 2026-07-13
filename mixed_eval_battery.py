"""Q8 mixed-eval battery (SME test, Merge 20 step 3) — measurement only, no training.

Question: at our scale, do truncated GREEDY ROLLOUTS expose eval holes the static value
head misses (original-AlphaGo mixed evaluator, Silver et al. 2016: neither pure form
dominated; λ=0.5 beat both)? Evaluators over the p7 champion net (pst-769 linear):

  λ=0   : V(x) static                                (pure value head)
  λ=1   : truncated rollout — both sides 1-ply greedy over V for ROLLOUT_PLIES,
          terminal outcome ±1 if reached, else V(leaf)  (pure rollout)
  λ=0.5 : mean of the two                              (mixed)

Scoring: sign agreement with a Stockfish verdict (movetime-limited) on a fixed battery
of quiet + tactical positions. SF is the measurement ANCHOR (lawful under the purity
law — never a training label). Verdict sign at |cp| >= CP_DEAD; mate scores = ±1.

Usage: python mixed_eval_battery.py   -> data/mixed_eval_battery.md
Failure modes: missing ckpt/stockfish -> SystemExit; illegal battery FEN -> ValueError.
"""
import glob
import os

import numpy as np
import torch
import chess
import chess.engine

os.environ.setdefault("QLEARN_ENC", "pst")
from qlearn import ValueNet, ENC_FN, NIN_ENC  # noqa: E402

CKPT = os.environ.get("BATTERY_CKPT", "models/qlearn_p7_best.pt")
ROLLOUT_PLIES = 8
CP_DEAD = 80          # |cp| below this = "balanced" class
SF_MOVETIME = 0.2

# battery: (label, FEN or SAN line from startpos) — varied by structure per quality gates
BATTERY = [
    ("quiet: startpos",        []),
    ("quiet: italian",         ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6"]),
    ("quiet: closed d4",       ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7"]),
    ("tactic: Qxf7# in 1",     ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"]),
    ("tactic: Qxe5+ fork",     ["e4", "e5", "Qh5", "g6"]),
    ("tactic: hanging queen",  ["e4", "e5", "Qg4", "d5", "Nf3"]),   # ...Bxg4 wins the queen
    ("tactic: back-rank Ra8#", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("endgame: KP win",        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"),
    ("endgame: R vs k",        "8/8/8/4k3/8/8/8/R3K3 w - - 0 1"),
]


def board_of(spec):
    if isinstance(spec, str):
        return chess.Board(spec)
    b = chess.Board()
    for san in spec:
        b.push_san(san)
    return b


def v_net(net, board):
    with torch.no_grad():
        return float(net(torch.from_numpy(ENC_FN(board)).unsqueeze(0)))


def greedy_move(net, board):
    """1-ply greedy over afterstate V (White maximizes, Black minimizes)."""
    best_mv, best = None, None
    sgn = 1.0 if board.turn == chess.WHITE else -1.0
    for mv in board.legal_moves:
        board.push(mv)
        v = 1e9 * sgn if board.is_checkmate() else v_net(net, board)
        board.pop()
        if best is None or sgn * v > sgn * best:
            best_mv, best = mv, v
    return best_mv


def rollout(net, board):
    b = board.copy()
    for _ in range(ROLLOUT_PLIES):
        if b.is_game_over():
            break
        b.push(greedy_move(net, b))
    if b.is_game_over():
        r = b.result()
        return 1.0 if r == "1-0" else -1.0 if r == "0-1" else 0.0
    return v_net(net, b)


def sf_sign(engine, board):
    info = engine.analyse(board, chess.engine.Limit(time=SF_MOVETIME))
    s = info["score"].white()
    if s.is_mate():
        return 1 if s.mate() > 0 else -1
    cp = s.score()
    return 0 if abs(cp) < CP_DEAD else (1 if cp > 0 else -1)


def sign(v, band=0.05):
    return 0 if abs(v) < band else (1 if v > 0 else -1)


def main():
    ck = torch.load(CKPT, map_location="cpu")
    assert ck.get("enc", "pst") == "pst" and ck.get("arch", "linear") == "linear", ck.keys()
    net = ValueNet("linear", 64, NIN_ENC)
    net.load_state_dict(ck["state_dict"]); net.eval()
    sfp = glob.glob("engines/**/stockfish*.exe", recursive=True)
    if not sfp:
        raise SystemExit("no stockfish binary under engines/ — battery needs the anchor")
    engine = chess.engine.SimpleEngine.popen_uci(sfp[0])
    rows, agree = [], {"lam0": 0, "lam05": 0, "lam1": 0}
    try:
        for label, spec in BATTERY:
            b = board_of(spec)
            truth = sf_sign(engine, b)
            v0 = v_net(net, b)
            v1 = rollout(net, b)
            v05 = 0.5 * (v0 + v1)
            for key, v in (("lam0", v0), ("lam05", v05), ("lam1", v1)):
                agree[key] += int(sign(v) == truth)
            rows.append((label, truth, v0, v05, v1))
            print(f"{label:26s} SF {truth:+d} | V {v0:+.3f} mix {v05:+.3f} roll {v1:+.3f}", flush=True)
    finally:
        engine.quit()
    n = len(rows)
    lines = ["# Q8 mixed-eval battery (Merge 20 step 3)", "",
             f"net {CKPT} | rollout {ROLLOUT_PLIES} plies 1-ply-greedy | SF anchor "
             f"{SF_MOVETIME}s, dead band {CP_DEAD}cp | sign-agreement / {n}", "",
             "| position | SF | λ=0 static | λ=0.5 mixed | λ=1 rollout |", "|---|---|---|---|---|"]
    lines += [f"| {la} | {t:+d} | {a:+.3f} | {m:+.3f} | {r:+.3f} |" for la, t, a, m, r in rows]
    lines += ["", f"agreement: λ=0 {agree['lam0']}/{n} | λ=0.5 {agree['lam05']}/{n} | "
                  f"λ=1 {agree['lam1']}/{n}"]
    os.makedirs("data", exist_ok=True)
    open("data/mixed_eval_battery.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"agreement: lam0 {agree['lam0']}/{n} | lam05 {agree['lam05']}/{n} | lam1 {agree['lam1']}/{n}")


if __name__ == "__main__":
    main()
