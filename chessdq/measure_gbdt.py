"""Measure the GBDT eval inside engine.py's real alpha-beta — the chosen path.

Fast numpy features (no per-node torch), GBDT predict -> centipawns via the inverse of the training
target (cp = 400*atanh(v)), dropped into AlphaBetaEngine as eval_fn. Fixed-depth ladder vs SF@1320
with binomial CIs, comparable to the heuristic ladder (1285/1367/1672 at d1/d2/d3).

Usage: python measure_gbdt.py [games] [depths]   e.g. python measure_gbdt.py 30 1,2,3
"""
import sys, glob, math, time
import numpy as np
import chess, chess.engine
import xgboost as xgb

from chessdq.engine import AlphaBetaEngine

SF_ANCHOR = 1320
MODEL = "models/gbdt.json"


def features_np(board):
    """1152-vector identical to resnet_model.encode18 (18 planes flattened C-order), pure numpy."""
    f = np.zeros(18 * 64, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        ch = (piece.piece_type - 1) + (0 if piece.color else 6)
        f[ch * 64 + sq] = 1.0
    if board.turn:
        f[12 * 64:13 * 64] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        f[13 * 64:14 * 64] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        f[14 * 64:15 * 64] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        f[15 * 64:16 * 64] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        f[16 * 64:17 * 64] = 1.0
    if board.ep_square is not None:
        f[17 * 64 + board.ep_square] = 1.0
    return f


def _verify_features():
    from chessdq.resnet_model import encode18
    for fen in [chess.STARTING_FEN, "rnb1kbnr/pppppppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"]:
        b = chess.Board(fen)
        a = encode18(b).numpy().reshape(-1)
        if not np.array_equal(a, features_np(b)):
            raise SystemExit("features_np does NOT match encode18 — retrain/align before measuring.")
    print("features_np matches encode18 exactly.")


def make_eval(model):
    from chessdq.gbdt_features import features as gbf
    bst = xgb.Booster(); bst.load_model(model)
    bst.set_param({"device": "cpu"})   # per-node single-row predict MUST be CPU (GPU latency kills it)
    def ev(board):
        return float(bst.inplace_predict(gbf(board)[None, :])[0])   # already White-absolute centipawns
    return ev


def _elo(x):
    x = min(max(x, 1e-4), 1 - 1e-4)
    return SF_ANCHOR - 400 * math.log10(1 / x - 1)


def play_match(mover, games, sf_ms=50, cap=100):
    sf = chess.engine.SimpleEngine.popen_uci(glob.glob('engines/**/stockfish*.exe', recursive=True)[0])
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': SF_ANCHOR})
    lim = chess.engine.Limit(time=sf_ms / 1000.0)
    W = D = L = 0
    for g in range(games):
        nw = (g % 2 == 0); b = chess.Board(); plies = 0
        while not b.is_game_over() and plies < cap:
            mv = mover(b) if (b.turn == chess.WHITE) == nw else sf.play(b, lim).move
            if mv is None or mv not in b.legal_moves:
                break
            b.push(mv); plies += 1
        if b.is_checkmate():
            res = 1.0 if (b.turn == chess.BLACK) == nw else 0.0
        elif b.is_game_over():
            res = 0.5
        else:
            cp = sf.analyse(b, chess.engine.Limit(time=0.05))['score'].white().score(mate_score=100000)
            res = 0.5 if (cp is None or abs(cp) <= 150) else (1.0 if (cp > 150) == nw else 0.0)
        W += res == 1.0; D += res == 0.5; L += res == 0.0
        print(f"  game {g+1}/{games}: {'W' if nw else 'B'} {'win' if res==1 else 'draw' if res==0.5 else 'loss'}", flush=True)
    sf.quit()
    return int(W), int(D), int(L)


def report(name, W, D, L):
    n = W + D + L; s = (W + 0.5 * D) / n
    se = math.sqrt(max(s * (1 - s), 1e-9) / n)
    lo, hi = max(1e-4, s - 1.96 * se), min(1 - 1e-4, s + 1.96 * se)
    print(f"\n{name}: {W}W-{D}D-{L}L  score {s:.2f} (95% CI {lo:.2f}-{hi:.2f})  "
          f"Elo ~{_elo(s):.0f} (CI {_elo(lo):.0f}..{_elo(hi):.0f})")


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    depths = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1', '2', '3'])]
    ev = make_eval(MODEL)
    for d in depths:
        eng = AlphaBetaEngine(eval_fn=ev, time_limit=1e9, max_depth=d)
        print(f"[GBDT-eval in alpha-beta, fixed depth {d}]")
        t = time.time()
        W, D, L = play_match(lambda b, e=eng: e.search(b)[0], games)
        report(f'gbdt-alphabeta-d{d}', W, D, L)
        print(f"  [wall {time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
