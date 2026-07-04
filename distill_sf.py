"""Fast, cumulative value distillation from Stockfish into the net's V(s).

Stockfish at a fixed shallow depth is both far stronger (~2200+) and far faster (~180 pos/s/engine)
than our Python engine, so it is the ideal distillation teacher for bootstrapping a strong eval
quickly. Several SF processes label random positions in parallel while the main thread trains the
net on the growing buffer. The labelled set is appended to disk (data/distill_sf.jsonl) so each run
ACCUMULATES data and continues climbing -- the intended "improvement per 5-min run" loop.

Classification (see spec/rl-categorization): OFFLINE, off-policy, supervised value regression /
knowledge distillation -- the Stage-1 warm start. Not RL; self-play TD refines on top later.

Target: tanh(sf_white_centipawns / 400) in [-1,1], White-absolute, matching the net's value head.

Usage: python distill_sf.py [minutes] [sf_depth] [num_labelers] [--fresh]
"""
import os, sys, glob, json, time, math, random, threading
import chess, chess.engine
import torch, torch.nn.functional as F

from legacy.neural_network import ChessQNetwork
from board_utils import board_to_tensor

DATA = "data/distill_sf.jsonl"
TARGET_CP_SCALE = 400.0
BUFFER_MAX = 200_000
MINIBATCH = 512


def sf_engine(path):
    eng = chess.engine.SimpleEngine.popen_uci(path)
    eng.configure({"Threads": 1, "Hash": 16})
    return eng


def gen_position(max_plies=44, capture_bias=0.5):
    b = chess.Board()
    for _ in range(random.randint(4, max_plies)):
        if b.is_game_over():
            break
        legal = list(b.legal_moves)
        caps = [m for m in legal if b.is_capture(m)]
        b.push(random.choice(caps) if (caps and random.random() < capture_bias) else random.choice(legal))
    return b


def sf_target(eng, board, depth):
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    cp = info["score"].white().score(mate_score=100000)
    return math.tanh((cp if cp is not None else 0) / TARGET_CP_SCALE)


def labeler(path, depth, stop_event, out_list, out_lock):
    eng = sf_engine(path)
    try:
        while not stop_event.is_set():
            b = gen_position()
            if b.is_game_over():
                continue
            try:
                t = sf_target(eng, b, depth)
            except Exception:
                continue
            with out_lock:
                out_list.append((b.fen(), t))
    finally:
        eng.quit()


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 5.0
    depth = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 8
    n_lab = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 4
    fresh = "--fresh" in sys.argv

    sf_path = glob.glob("engines/**/stockfish*.exe", recursive=True)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ChessQNetwork().to(device)
    if not fresh:
        try:
            ck = torch.load("models/chess_model.pth", map_location=device)
            net.load_state_dict(ck["q_network_state_dict"]); print("Loaded existing weights.")
        except Exception as e:
            print(f"fresh start ({e})")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Cumulative dataset: seed the training buffer from disk so runs build on each other.
    os.makedirs("data", exist_ok=True)
    Xbuf, ybuf = [], []
    if not fresh and os.path.exists(DATA):
        with open(DATA) as fh:
            for line in fh:
                try:
                    fen, t = json.loads(line)
                    Xbuf.append(board_to_tensor(chess.Board(fen))); ybuf.append(float(t))
                except Exception:
                    continue
        print(f"Seeded {len(Xbuf)} positions from {DATA}")

    # Held-out validation (labelled once, fixed).
    veng = sf_engine(sf_path)
    val_boards = [gen_position() for _ in range(400)]
    val_boards = [b for b in val_boards if not b.is_game_over()]
    val_x = torch.stack([board_to_tensor(b) for b in val_boards]).to(device)
    val_y = torch.tensor([sf_target(veng, b, depth) for b in val_boards], device=device)
    veng.quit()

    # Launch parallel labelers.
    new_samples, lock, stop = [], threading.Lock(), threading.Event()
    threads = [threading.Thread(target=labeler, args=(sf_path, depth, stop, new_samples, lock), daemon=True)
               for _ in range(n_lab)]
    for th in threads:
        th.start()

    budget = minutes * 60.0
    start = time.time()
    step, added_total = 0, 0
    fh = open(DATA, "a")
    print(f"Distilling SF depth-{depth} with {n_lab} labelers for {minutes:.1f} min on {device}...\n")
    try:
        while time.time() - start < budget:
            # Drain freshly labelled positions into the buffer and to disk.
            with lock:
                batch = new_samples[:]; new_samples.clear()
            for fen, t in batch:
                Xbuf.append(board_to_tensor(chess.Board(fen))); ybuf.append(t)
                fh.write(json.dumps([fen, t]) + "\n")
            added_total += len(batch)
            if len(Xbuf) > BUFFER_MAX:
                Xbuf = Xbuf[-BUFFER_MAX:]; ybuf = ybuf[-BUFFER_MAX:]

            if len(Xbuf) >= MINIBATCH:
                for _ in range(8):
                    idx = random.sample(range(len(Xbuf)), MINIBATCH)
                    xb = torch.stack([Xbuf[i] for i in idx]).to(device)
                    yb = torch.tensor([ybuf[i] for i in idx], dtype=torch.float32, device=device)
                    loss = F.mse_loss(net(xb).squeeze(1), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
            else:
                time.sleep(0.2)
            step += 1

            if step % 10 == 0:
                net.eval()
                with torch.no_grad():
                    vp = net(val_x).squeeze(1)
                    vmse = F.mse_loss(vp, val_y).item()
                    sign = ((vp > 0) == (val_y > 0)).float().mean().item()
                net.train()
                el = time.time() - start
                print(f"[{el:5.0f}s] step {step:4d} data {len(Xbuf):7d} (+{added_total}) "
                      f"val_mse {vmse:.4f} sign_acc {sign:.2f} ({added_total/el:.0f} lbl/s)")
    finally:
        stop.set(); fh.close()
        for th in threads:
            th.join(timeout=2)

    net.eval()
    try:
        ck = torch.load("models/chess_model.pth", map_location="cpu")
    except Exception:
        ck = {}
    sd = {k: v.cpu() for k, v in net.state_dict().items()}
    ck["q_network_state_dict"] = sd; ck["target_network_state_dict"] = sd
    ck["distill_sf"] = {"depth": depth, "data": len(Xbuf)}
    torch.save(ck, "models/chess_model.pth")
    print(f"\nDone. dataset now {len(Xbuf)} positions (+{added_total} this run). Saved model.")


if __name__ == "__main__":
    main()
