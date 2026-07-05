"""Distributed Advantage Actor-Critic (:DA2C:) self-improve loop — spec/self-play-leela.spec.md.

The POLICY-GRADIENT mode of the :Expert-iteration-cycle:, with the SOUND :Spearhead-profile: alpha-beta
search as the improvement operator (NOT the falsified sampled beam of mcts_chess.py, NOT the flat d1
net-minimax). A single shared-memory NNUE critic is updated Hogwild-style by N parallel worker
processes (Zai & Brown ch5, Listings 5.5-5.8), each running spearhead self-play with ONE persistent
`AlphaBetaEngine(phi_widen=True)` (tree reuse) and its OWN Adam over the shared params.

Value frame: everything is WHITE-ABSOLUTE in [-1,1] (value_targets' frame). The critic NNUENet emits
White-absolute centipawns; both it and the search value are squashed via tanh(cp / VALUE_SQUASH_CP)
so they live on the same [-1,1] scale as the terminal +/-1 reward (repo convention, puct_selfplay.py).

Per move the spearhead search returns the chosen move AND its backed-up value V_search(s) (info
score, converted to White-absolute via value_targets.to_white then squashed). Per N-step segment
(DA2C_NSTEP) each worker takes ONE async gradient step on the shared model:

    CRITIC: (V_net(s) - target)^2,  target = lambda_return(rewards, boot=V_search[1:], dones, lam)
            -- the search-bootstrapped lambda-return; the self-improve bootstrap regresses the raw
            critic toward its own deeper SOUND search value.
    ACTOR : -logpi(chosen|s) * advantage.detach(),  advantage = V_search(s.chosen) - V_net(s)
            pi(a|s) = softmax over afterstate values V_net(s.a) of legal moves. There is no
            move-encoding head: the policy is defined by the critic's afterstate evaluations, so the
            log-softmax over afterstate values carries the actor gradient while the value in the
            advantage is detached.

    loss = actor_loss + DA2C_CLC * critic_loss     (DA2C_CLC down-weights the critic so the actor
                                                    learns faster -- the tunable balance, per spec)

TWO-PLAYER SIGN (resolved spec ambiguity): the critic is White-absolute, but a policy-gradient must
reinforce moves from the MOVER's perspective, or Black's good moves (which LOWER the White-absolute
value) would be pushed the wrong way. So the actor's afterstate softmax and its advantage are taken
in the side-to-move frame (negamax sign via value_targets.to_stm/to_white -- the single source of
truth for the flip). The critic loss stays White-absolute (its target, the lambda_return, is).

:Loss-trace: actor_loss and critic_loss are traced SEPARATELY per worker to models/da2c_trace_<wid>.jsonl
(they are adversarial / chaotic by design, ch5 fig 5.13 -- never combined). Alongside them the trace
carries mean-game-length rows (ch5 fig 5.14/5.15) and, when the periodic :Ladder: runs (worker 0
only), an Elo row -- the RISING per-iteration ladder/Elo is the real :Amplification-experiment: verdict.

CLI: python da2c.py [total_updates] [workers]
"""
import os
import sys
import json
import math
import random

import chess
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from nnue_model import (NNUENet, load_nnue, features, make_nnue_eval,
                        NNUE_ACC_DIM, NNUE_HIDDEN)
from engine import AlphaBetaEngine
from value_targets import lambda_return, to_white


# --- constants (all env-overridable) --------------------------------------------------------
def _envi(name, default):
    return int(os.environ.get(name, default))


def _envf(name, default):
    return float(os.environ.get(name, default))


DA2C_WORKERS = _envi("DA2C_WORKERS", min(6, max(1, (os.cpu_count() or 2) - 2)))
DA2C_NSTEP   = _envi("DA2C_NSTEP", 10)       # bootstrap horizon / gradient-step granularity
DA2C_LR      = _envf("DA2C_LR", 1e-3)
DA2C_CLC     = _envf("DA2C_CLC", 0.1)        # actor/critic balance -- TUNABLE (spec), not fixed
DA2C_LAM     = _envf("DA2C_LAM", 0.8)        # lambda-return mixing (search-bootstrapped)
DA2C_GAMMA   = _envf("DA2C_GAMMA", 1.0)      # undiscounted outcome (chess)
DA2C_TEMP    = _envf("DA2C_TEMP", 0.25)      # actor softmax temperature over afterstate values
DA2C_TIME    = _envf("DA2C_TIME", 0.10)      # per-move spearhead time limit (small -- parallelised)
DA2C_DEPTH   = _envi("DA2C_DEPTH", 4)        # spearhead depth cap (:Full-width-floor: shaft + tip)
GAME_CAP     = _envi("DA2C_GAME_CAP", 120)   # ply cap -> truncation-to-draw
LADDER_EVERY = _envi("DA2C_LADDER_EVERY", 50)  # global-update milestone for the cheap ladder
LADDER_GAMES = _envi("DA2C_LADDER_GAMES", 6)
VALUE_SQUASH_CP = _envf("DA2C_SQUASH_CP", 400.0)  # cp scale for tanh (repo convention)

MODEL_PATH = "models/nnue.pt"
SAVE_PATH  = os.environ.get("DA2C_OUT", MODEL_PATH)  # self-improve target; override to protect nnue.pt


# --- value helpers (White-absolute, squashed to [-1,1]) -------------------------------------
def terminal_white_value(board):
    """White-absolute outcome of a game-over board: mated side loses, else draw."""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    return 0.0


def net_values_white(net, boards):
    """Batched White-absolute squashed values in [-1,1] for a list of boards (grad-enabled).

    One EmbeddingBag forward over all boards (variable-length feature bags via offsets). A board
    with no non-king features (bare kings) contributes an empty bag -> zero accumulator -> tanh(0).
    Require: len(boards) >= 1. Guarantee: returns a 1-D tensor of length len(boards).
    """
    feats, offsets, cur = [], [], 0
    for b in boards:
        offsets.append(cur)
        idx = features(b)
        if idx:
            feats.extend(idx)
            cur += len(idx)
    if not feats:                                   # every board bare-king (degenerate); constants
        return torch.zeros(len(boards))
    feats_t = torch.tensor(feats, dtype=torch.long)
    off_t = torch.tensor(offsets, dtype=torch.long)
    cp = net(feats_t, off_t)                         # (B,) White-absolute centipawns
    return torch.tanh(cp / VALUE_SQUASH_CP)          # (B,) White-absolute value in (-1,1)


def search_value_white(eng, board):
    """Spearhead search -> (chosen_move, V_search White-absolute squashed to [-1,1]).

    info['score_cp'] is the side-to-move negamax root value; to_white flips it to White-absolute,
    then tanh squashes it onto the value frame (mate scores saturate to +/-1, which is correct).
    """
    move, info = eng.search(board)
    cp_white = to_white(info["score_cp"], board.turn == chess.WHITE)
    return move, math.tanh(cp_white / VALUE_SQUASH_CP)


# --- the DA2C gradient step (async / Hogwild on the shared critic) --------------------------
def gradient_step(net, opt, seg, counter, trace):
    """One advantage-actor-critic update on an N-step segment; returns the global update index.

    Require: every non-terminal transition in `seg` has its `boot` (= V_search(s_{t+1})) set, and the
    terminal transition (if any) is last with done=True and reward=White-absolute outcome.
    Guarantee: steps the shared params once; appends one {update,actor_loss,critic_loss} trace row.
    """
    rewards = [t["reward"] for t in seg]
    dones   = [t["done"] for t in seg]
    boot    = [t["boot"] for t in seg]
    targets = lambda_return(rewards, boot, dones, DA2C_LAM, gamma=DA2C_GAMMA)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    # CRITIC -- White-absolute V_net(s_t) regressed to the search-bootstrapped lambda-return.
    states = [t["state"] for t in seg]
    v_net = net_values_white(net, states)                 # (L,), grad
    critic_loss = ((v_net - targets_t) ** 2).mean()

    # ACTOR -- softmax over afterstate values (mover perspective); reinforce by spearhead advantage.
    v_net_detached = v_net.detach()
    actor_terms = []
    for i, t in enumerate(seg):
        board = t["state"]
        mover_white = (board.turn == chess.WHITE)
        legal = list(board.legal_moves)
        after = []
        for m in legal:
            board.push(m)
            after.append(board.copy(stack=False))
            board.pop()
        vals_white = net_values_white(net, after)          # (n_legal,), grad
        vals_stm = vals_white if mover_white else -vals_white   # to_stm: mover-perspective values
        logp = F.log_softmax(vals_stm / DA2C_TEMP, dim=0)
        ci = legal.index(t["move"])
        # advantage = V_search(s.chosen) - V_net(s), in the mover frame, detached.
        child_white = t["reward"] if t["done"] else t["boot"]   # V_search(s.chosen), White-absolute
        adv_white = child_white - float(v_net_detached[i])
        adv = adv_white if mover_white else -adv_white          # to_stm
        actor_terms.append(-logp[ci] * adv)
    actor_loss = torch.stack(actor_terms).mean()

    loss = actor_loss + DA2C_CLC * critic_loss
    opt.zero_grad()
    loss.backward()
    opt.step()

    al, cl = actor_loss.detach().item(), critic_loss.detach().item()
    with counter.get_lock():
        counter.value += 1
        k = counter.value
    trace.write(json.dumps({"update": k, "actor_loss": al, "critic_loss": cl}) + "\n")
    trace.flush()
    return k


# --- cheap per-iteration :Ladder: (worker 0 only) -------------------------------------------
def run_ladder(net, k, trace):
    """Place the shared critic's ACTUAL agent (spearhead search) on the ladder vs random +
    heuristic-1ply, convert the score to an Elo estimate, and log it. Kept cheap (few games, a
    shallower/faster engine) -- the RISING curve across milestones is the amplification verdict."""
    from measure_ladder import play, random_mover, heuristic_mover, adj_pst, elo_diff
    eng = AlphaBetaEngine(eval_fn=make_nnue_eval(net), phi_widen=True,
                          time_limit=max(0.02, DA2C_TIME / 2.0),
                          max_depth=max(2, DA2C_DEPTH - 1))
    net_move = lambda b: eng.search(b)[0]
    rW, rD, rL = play(net_move, random_mover, LADDER_GAMES, adj_pst)
    hW, hD, hL = play(net_move, heuristic_mover, LADDER_GAMES, adj_pst)
    rs = (rW + 0.5 * rD) / max(1, rW + rD + rL)
    hs = (hW + 0.5 * hD) / max(1, hW + hD + hL)
    elo = elo_diff(hs)                                        # heuristic-1ply is the milestone rung
    trace.write(json.dumps({"update": k, "elo": round(elo, 1), "ladder_score": round(hs, 3),
                            "vs_random": round(rs, 3), "vs_heuristic": round(hs, 3)}) + "\n")
    trace.flush()
    print(f"[ladder @update {k}] vs_random={rs:.2f}  vs_heuristic={hs:.2f}  elo~{elo:+.0f}", flush=True)


# --- worker: spearhead self-play + async DA2C updates ---------------------------------------
def worker_loop(wid, net, total_updates, counter):
    """One mp.Process: persistent spearhead engine (tree reuse) + own Adam over the SHARED critic.

    Plays self-play games, records (state, move, V_search) per ply, and every DA2C_NSTEP transitions
    (or at game end) takes one async gradient step. Stops once the shared update counter reaches
    total_updates. Writes its own trace shard (no cross-process file contention)."""
    seed = (os.getpid() * 2654435761 + wid) & 0x7FFFFFFF
    torch.manual_seed(seed)
    random.seed(seed)

    opt = torch.optim.Adam(net.parameters(), lr=DA2C_LR)
    eng = AlphaBetaEngine(eval_fn=make_nnue_eval(net), phi_widen=True,
                          time_limit=DA2C_TIME, max_depth=DA2C_DEPTH)
    os.makedirs("models", exist_ok=True)
    trace = open(f"models/da2c_trace_{wid}.jsonl", "a")
    next_ladder = LADDER_EVERY

    def maybe_ladder(k):
        nonlocal next_ladder
        if wid == 0 and k >= next_ladder:
            run_ladder(net, k, trace)
            next_ladder += LADDER_EVERY

    while counter.value < total_updates:
        board = chess.Board()
        seg = []
        ply = 0
        while True:
            over = board.is_game_over(claim_draw=True)
            if over or ply >= GAME_CAP:
                if seg:
                    z = terminal_white_value(board) if over else 0.0   # truncation -> draw
                    seg[-1]["reward"] = z
                    seg[-1]["done"] = True
                    maybe_ladder(gradient_step(net, opt, seg, counter, trace))
                trace.write(json.dumps({"game_len": ply}) + "\n")
                trace.flush()
                break

            move, v_s = search_value_white(eng, board)
            if seg:
                seg[-1]["boot"] = v_s                     # successor value of the previous transition
            seg.append({"state": board.copy(stack=False), "move": move,
                        "v_search": v_s, "reward": 0.0, "done": False, "boot": 0.0})
            board.push(move)
            ply += 1

            if len(seg) > DA2C_NSTEP:                     # flush a full N-step segment (tail boot set)
                flush_seg, seg = seg[:DA2C_NSTEP], seg[DA2C_NSTEP:]
                maybe_ladder(gradient_step(net, opt, flush_seg, counter, trace))
                if counter.value >= total_updates:
                    break
        if counter.value >= total_updates:
            break

    trace.close()


def main():
    total_updates = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else DA2C_WORKERS

    if os.path.exists(MODEL_PATH):
        net = load_nnue(MODEL_PATH, device="cpu")
        print(f"loaded critic {MODEL_PATH}")
    else:
        net = NNUENet()
        print("fresh NNUENet critic (no models/nnue.pt)")
    net.train()
    net.share_memory()                                   # Hogwild: all workers update these params

    counter = mp.Value("i", 0)
    print(f"DA2C: {workers} workers x {total_updates} updates | NSTEP={DA2C_NSTEP} LR={DA2C_LR} "
          f"CLC={DA2C_CLC} LAM={DA2C_LAM} | spearhead d{DA2C_DEPTH}/{DA2C_TIME}s | save->{SAVE_PATH}",
          flush=True)

    procs = []
    for wid in range(workers):
        p = mp.Process(target=worker_loop, args=(wid, net, total_updates, counter))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "acc_dim": NNUE_ACC_DIM, "hidden": NNUE_HIDDEN},
               SAVE_PATH)
    print(f"done: {counter.value} updates | critic -> {SAVE_PATH} | traces models/da2c_trace_*.jsonl",
          flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
