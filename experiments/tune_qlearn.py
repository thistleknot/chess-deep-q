import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Optuna (TPE) tuning of Merge 2 Q-learning — objective = final SF-anchored Elo (spec :Elo-objective:).

Method (memory: hyperparam-tuning-method): GROUND the search space in literature/historical priors (means ±
spreads), use TPE (Bayesian, not grid) seeded at those means, and run enough Elo games that the per-trial
effect beats the score's sdev. FOUR knobs are tuned; trials run the SAME regime as the console (frozen
generator per epoch, asymptotic τ/λ anneal, variance-adaptive λ):

  decay  γ  ~0.99   [0.95, 0.999]     discount / TD(λ) decay factor
  alpha  α  ~3e-3   [1e-3, 1e-2] log  SGD step
  lambda λ  ~0.8    [0.50, 0.95]      eligibility-trace TD<->MC dial (annealed base; variance-adapted)
  warmup    ~0.4    [0.10, 0.80]      τ anneal e-folding fraction (behavior explore->exploit)
  lambda_warmup ~0.8 [0.10, 1.20]     λ anneal e-folding fraction (targets MC->TD) — SPLIT from τ's:
                                      the two trust dials may need different speeds (V lags behavior)
  tau_floor ~0.05  [0.03, 0.25]       late-training temperature floor. HIGH floor sustains DECISIVE
                                      self-play (mates = the only reward); observed failure: as τ anneals
                                      the weak-but-sharp policy draws itself -> signal starvation

SAMPLE SIZE (games/epoch) and BATCH SIZE (games/gradient-cycle) are PASSED, not tuned — same settings the
console form takes; batch must be <= sample.

Usage: python tune_qlearn.py [n_trials] [sample_games] [max_epochs] [elo_games] [batch_games] [arch]
"""
import os
import re
import sys
import hashlib
import subprocess

import optuna

N_TRIALS    = int(sys.argv[1]) if len(sys.argv) > 1 else 21
EPOCH_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
MAX_EPOCHS  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ELO_GAMES   = int(sys.argv[4]) if len(sys.argv) > 4 else 20
BATCH_GAMES = int(sys.argv[5]) if len(sys.argv) > 5 else 20
ARCH        = sys.argv[6] if len(sys.argv) > 6 else "linear"   # linear | mlp — part of the study identity
HIDDEN      = int(sys.argv[7]) if len(sys.argv) > 7 else 64    # mlp hidden width — part of the identity
ALGO        = sys.argv[8] if len(sys.argv) > 8 else "q"        # q | ac — selects trainer + manifold
assert BATCH_GAMES <= EPOCH_GAMES, "batch size (games) must be <= sample size (games/epoch)"
REPLAY_TUNE = os.environ.get("QLEARN_REPLAY_TUNE", "0") == "1"  # Merge 15 :Replay-temperature:
# dim (replay_t, log): the admixture dial — new manifold knob, searched not hand-picked.
TRIV_TUNE = os.environ.get("QLEARN_TRIV_TUNE", "0") == "1"    # :Trivium-anneal: dims (c_start,
# c_end, b_srch, triv_warmup); a = 1-b-c(t). Search-space change -> new fingerprint.
LRWU_TUNE  = os.environ.get("QLEARN_LRWU_TUNE", "0") == "1"   # :LR-warmup: dim (spec/schedules)
PHASE_TUNE = os.environ.get("QLEARN_PHASE_TUNE", "0") == "1"  # :Phase-mix: dim
DISG_TUNE  = os.environ.get("QLEARN_DISG_TUNE", "0") == "1"   # :Dis-gamma: dim
ZCA_ENV  = os.environ.get("QLEARN_ZCA", "")                   # whitened training (E3 treatment);
# with kc+TDLEAF, trials seed from models/qlearn_wseed.pt (pristine whitened) instead
PARGEN_ENV = os.environ.get("QLEARN_PARGEN", "")              # Merge 9 native parallel generation —
# changes the BEHAVIOR/TARGET machinery (greedy+eps exact-min batches vs serial python) -> identity
ENC      = os.environ.get("QLEARN_ENC", "pst")                # pst | kc (Merge 6 donor features) —
# changes the INPUT manifold -> study identity; kc trials run FRESH nets (champion seed is 769-dim)
OPP      = os.environ.get("QLEARN_OPP", "self")               # self | graded — changes the TRAINING
# DATA DISTRIBUTION (spec :Graded-opponents:) -> part of the study identity, passed through to trials
TDLEAF   = os.environ.get("QLEARN_TDLEAF", "0") == "1"        # Merge 5 regime (spec/tdleaf.spec.md
# :Tuning:): leaf-target manifold, trials seeded from the champion; part of the study identity
SEARCH_D = os.environ.get("QLEARN_SEARCH_DEPTH", "2")
SEARCH_W = os.environ.get("QLEARN_SEARCH_WIDTH", "8")
TDLEAF_SEED = "models/qlearn_tdleaf_seed.pt"                  # bar-stripped champion copy
# mlp V is far more step-size sensitive (E1: collapsed at α=1e-3, the BOTTOM of the linear range);
# TDLeaf leaf targets are minimax-selected (larger magnitude/variance than 1-ply) — same lower range
ALPHA_RANGE = (1e-4, 3e-3) if (ARCH == "mlp" or TDLEAF) else (1e-3, 1e-2)
_ELO = re.compile(r"KILL-CHECK\s+elo\s+(-?[0-9.]+)")

# grounded priors: (mean) for TPE enqueue; ranges in objective()
PRIOR = {"decay": 0.99, "alpha": 3e-3, "lambda": 0.8, "warmup": 0.4, "lambda_warmup": 0.8,
         "tau_floor": 0.05}
if TRIV_TUNE:
    PRIOR = dict(PRIOR, c_start=0.374, c_end=0.143, b_srch=0.341, triv_warmup=0.481)
if REPLAY_TUNE:
    PRIOR = dict(PRIOR, replay_t=1.0, surprise_k=32.0)
if LRWU_TUNE:
    PRIOR = dict(PRIOR, lr_warmup=0.25)      # organ midpoints: trial 0 = proven parms + organ ON
if PHASE_TUNE:
    PRIOR = dict(PRIOR, phase_mix=0.5)
if DISG_TUNE:
    PRIOR = dict(PRIOR, dis_gamma=0.4)
if TDLEAF:
    # S&B §9.6 rule of thumb (skill 086_step-size): α ≈ 1/(τ·E[||x||²]) ≈ 1/(100 experiences ·
    # ~33 active features) ≈ 3e-4 — and measured: α=3e-3 (trial 0, self-play arm) still damaged
    # the champion (pooled 741 vs ~900 parity)
    PRIOR = dict(PRIOR, alpha=3e-4)
# Merge 3 actor-critic manifold (spec/actor-critic.spec.md): two-timescale steps, two trace decays,
# entropy bonus. Infrastructure passed fixed as ever.
PRIOR_AC = {"decay": 0.99, "alpha_w": 3e-3, "alpha_th": 1e-3, "lambda_w": 0.8, "lambda_th": 0.8,
            "beta": 0.005}

# RESUME-COMPATIBLE study identity: the study name fingerprints (search space, training regime, fixed
# protocol). UNCHANGED set -> same name -> Optuna RESUMES and TPE keeps learning from every prior run.
# Any change -> a NEW study starts alongside; old studies stay in the DB for reference (never deleted).
if ALGO == "ac":
    # v2 ranges: D9 measured the actor PINNED near-uniform (entropy 3.27->3.25 over 2k games) at
    # alpha_th<=1e-3 — the policy must be allowed to actually commit. Actor range up 10x, beta floor down.
    SPACE  = ("decay[0.95,0.999]|alpha_w[1e-4,1e-2]log|alpha_th[1e-4,1e-2]log"
              "|lambda_w[0.3,0.95]|lambda_th[0.3,0.95]|beta[3e-4,3e-2]log")
    REGIME = "ac-episodic-traces|onpolicy|entropy|pooled-objective|v1"
else:
    SPACE  = (f"decay[0.95,0.999]|alpha[{ALPHA_RANGE[0]:g},{ALPHA_RANGE[1]:g}]log|lambda[0.50,0.95]"
              f"|warmup[0.10,0.80]|lambda_warmup[0.10,1.20]|tau_floor[0.03,0.25]")
    REGIME = "frozen-gen|asym-anneal|adaptive-lam|elo-patience|pooled-objective|v2"  # bump when training
    # mechanics OR objective semantics change (v2: objective = run-pooled SF Elo + 10·vs_random tiebreak)
    if TDLEAF:
        REGIME += "|tdleaf-leafgen|resume-champ|v2"   # v2: :Backed-bootstrap: spec repair —
        # bootstrap max over depth-backed values only (S&B 015); v1 studies measured the
        # maximization-bias-inflated variant and are kept for reference, never resumed
    if TRIV_TUNE:
        REGIME += "|triv-anneal|v1"
        SPACE += "|c_start[0.1,0.6]|c_end[0,0.2]|b_srch[0.1,0.5]|triv_warmup[0.1,0.8]"
    if ZCA_ENV:
        REGIME += "|zca|v1"
    if PARGEN_ENV:
        REGIME += "|pargen|v1"
    if os.environ.get("QLEARN_PARGEN_SOFTMAX") == "1":
        REGIME += "|softmax-tau|v1"
    if os.environ.get("QLEARN_SURPRISE") == "1":
        REGIME += "|surprise|v1"
    if os.environ.get("QLEARN_GRPO") == "1":
        REGIME += "|grpo|v1"
    if os.environ.get("QLEARN_DDQN") == "1":
        REGIME += "|ddqn|v1"
    if REPLAY_TUNE:
        REGIME += "|replay-temp|v2"                       # v2: widened + surprise_k dim
        SPACE += "|replay_t[0.1,5]log|surprise_k[8,64]log"
    if os.environ.get("QLEARN_DECK"):
        REGIME += "|deck|v1"
    if LRWU_TUNE:
        REGIME += "|lr-warmup|v1"                        # :Schedules: organs (one per study)
        SPACE += "|lr_warmup[0.0,0.5]"
    if PHASE_TUNE:
        REGIME += "|phase-mix|v1"
        SPACE += "|phase_mix[0.0,1.0]"
    if DISG_TUNE:
        REGIME += "|dis-gamma|v1"
        SPACE += "|dis_gamma[0.0,0.8]"
    if os.environ.get("QLEARN_KC_FAITHFUL") == "0":
        REGIME += "|adam-replay|v1"                      # Merge 20 capacity re-audit: explicit
        # non-faithful regime (Adam + replay batches). Token added ONLY on the new value ("0")
        # so every prior study hash (faithful/unset launches) stays resume-compatible.
    if os.environ.get("QLEARN_CRELU") == "1":
        REGIME += "|crelu|v1"                            # clipped-ReLU hidden activation
    if os.environ.get("QLEARN_TUNE_SEED"):
        REGIME += "|seed-" + os.environ["QLEARN_TUNE_SEED"].split("/")[-1].replace(".pt", "")
ACTOR_ARCH = os.environ.get("QLEARN_ACTOR_ARCH", "linear")     # (ac) actor head arch — study identity
BEHAVIOR   = os.environ.get("QLEARN_BEHAVIOR", "softmax")      # (q) behavior policy — study identity
CURRICULUM = os.environ.get("QLEARN_CURRICULUM", "0")          # exploring-starts fraction (passthrough
# to trials); >0 changes the TRAINING DATA DISTRIBUTION -> must be part of the study identity
PROTO  = f"{ALGO}-s{EPOCH_GAMES}-b{BATCH_GAMES}-e{MAX_EPOCHS}-elo{ELO_GAMES}-{ARCH}" + \
         (f"-h{HIDDEN}" if ARCH == "mlp" else "") + \
         (f"-actor{ACTOR_ARCH}" if ACTOR_ARCH != "linear" else "") + \
         (f"-{BEHAVIOR}" if BEHAVIOR != "softmax" else "") + \
         (f"-cur{float(CURRICULUM):g}" if float(CURRICULUM) > 0 else "") + \
         (f"-tdleaf-d{SEARCH_D}w{SEARCH_W}" if TDLEAF else "") + \
         (f"-opp{OPP}" if OPP != "self" else "") + \
         (f"-enc{ENC}" if ENC != "pst" else "")
STUDY  = "qlearn_elo_" + hashlib.sha1(f"{SPACE}|{REGIME}|{PROTO}".encode()).hexdigest()[:8]


def run_trial(p, trial_no):
    """One trainer run (subprocess, fresh env) -> pooled-objective Elo from the KILL-CHECK line."""
    common = dict(QLEARN_GAMMA=f"{p['decay']:.4f}",
                  QLEARN_ARCH=ARCH, QLEARN_HIDDEN=str(HIDDEN),
                  QLEARN_BATCH_GAMES=str(BATCH_GAMES),
                  QLEARN_ELO_GAMES=str(ELO_GAMES),
                  QLEARN_EPOCH_ELO_GAMES=os.environ.get("QLEARN_EPOCH_ELO_GAMES", "12"),  # live SF games
                  # keep the console's Elo/score panels live DURING trials (~2 min/trial; env=0 = cheapest)
                  QLEARN_OPP=OPP, QLEARN_ENC=ENC,
                  QLEARN_SEED=str(trial_no),             # vary seed per trial
                  QLEARN_RESUME="0",                     # trials are ALWAYS hermetic (fresh net)
                  QLEARN_TAG=f"trial{trial_no}")
    if ALGO == "ac":
        script = "ac_learn.py"
        env = dict(os.environ, **common,
                   QLEARN_ALPHA_W=f"{p['alpha_w']:.6f}", QLEARN_ALPHA_TH=f"{p['alpha_th']:.6f}",
                   QLEARN_LAMBDA_W=f"{p['lambda_w']:.4f}", QLEARN_LAMBDA_TH=f"{p['lambda_th']:.4f}",
                   QLEARN_BETA=f"{p['beta']:.5f}",
                   QLEARN_CKPT="models/ac_optuna.pt")    # trial checkpoints never clobber console models
    else:
        script = "qlearn.py"
        env = dict(os.environ, **common,
                   QLEARN_ALPHA=f"{p['alpha']:.6f}", QLEARN_LAMBDA=f"{p['lambda']:.4f}",
                   QLEARN_WARMUP=f"{p['warmup']:.4f}", QLEARN_LAMBDA_WARMUP=f"{p['lambda_warmup']:.4f}",
                   QLEARN_TAU_FLOOR=f"{p['tau_floor']:.4f}",
                   QLEARN_FREEZE_EPOCH="1", QLEARN_ADAPTIVE_LAMBDA="1",
                   QLEARN_CKPT="models/qlearn_optuna.pt")
        if TDLEAF:
            # spec :Tuning:: seed EVERY trial from the bar-stripped champion (identical start ->
            # hermetic; fresh-net TDLeaf sits flat at the floor) and resume from it. Ckpt path is
            # per-regime so a self and a graded study can never poison each other's resume.
            # enc=kc: seed = the CHAMPION-EMBED (head.weight[:769]=m4-best, donor dims 0) —
            # validated by arm B-prime (spec :Backed-bootstrap: era, 862 @60g).
            import shutil
            ck = "models/qlearn_optuna.pt" if OPP == "self" else f"models/qlearn_optuna_{OPP}.pt"
            seed = TDLEAF_SEED
            if ENC != "pst":
                ck = ck.replace(".pt", f"_{ENC}.pt")
                seed = ("models/qlearn_wseed.pt" if ZCA_ENV
                        else TDLEAF_SEED.replace(".pt", f"_{ENC}.pt"))
            seed = os.environ.get("QLEARN_TUNE_SEED", seed)   # :Provenance: clean-seed override
            ck = os.environ.get("QLEARN_TUNE_CKPT") or ck     # per-study ckpt override: parallel
            # studies sharing OPP+ENC (e.g. two pst/graded arch arms) must not clobber one
            # another's trial checkpoints; path is NOT part of the study identity hash
            env["QLEARN_CKPT"] = ck
            shutil.copyfile(seed, ck)
            shutil.copyfile(seed, ck.replace(".pt", "_best.pt"))
            env.update(QLEARN_RESUME="1", QLEARN_TDLEAF="1",
                       QLEARN_SEARCH_DEPTH=SEARCH_D, QLEARN_SEARCH_WIDTH=SEARCH_W,
                       QLEARN_PROXY_GAMES=os.environ.get("QLEARN_PROXY_GAMES", "6"),
                       QLEARN_DEV=os.environ.get("QLEARN_DEV", "cpu"))
            for k in ("QLEARN_ZCA", "QLEARN_RSEARCH_DEPTH", "QLEARN_RSEARCH_MOD",
                      "QLEARN_KC_FAITHFUL", "QLEARN_RAMP", "QLEARN_OPP_REACH",
                      "QLEARN_CONFIRM", "QLEARN_PARGEN", "QLEARN_PARGEN_OPP_D",
                      "QLEARN_PARGEN_EPS", "QLEARN_PARGEN_THREADS",
                      "QLEARN_PARGEN_SOFTMAX", "QLEARN_STALE_REHEAT", "QLEARN_TUNE_SEED",
                      "QLEARN_SURPRISE", "QLEARN_GRPO", "QLEARN_DECK", "QLEARN_DDQN",
                      "QLEARN_DECK_MIX", "QLEARN_DECK_DECAY",
                      "QLEARN_LR_WARMUP", "QLEARN_PHASE_MIX", "QLEARN_DIS_GAMMA",
                      "QLEARN_CRELU"):
                if os.environ.get(k):
                    env[k] = os.environ[k]
            if REPLAY_TUNE:
                env["QLEARN_REPLAY_T"] = f"{p['replay_t']:.3f}"
                env["QLEARN_SURPRISE_K"] = f"{p['surprise_k']:.1f}"
            if LRWU_TUNE:
                env["QLEARN_LR_WARMUP"] = f"{p['lr_warmup']:.3f}"
            if PHASE_TUNE:
                env["QLEARN_PHASE_MIX"] = f"{p['phase_mix']:.3f}"
            if DISG_TUNE:
                env["QLEARN_DIS_GAMMA"] = f"{p['dis_gamma']:.3f}"
            if TRIV_TUNE:
                a_s = 1.0 - p["b_srch"] - p["c_start"]
                a_e = 1.0 - p["b_srch"] - p["c_end"]
                env.update(QLEARN_TRIVIUM=f"{a_s:.3f},{p['b_srch']:.3f},{p['c_start']:.3f}",
                           QLEARN_TRIVIUM_END=f"{a_e:.3f},{p['b_srch']:.3f},{p['c_end']:.3f}",
                           QLEARN_TRIVIUM_WARMUP=f"{p['triv_warmup']:.3f}")
    out = subprocess.run([sys.executable, script, str(EPOCH_GAMES), str(MAX_EPOCHS)],
                         env=env, capture_output=True, text=True, timeout=14400).stdout
    m = _ELO.search(out)
    return float(m.group(1)) if m else -999.0


def objective(trial):
    if ALGO == "ac":
        p = {
            "decay":     trial.suggest_float("decay", 0.95, 0.999),
            "alpha_w":   trial.suggest_float("alpha_w", 1e-4, 1e-2, log=True),
            "alpha_th":  trial.suggest_float("alpha_th", 1e-4, 1e-2, log=True),
            "lambda_w":  trial.suggest_float("lambda_w", 0.30, 0.95),
            "lambda_th": trial.suggest_float("lambda_th", 0.30, 0.95),
            "beta":      trial.suggest_float("beta", 3e-4, 3e-2, log=True),
        }
    else:
        p = {
            "decay":  trial.suggest_float("decay", 0.95, 0.999),
            "alpha":  trial.suggest_float("alpha", *ALPHA_RANGE, log=True),
            "lambda": trial.suggest_float("lambda", 0.50, 0.95),
            "warmup": trial.suggest_float("warmup", 0.10, 0.80),
            "lambda_warmup": trial.suggest_float("lambda_warmup", 0.10, 1.20),
            "tau_floor": trial.suggest_float("tau_floor", 0.03, 0.25),
        }
        if REPLAY_TUNE:
            p["replay_t"] = trial.suggest_float("replay_t", 0.1, 5.0, log=True)
            p["surprise_k"] = trial.suggest_float("surprise_k", 8.0, 64.0, log=True)
        if TRIV_TUNE:
            p["c_start"] = trial.suggest_float("c_start", 0.1, 0.6)
            p["c_end"] = trial.suggest_float("c_end", 0.0, 0.2)
            p["b_srch"] = trial.suggest_float("b_srch", 0.1, 0.5)
            p["triv_warmup"] = trial.suggest_float("triv_warmup", 0.1, 0.8)
        if LRWU_TUNE:
            p["lr_warmup"] = trial.suggest_float("lr_warmup", 0.0, 0.5)
        if PHASE_TUNE:
            p["phase_mix"] = trial.suggest_float("phase_mix", 0.0, 1.0)
        if DISG_TUNE:
            p["dis_gamma"] = trial.suggest_float("dis_gamma", 0.0, 0.8)
    elo = run_trial(p, trial.number)
    print(f"trial {trial.number}: elo {elo:.0f}  {p}", flush=True)
    return elo


def main():
    os.makedirs("models", exist_ok=True)
    storage = "sqlite:///models/qlearn_optuna.db"
    # seed the sampler by the RESUME depth: a fixed seed replays the identical candidate sequence on
    # every rerun (observed: trials 10-14 duplicated trials 1-4's params); seeding by prior trial count
    # keeps runs deterministic per state while continuing the stream instead of repeating it.
    prior_n = 0
    try:
        prior_n = next((s.n_trials for s in optuna.get_all_study_summaries(storage)
                        if s.study_name == STUDY), 0)
    except Exception:
        pass
    study = optuna.create_study(
        study_name=STUDY, direction="maximize",
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=prior_n),
    )
    done = [t for t in study.trials if t.value is not None]
    if done:
        print(f"RESUMING {STUDY} ({PROTO}): {len(done)} prior trial(s), best {study.best_value:.0f} — "
              f"TPE continues from them", flush=True)
    else:
        print(f"new study {STUDY} ({PROTO})", flush=True)
        study.enqueue_trial(PRIOR_AC if ALGO == "ac" else PRIOR)   # start from grounded-prior means
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"\nbest elo {study.best_value:.0f} @ {study.best_params}")


if __name__ == "__main__":
    main()
