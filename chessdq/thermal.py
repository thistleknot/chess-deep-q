""":Thermal-guard: — long-burn lanes cap themselves so hours of training/measurement
cannot peg 100% CPU and thermally reboot the laptop (operator 2026-07-15, after a
mid-claims-run reboot).

Cap = half the logical cores + below-normal priority (foreground use always wins);
torch thread pool matched to the cap. Windows children (e.g. Stockfish spawned by
measure_elo) inherit both the affinity mask and a below-normal priority class, so
one call at the entry point covers the whole process tree.

Override: CHESS_THERMAL_FRACTION (0..1, default 0.75 — operator-set; 1 disables the
affinity cap), CHESS_THERMAL_CORES ("6,7,8" — explicit core list, wins over fraction;
lets two lanes PARTITION the box instead of contending, which keeps the SF-clocked
claims lane's cores dedicated), CHESS_THERMAL_OFF=1 (skip entirely).
Failure modes: psutil missing or access denied -> warns and continues uncapped
(the guard is operational, never correctness-bearing).
"""
import os


def engage(fraction=0.75):
    """Cap this process (and future children) to fraction of the logical cores (or an
    explicit CHESS_THERMAL_CORES list) at below-normal priority. Call once at the top
    of any entry point expected to run longer than a few minutes."""
    if os.environ.get("CHESS_THERMAL_OFF") == "1":
        return
    try:
        fraction = float(os.environ.get("CHESS_THERMAL_FRACTION", fraction))
    except ValueError:
        pass
    try:
        import psutil
        p = psutil.Process()
        n = psutil.cpu_count(logical=True) or 1
        cores = os.environ.get("CHESS_THERMAL_CORES")
        if cores:
            lst = sorted({int(c) for c in cores.split(",") if c.strip() != ""})
            p.cpu_affinity(lst)
            k = len(lst)
        else:
            k = max(1, min(n, round(n * fraction)))
            if k < n:
                p.cpu_affinity(list(range(k)))
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 5)
        try:
            import torch
            torch.set_num_threads(k)
        except Exception:
            pass
        print(f"[thermal] capped to {k}/{n} cores, below-normal priority", flush=True)
    except Exception as e:
        print(f"[thermal] guard unavailable ({e}) - running uncapped", flush=True)
