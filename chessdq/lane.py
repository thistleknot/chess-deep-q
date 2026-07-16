""":Lane-registry: (spec/bullet-route.spec.md) — the ONE owner of background runs.

Born from the 2026-07-15 zombie incident: a pre-reboot wait-loop chain (triggered by a
process-name grep) revived, retrained shared checkpoints mid-tournament, and
contaminated a duel. This module replaces ad-hoc session commands + hand-set core
lists with a sqlite registry (stack default) that knows what runs, on which cores,
touching which files.

Commands:
  python lane.py run --tag T --cores 0,1,2 [--after ID_OR_TAG] [--inputs a.pt,b.pt]
                     [--outputs c.pt] [--log data/T.log] -- <command ...>
      Register + babysit a run: refuses on input/output/core conflicts with RUNNING
      rows (Require), waits for --after to reach 'done' (registry state, NEVER a
      process-name grep), sets CHESS_THERMAL_CORES, spawns, marks done/failed.
      Launch lane.py itself detached; it blocks while babysitting its child.
  python lane.py ls              — live runs (registry rows verified against live pids)
  python lane.py eta             — measured ETA per live run from its log's own pace
  python lane.py reap [--kill]   — mark rows whose pid died; flag (or --kill) any
                                   UNREGISTERED qlearn/head2head/claims_rung process

Failure modes: a dead lane.py babysitter leaves its row 'running' with a live child —
reap detects the dead babysitter pid and marks the row; conflicts are checked at
launch only (no file locks — single-operator box, cheap and sufficient).
"""
import argparse
import os
import re
import sqlite3
import subprocess
import sys
import time

DB = os.path.join("data", "lanes.db")
WATCHED = ("qlearn.py", "head2head.py", "claims_rung.py")


def _db():
    os.makedirs("data", exist_ok=True)
    c = sqlite3.connect(DB, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag TEXT NOT NULL, cmd TEXT NOT NULL, cores TEXT DEFAULT '',
        inputs TEXT DEFAULT '', outputs TEXT DEFAULT '', log TEXT DEFAULT '',
        status TEXT DEFAULT 'running', pid INTEGER, child_pid INTEGER,
        started REAL, finished REAL)""")
    c.commit()
    return c


def _alive(pid):
    try:
        import psutil
        return pid is not None and psutil.pid_exists(int(pid))
    except Exception:
        return False


def _running(c):
    return c.execute("SELECT id,tag,cmd,cores,inputs,outputs,log,pid,child_pid,started "
                     "FROM runs WHERE status='running'").fetchall()


def _conflicts(c, cores, files):
    """(Require) no RUNNING row may write files this run touches, nor share cores."""
    out = []
    my_cores = set(x for x in cores.split(",") if x != "")
    for rid, tag, _cmd, rcores, _rin, routs, *_ in _running(c):
        theirs_out = set(x for x in routs.split(",") if x)
        if theirs_out & files:
            out.append(f"run {rid} ({tag}) writes {sorted(theirs_out & files)}")
        their_cores = set(x for x in rcores.split(",") if x != "")
        if my_cores and their_cores and (my_cores & their_cores):
            out.append(f"run {rid} ({tag}) holds cores {sorted(my_cores & their_cores)}")
    return out


def cmd_run(args, extra):
    c = _db()
    files = set(x for x in (args.inputs + "," + args.outputs).split(",") if x)
    if args.after:
        # wait FIRST, guard after: a chained run's inputs are its predecessor's
        # outputs by design — once the predecessor is 'done' there is no conflict
        while True:
            row = c.execute("SELECT status FROM runs WHERE id=? OR tag=? "
                            "ORDER BY id DESC LIMIT 1",
                            (args.after if args.after.isdigit() else -1, args.after)
                            ).fetchone()
            if row is None:
                print(f"REFUSED: --after '{args.after}' not in registry"); sys.exit(2)
            if row[0] == "done":
                break
            if row[0] in ("failed", "reaped"):
                print(f"ABORT: --after '{args.after}' ended '{row[0]}'"); sys.exit(3)
            time.sleep(15)
            c = _db()
    reap(c, kill=False, quiet=True)                     # clear dead rows before guarding
    conf = _conflicts(c, args.cores, files)
    if conf:
        print("REFUSED (conflicts):\n  " + "\n  ".join(conf))
        sys.exit(2)
    env = dict(os.environ)
    if args.cores:
        env["CHESS_THERMAL_CORES"] = args.cores
    if extra[0] in ("python", "python.exe"):
        extra[0] = sys.executable          # pin child to THIS interpreter — bare
                                           # "python" resolves differently under
                                           # CreateProcess than in the caller's shell
    log = args.log or os.path.join("data", f"lane_{args.tag}.log")
    with open(log, "ab") as lf:
        child = subprocess.Popen(extra, env=env, stdout=lf, stderr=subprocess.STDOUT)
    c.execute("INSERT INTO runs(tag,cmd,cores,inputs,outputs,log,pid,child_pid,started) "
              "VALUES(?,?,?,?,?,?,?,?,?)",
              (args.tag, " ".join(extra), args.cores, args.inputs, args.outputs,
               log, os.getpid(), child.pid, time.time()))
    c.commit()
    rid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    print(f"lane {rid} ({args.tag}) started pid {child.pid} cores [{args.cores}] log {log}",
          flush=True)
    rc = child.wait()
    c = _db()
    c.execute("UPDATE runs SET status=?, finished=? WHERE id=?",
              ("done" if rc == 0 else "failed", time.time(), rid))
    c.commit()
    print(f"lane {rid} ({args.tag}) {'done' if rc == 0 else f'FAILED rc={rc}'}")
    sys.exit(rc)


def _eta_from_log(log, started):
    """Measured ETA: parse the run's own progress lines — never intuition.
    Understands 'game N/M ... Xs/game' (ladder) and duel block/shard lines."""
    try:
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            txt = fh.read()[-6000:]
    except OSError:
        return None
    m = list(re.finditer(r"game (\d+)/(\d+).*?([\d.]+)s/game", txt))
    if m:
        n, tot, spg = int(m[-1].group(1)), int(m[-1].group(2)), float(m[-1].group(3))
        return (tot - n) * spg
    m = list(re.finditer(r"block \d+: (\d+)g", txt))
    cap = re.search(r"cap (\d+)g", txt)
    if m and cap:
        done, tot = int(m[-1].group(1)), int(cap.group(1))
        rate = done / max(1.0, time.time() - started)
        return (tot - done) / max(rate, 1e-9)
    m = list(re.finditer(r"(\d+)/(\d+)", txt))
    if m:
        n, tot = int(m[-1].group(1)), int(m[-1].group(2))
        if 0 < n <= tot:
            rate = n / max(1.0, time.time() - started)
            return (tot - n) / max(rate, 1e-9)
    return None


def cmd_ls(_args):
    c = _db()
    rows = _running(c)
    if not rows:
        print("no live lanes")
        return
    for rid, tag, cmd, cores, _i, _o, log, pid, child, started in rows:
        state = "live" if _alive(child) else "STALE(pid dead - run reap)"
        print(f"{rid:>3} {tag:<20} cores[{cores}] {state:<8} "
              f"age {int(time.time() - started)}s  {cmd[:60]}")


def cmd_eta(_args):
    c = _db()
    rows = _running(c)
    if not rows:
        print("no live lanes")
        return
    for rid, tag, _cmd, _cores, _i, _o, log, _pid, child, started in rows:
        if not _alive(child):
            print(f"{rid:>3} {tag:<20} pid dead — run reap")
            continue
        eta = _eta_from_log(log, started)
        print(f"{rid:>3} {tag:<20} "
              + (f"~{int(eta // 60)}m{int(eta % 60):02d}s remaining (measured)"
                 if eta is not None else "no progress lines yet"))


def reap(c=None, kill=False, quiet=False):
    """Mark rows whose babysitter/child died; flag (or --kill) unregistered watched
    processes. The FIRST command after any reboot/session start (spec rule)."""
    c = c or _db()
    n = 0
    for rid, tag, _cmd, _cores, _i, _o, _log, pid, child, _s in _running(c):
        if not _alive(child) and not _alive(pid):
            c.execute("UPDATE runs SET status='reaped', finished=? WHERE id=?",
                      (time.time(), rid))
            n += 1
            if not quiet:
                print(f"reaped lane {rid} ({tag}) — pids dead")
    c.commit()
    try:
        import psutil
        known = {r[8] for r in _running(c)} | {r[7] for r in _running(c)}
        for p in psutil.process_iter(["pid", "cmdline"]):
            cl = " ".join(p.info["cmdline"] or [])
            if any(w in cl for w in WATCHED) and p.info["pid"] not in known \
                    and "lane.py" not in cl:
                if kill:
                    p.kill()
                    print(f"KILLED unregistered pid {p.info['pid']}: {cl[:80]}")
                elif not quiet:
                    print(f"WARN unregistered pid {p.info['pid']}: {cl[:80]}")
    except Exception as e:
        if not quiet:
            print(f"process scan unavailable ({e})")
    return n


def api_rows():
    """Registry rows + measured ETA for server.py /api/lanes (JSON-ready dicts)."""
    c = _db()
    out = []
    for rid, tag, cmd, cores, _i, _o, log, _pid, child, started in _running(c):
        alive = _alive(child)
        eta = _eta_from_log(log, started) if alive else None
        out.append({"id": rid, "tag": tag, "cores": cores, "cmd": cmd[:80],
                    "alive": alive, "age_s": int(time.time() - started),
                    "eta_s": int(eta) if eta is not None else None, "log": log})
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--cores", default="")
    r.add_argument("--after", default="")
    r.add_argument("--inputs", default="")
    r.add_argument("--outputs", default="")
    r.add_argument("--log", default="")
    sub.add_parser("ls")
    sub.add_parser("eta")
    rp = sub.add_parser("reap")
    rp.add_argument("--kill", action="store_true")
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    if args.mode == "run":
        if not extra:
            print("run needs a command after --")
            sys.exit(2)
        cmd_run(args, extra)
    elif args.mode == "ls":
        cmd_ls(args)
    elif args.mode == "eta":
        cmd_eta(args)
    elif args.mode == "reap":
        reap(kill=args.kill)


if __name__ == "__main__":
    main()
