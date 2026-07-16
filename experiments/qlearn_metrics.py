import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""SQLite metrics store for the Merge 2 Q-learning trainer (spec/q-learning.spec.md).

Decouples training from the dashboard: the detached trainer WRITES metrics here; the FastAPI server only
READS. WAL journal mode lets the reader see rows while the writer holds the DB. Schema follows the
mcts_chess.py precedent (a flat per-iteration metrics table) plus a runs table and a control table used to
signal a stop to a background run.

Preconditions: sqlite3 (stdlib). Failure modes: none critical — a fresh DB is created on first connect.
"""
import json
import os
import sqlite3
import time

DB_PATH = os.environ.get("QLEARN_DB", "models/qlearn.sqlite")


def connect(db_path=DB_PATH):
    """Open (creating if needed) the metrics DB in WAL mode so reader and writer coexist."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    _init(conn)
    return conn


def _init(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, started REAL, status TEXT, hparams_json TEXT, pid INTEGER)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS metrics (
        run_id TEXT, iter INTEGER, ts REAL, epsilon REAL, loss REAL,
        checkmate_rate REAL, wr_random REAL, wr_pst REAL, elo_cal REAL,
        buffer_size INTEGER, games INTEGER, wall_s REAL,
        PRIMARY KEY (run_id, iter))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS control (run_id TEXT PRIMARY KEY, cmd TEXT)""")
    conn.commit()


def start_run(conn, run_id, hparams, pid):
    conn.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                 (run_id, time.time(), "running", json.dumps(hparams), pid))
    conn.execute("INSERT OR REPLACE INTO control VALUES (?,?)", (run_id, "run"))
    conn.commit()


def set_status(conn, run_id, status):
    conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
    conn.commit()


def log_metric(conn, run_id, it, epsilon, loss, checkmate_rate, wr_random, wr_pst,
               elo_cal, buffer_size, games, wall_s):
    conn.execute("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (run_id, it, time.time(), epsilon, loss, checkmate_rate, wr_random, wr_pst,
                  elo_cal, buffer_size, games, wall_s))
    conn.commit()


def request_stop(conn, run_id):
    conn.execute("INSERT OR REPLACE INTO control VALUES (?,?)", (run_id, "stop"))
    conn.commit()


def stop_requested(conn, run_id):
    row = conn.execute("SELECT cmd FROM control WHERE run_id=?", (run_id,)).fetchone()
    return bool(row) and row["cmd"] == "stop"


def latest_run(conn):
    return conn.execute("SELECT * FROM runs ORDER BY started DESC LIMIT 1").fetchone()


def get_run(conn, run_id):
    return conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()


def get_metrics(conn, run_id, since_iter=-1):
    return conn.execute("SELECT * FROM metrics WHERE run_id=? AND iter>? ORDER BY iter",
                        (run_id, since_iter)).fetchall()
