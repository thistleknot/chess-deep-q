"""Tests for the Stage-1 teacher-distillation label pipeline (spec teacher-distillation).

Fail-fast: plain asserts, no try/except swallowing. Engine-dependent tests SKIP cleanly
when no Stockfish binary is present; augmentation tests are pure and always run.

Covers:
  (a) :Quiet-filter: a tactical FEN (side to move wins a queen) is REJECTED; a calm FEN
      is ADMITTED.
  (b) soft :Distillation-label: policy sums to ~1.0 and puts most mass on the best move.
  (c) :Label-augmentation: mirror(mirror(row)) == row; color_flip negates value and
      reverses wdl (and is itself an involution).
  (d) MP smoke: multiprocess labelling throughput exceeds the retired 58 labels/s GIL
      baseline.
"""

import os
import sys
import glob
import json
import time
import math
import tempfile
import subprocess

import chess
import chess.engine

import labeler_sf as L
import augment as A

# Tactical: White Rd3 can capture the hanging black queen on d5 (wins a queen).
TACTICAL_FEN = "4k3/8/8/3q4/8/3R4/8/4K3 w - - 0 1"
# Calm: locked pawns, best move is a quiet king step (no capture/check, no swing).
CALM_FEN = "8/5k2/4p3/4P3/5K2/8/8/8 w - - 0 1"


def engine_available():
    return bool(glob.glob("engines/**/stockfish*.exe", recursive=True))


def open_engine():
    sf = glob.glob("engines/**/stockfish*.exe", recursive=True)[0]
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Hash": 16})
    return eng


def test_quiet_filter():
    """(a) tactical rejected, calm admitted."""
    if not engine_available():
        print("SKIP test_quiet_filter (no engine)")
        return
    eng = open_engine()
    limit = chess.engine.Limit(depth=L.TEACHER_DEPTH)

    bt = chess.Board(TACTICAL_FEN)
    _, ct = L.build_label(bt, eng.analyse(bt, limit, multipv=L.MULTIPV_K))
    assert L.is_quiet(bt, ct) is False, "tactical position must be rejected by quiet filter"

    bc = chess.Board(CALM_FEN)
    _, cc = L.build_label(bc, eng.analyse(bc, limit, multipv=L.MULTIPV_K))
    assert L.is_quiet(bc, cc) is True, "calm position must be admitted by quiet filter"

    eng.quit()
    print("PASS test_quiet_filter")


def test_soft_policy():
    """(b) policy sums to ~1.0 and concentrates mass on the best move."""
    if not engine_available():
        print("SKIP test_soft_policy (no engine)")
        return
    eng = open_engine()
    limit = chess.engine.Limit(depth=L.TEACHER_DEPTH)
    b = chess.Board(TACTICAL_FEN)
    row, cands = L.build_label(b, eng.analyse(b, limit, multipv=L.MULTIPV_K))
    eng.quit()

    probs = [p for _, p in row["policy"]]
    assert abs(sum(probs) - 1.0) < 1e-6, f"policy must sum to 1.0, got {sum(probs)}"
    # best-first MultiPV -> index 0 is the best move and must hold the most mass.
    assert row["policy"][0][1] == max(probs), "best move must have the max policy mass"
    assert row["policy"][0][1] > 0.5, "dominant tactic must concentrate mass on the best move"
    assert row["policy"][0][0] == cands[0][0].uci()
    print("PASS test_soft_policy")


def test_augment_roundtrip():
    """(c) mirror involution; color_flip negates value + reverses wdl and is an involution."""
    row = {
        "fen": CALM_FEN,  # no castling rights -> horizontal mirror is a clean involution
        "value": 0.1234,
        "wdl": [0.3, 0.5, 0.2],
        "policy": [["f4g4", 0.6], ["e5e6", 0.4]],
    }

    # mirror is an involution on the full row.
    assert A.mirror(A.mirror(row)) == row, "mirror(mirror(row)) must equal row"

    # horizontal mirror leaves value/wdl untouched and file-flips policy moves.
    m = A.mirror(row)
    assert m["value"] == row["value"]
    assert m["wdl"] == row["wdl"]
    assert m["policy"][0][0] == "c4b4"  # f4->c4, g4->b4 (files a<->h)

    # color_flip: value sign-negated, wdl reversed w<->l.
    cf = A.color_flip(row)
    assert cf["value"] == -row["value"], "color_flip must negate value"
    assert cf["wdl"] == [0.2, 0.5, 0.3], "color_flip must reverse wdl (w<->l)"
    assert cf["policy"][0][0] == "f5g5"  # f4->f5, g4->g5 (ranks 1<->8)

    # color_flip is also an involution.
    assert A.color_flip(A.color_flip(row)) == row, "color_flip(color_flip(row)) must equal row"

    # None wdl tolerated.
    row2 = dict(row, wdl=None)
    assert A.color_flip(row2)["wdl"] is None
    print("PASS test_augment_roundtrip")


def test_mp_throughput_smoke():
    """(d) multiprocess labelling beats the retired 58 labels/s GIL baseline."""
    if not engine_available():
        print("SKIP test_mp_throughput_smoke (no engine)")
        return
    seconds, n_workers = 6.0, 4
    tmpdir = tempfile.mkdtemp(prefix="distill_smoke_")
    shards = [os.path.join(tmpdir, f"s{i}.jsonl") for i in range(n_workers)]

    start = time.time()
    procs = [
        subprocess.Popen(
            [sys.executable, "labeler_sf.py", sp, str(seconds), str(L.TEACHER_DEPTH), str(1000 + i)]
        )
        for i, sp in enumerate(shards)
    ]
    for p in procs:
        p.wait()
    elapsed = time.time() - start

    total = 0
    sample = None
    for sp in shards:
        with open(sp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if sample is None:
                    sample = json.loads(line)

    lps = total / elapsed
    # Sanity on the emitted dict schema.
    assert isinstance(sample, dict) and {"fen", "value", "policy"} <= set(sample), \
        f"rows must use the dict schema, got {sample}"
    assert -1.0 <= sample["value"] <= 1.0, "value must be tanh-bounded"
    assert lps > 58, f"MP throughput {lps:.1f} labels/s must exceed the 58 GIL baseline"

    for sp in shards:
        os.remove(sp)
    os.rmdir(tmpdir)
    print(f"PASS test_mp_throughput_smoke ({lps:.1f} labels/s, {n_workers} workers)")


if __name__ == "__main__":
    test_quiet_filter()
    test_soft_policy()
    test_augment_roundtrip()
    test_mp_throughput_smoke()
    print("\nAll tests passed.")
