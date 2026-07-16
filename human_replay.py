"""Shim: forwards to `python -m chessdq.human_replay` (spec/repo-layout.spec.md).
Keeps the documented `python human_replay.py ...` invocation working verbatim."""
import os
import subprocess
import sys

sys.exit(subprocess.call(
    [sys.executable, "-m", "chessdq.human_replay", *sys.argv[1:]],
    cwd=os.path.dirname(os.path.abspath(__file__))))
