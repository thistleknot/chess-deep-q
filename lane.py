"""Shim: forwards to `python -m chessdq.lane` (spec/repo-layout.spec.md).
Keeps the documented `python lane.py ...` invocation working verbatim."""
import os
import subprocess
import sys

sys.exit(subprocess.call(
    [sys.executable, "-m", "chessdq.lane", *sys.argv[1:]],
    cwd=os.path.dirname(os.path.abspath(__file__))))
