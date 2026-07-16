"""Shim: forwards to `python -m chessdq.qlearn` (spec/repo-layout.spec.md).
Keeps the documented `python qlearn.py ...` invocation working verbatim."""
import os
import subprocess
import sys

sys.exit(subprocess.call(
    [sys.executable, "-m", "chessdq.qlearn", *sys.argv[1:]],
    cwd=os.path.dirname(os.path.abspath(__file__))))
