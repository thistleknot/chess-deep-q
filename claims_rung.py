"""Shim: forwards to `python -m chessdq.claims_rung` (spec/repo-layout.spec.md).
Keeps the documented `python claims_rung.py ...` invocation working verbatim."""
import os
import subprocess
import sys

sys.exit(subprocess.call(
    [sys.executable, "-m", "chessdq.claims_rung", *sys.argv[1:]],
    cwd=os.path.dirname(os.path.abspath(__file__))))
