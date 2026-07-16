"""Shim: forwards to `python -m chessdq.head2head` (spec/repo-layout.spec.md).
Keeps the documented `python head2head.py ...` invocation working verbatim."""
import os
import subprocess
import sys

sys.exit(subprocess.call(
    [sys.executable, "-m", "chessdq.head2head", *sys.argv[1:]],
    cwd=os.path.dirname(os.path.abspath(__file__))))
