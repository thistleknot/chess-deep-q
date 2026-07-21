"""Regression test for the fresh-clone "missing rsearch4" surprise (2026-07-21).

requirements.txt installs pure-Python deps only; rsearch4 (chessdq/agents.py :: make_agent
"champion") is a separately-built Rust extension (rsearch/, README "Building the native
search extension"). A user who only runs `pip install -r requirements.txt` and picks the
default CHAMPION agent must get an actionable error, not a bare ModuleNotFoundError.

This simulates "rsearch4 was never built" by blocking it at import time (no venv/network
needed — fast and deterministic) and asserts make_agent("champion", ...) raises a
ModuleNotFoundError whose message names the fix (maturin develop), rather than crashing
opaquely or leaving the menu's generic "unexpected error occurred" one-liner as the only signal.
"""
import sys
import importlib
import importlib.abc
import pytest


class _BlockRsearch4(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "rsearch4" or name.startswith("rsearch4."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


@pytest.fixture
def rsearch4_missing():
    sys.modules.pop("rsearch4", None)
    blocker = _BlockRsearch4()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def test_champion_agent_gives_actionable_error_without_rsearch4(rsearch4_missing, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import torch
    (tmp_path / "models").mkdir()
    ckpt = {
        "enc": "amap", "arch": "linear",
        "state_dict": {"head.weight": torch.zeros(1, 897), "head.bias": torch.zeros(1)},
    }
    torch.save(ckpt, tmp_path / "models" / "champion.pt")

    import chessdq.agents as agents

    with pytest.raises(ModuleNotFoundError) as exc_info:
        agents.make_agent("champion")

    msg = str(exc_info.value)
    assert "maturin develop" in msg
    assert "README.md" in msg
