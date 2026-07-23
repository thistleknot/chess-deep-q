import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim
"""Lane-1 regression: the CHAMPION must play via the pure-Python fallback when rsearch4 is absent
(pip-only playability, no Rust toolchain). Blocks rsearch4 at import time and asserts
make_agent('champion') returns a working mover that produces a legal move — instead of raising.
Pairs with test_fresh_env_rsearch.py (which pins the actionable-error behavior for the native path)."""
import sys
import importlib
import importlib.abc

import chess


class _BlockRsearch4(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "rsearch4" or name.startswith("rsearch4."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


def test_champion_plays_without_rsearch4():
    sys.modules.pop("rsearch4", None)
    blocker = _BlockRsearch4()
    sys.meta_path.insert(0, blocker)
    try:
        import chessdq.agents as agents
        importlib.reload(agents)
        label, move_fn, value_fn = agents.make_agent("champion", engine_time=0.5)
        assert move_fn is not None, "champion fallback returned no mover"
        assert "python-fallback" in label, f"expected python-fallback label, got {label!r}"
        board = chess.Board()
        mv = move_fn(board)
        assert mv in board.legal_moves, f"illegal move {mv}"
        v = value_fn(board)
        assert -1.0 <= v <= 1.0, f"value out of range: {v}"
        print(f"PASS: python-fallback champion played {mv}, value {v:+.3f}, label={label!r}")
    finally:
        sys.meta_path.remove(blocker)


if __name__ == "__main__":
    test_champion_plays_without_rsearch4()
