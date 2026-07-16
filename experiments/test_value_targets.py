import sys as _sys, pathlib as _plib; _sys.path.insert(0, str(_plib.Path(__file__).resolve().parents[1]))  # repo-root shim (:Package-restructure:)
"""Contract tests for value_targets.py (spec/value-target.spec.md test reqs).

Pins the four invariants before any trainer consumes the module:
  1. lambda_return(lam=0) == td0 per step (== the neural_network.py:274 formula).
  2. lambda_return(lam=1) == mc_return (discounted terminal outcome).
  3. n-step caps at plies-to-terminal (large n collapses to Monte-Carlo).
  4. White-absolute <-> side-to-move frame conversion round-trips and flips for Black.

Run: python test_value_targets.py
"""

from chessdq import value_targets as vt

# A scripted 3-ply White-win trajectory, White-absolute.
REWARDS = [0.0, 0.0, 1.0]
BOOT = [0.2, 0.5, None]          # search values V(s1), V(s2); terminal has none
DONES = [False, False, True]


def approx(a, b, eps=1e-9):
    return abs(a - b) <= eps


def test_lambda0_equals_td0():
    for gamma in (1.0, 0.99):
        lam0 = vt.lambda_return(REWARDS, BOOT, DONES, lam=0.0, gamma=gamma)
        for t in range(len(REWARDS)):
            v_next = BOOT[t] if BOOT[t] is not None else 0.0
            expected = vt.td0(REWARDS[t], v_next, DONES[t], gamma=gamma)
            assert approx(lam0[t], expected), (t, gamma, lam0[t], expected)
    print("ok  lambda=0 == td0 (matches neural_network.py:274 formula)")


def test_lambda1_equals_mc():
    for gamma in (1.0, 0.9):
        lam1 = vt.lambda_return(REWARDS, BOOT, DONES, lam=1.0, gamma=gamma)
        mc = vt.mc_return(REWARDS, DONES, gamma=gamma)
        assert all(approx(a, b) for a, b in zip(lam1, mc)), (gamma, lam1, mc)
    print("ok  lambda=1 == mc_return")


def test_nstep_caps_at_terminal():
    # Large n must equal Monte-Carlo (terminal reached before n steps expire).
    big = vt.nstep_return(REWARDS, BOOT, DONES, n=99, gamma=1.0)
    mc = vt.mc_return(REWARDS, DONES, gamma=1.0)
    assert all(approx(a, b) for a, b in zip(big, mc)), (big, mc)
    # n=1 must equal td0 per step.
    one = vt.nstep_return(REWARDS, BOOT, DONES, n=1, gamma=1.0)
    for t in range(len(REWARDS)):
        v_next = BOOT[t] if BOOT[t] is not None else 0.0
        assert approx(one[t], vt.td0(REWARDS[t], v_next, DONES[t], 1.0)), (t, one[t])
    print("ok  n-step caps at terminal (n=99 == MC; n=1 == td0)")


def test_fallback_bootstrap():
    # Missing search value falls back to the provided target-net values.
    boot_missing = [None, None, None]
    fallback = [0.2, 0.5, 0.0]
    got = vt.lambda_return(REWARDS, boot_missing, DONES, lam=0.0, gamma=1.0,
                           fallback_values=fallback)
    expected = vt.lambda_return(REWARDS, [0.2, 0.5, None], DONES, lam=0.0, gamma=1.0)
    assert all(approx(a, b) for a, b in zip(got, expected)), (got, expected)
    print("ok  missing search value falls back to target-net V")


def test_frame_sign():
    assert approx(vt.to_stm(0.7, True), 0.7)
    assert approx(vt.to_stm(0.7, False), -0.7)
    for v, wtm in ((0.7, True), (0.7, False), (-0.3, False)):
        assert approx(vt.to_white(vt.to_stm(v, wtm), wtm), v)
    print("ok  White-absolute <-> side-to-move frame round-trips")


def test_retrace_default_off():
    assert vt.retrace_weights([0.5, 0.5], [0.4, 0.6]) is None
    w = vt.retrace_weights([0.6, 0.5], [0.3, 0.5], lam=1.0, enabled=True)
    assert approx(w[0], 1.0) and approx(w[1], 1.0), w   # clipped to 1
    print("ok  retrace off by default; clipped when enabled")


if __name__ == "__main__":
    test_lambda0_equals_td0()
    test_lambda1_equals_mc()
    test_nstep_caps_at_terminal()
    test_fallback_bootstrap()
    test_frame_sign()
    test_retrace_default_off()
    print("\nall value-target contract tests passed")
