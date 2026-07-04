"""Value targets for the off-policy anneal and self-play — spec/value-target.spec.md.

The chosen rule (see spec/rl-categorization): the value head learns a **search-bootstrapped
lambda-return**, NOT pure TD(0) and NOT pure Monte-Carlo(z). This is the AlphaZero/MuZero lineage:
the policy improves by MCTS visit distillation (expert iteration), and the value target is a
lambda-return whose n-step bootstraps come from the MCTS *search value* at each node (fallback: the
target-net V(s')). It is off-policy-safe by construction (tree-backup: the bootstrap is the search's
improved on-policy value), so no importance sampling is needed at our scale.

Frame: everything here is White-absolute in [-1,1] (the single frame shared with the net value head
and the negamax leaf convention). `to_stm` / `to_white` convert to/from side-to-move perspective and
are the single source of truth for that sign flip.

Special cases the tests pin:
    lambda_return with lam=0            == td0  (r + gamma*V(s'))       -> matches neural_network.py:274
    lambda_return with lam=1            == mc_return (discounted terminal outcome)
    n is always capped at plies-to-terminal (terminal positions collapse to Monte-Carlo).
"""

from typing import Sequence, List, Optional


# --- frame conversion (single source of truth, shared with the negamax convention) ----------
def to_stm(white_value, white_to_move):
    """White-absolute -> side-to-move-relative."""
    return white_value if white_to_move else -white_value


def to_white(stm_value, white_to_move):
    """Side-to-move-relative -> White-absolute."""
    return stm_value if white_to_move else -stm_value


# --- point targets --------------------------------------------------------------------------
def td0(reward, v_next, done, gamma=1.0):
    """One-step TD target: r + (1-done)*gamma*V(s'). The lam=0 / n=1 special case, and exactly
    the formula at neural_network.py:274 (with its gamma=0.99). White-absolute."""
    return reward + (0.0 if done else 1.0) * gamma * v_next


# --- trajectory targets ---------------------------------------------------------------------
# A trajectory is a time-ordered list of transitions for ONE game, White-absolute:
#   rewards[t]           reward received entering step t+1 (0 for non-terminal shaped-free play)
#   bootstrap_values[t]  value estimate of state s_{t+1} to bootstrap from (search value; may be None)
#   dones[t]             True if step t is terminal
# For pure game-outcome learning, rewards are all 0 except the terminal +/-1 (White-absolute).

def _bootstrap(bootstrap_values, t, fallback_values):
    v = bootstrap_values[t] if bootstrap_values is not None else None
    if v is None and fallback_values is not None:
        v = fallback_values[t]
    return 0.0 if v is None else v


def mc_return(rewards: Sequence[float], dones: Sequence[float], gamma=1.0) -> List[float]:
    """Discounted Monte-Carlo return to the end of the game for each step (lambda=1 limit)."""
    n = len(rewards)
    out = [0.0] * n
    g = 0.0
    for t in reversed(range(n)):
        # On a done step the return is just the terminal reward; otherwise discount the future.
        g = rewards[t] if dones[t] else rewards[t] + gamma * g
        out[t] = g
    return out


def nstep_return(rewards, bootstrap_values, dones, n, gamma=1.0, fallback_values=None) -> List[float]:
    """n-step return for each step, bootstrapping from bootstrap_values[t+n-1] unless a terminal is
    reached first (then it is a pure Monte-Carlo return to that terminal). White-absolute."""
    T = len(rewards)
    out = [0.0] * T
    for t in range(T):
        g = 0.0
        discount = 1.0
        k = t
        steps = 0
        terminated = False
        while steps < n and k < T:
            g += discount * rewards[k]
            discount *= gamma
            if dones[k]:
                terminated = True
                k += 1
                break
            k += 1
            steps += 1
        if not terminated:
            # bootstrap from V(s_{k}) == bootstrap_values[k-1] (value of the state after the last
            # counted reward). k-1 is the index whose bootstrap_values holds V(s_k).
            g += discount * _bootstrap(bootstrap_values, k - 1, fallback_values)
        out[t] = g
    return out


def lambda_return(rewards, bootstrap_values, dones, lam, gamma=1.0, fallback_values=None) -> List[float]:
    """Search-bootstrapped lambda-return G^lam for each step (the PRIMARY value target).

    Recursive TD(lambda) form (caps n at terminal automatically):
        G_t = r_t + gamma * [ (1-lam)*V(s_{t+1}) + lam*G_{t+1} ]   for non-terminal t
        G_t = r_t                                                   for terminal t
    lam=0 -> td0 (r + gamma*V(s')); lam=1 -> mc_return. bootstrap V comes from the search value
    (bootstrap_values), falling back to fallback_values (target-net V) then 0.0. White-absolute.
    """
    T = len(rewards)
    out = [0.0] * T
    g_next = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            g = rewards[t]
        else:
            v_next = _bootstrap(bootstrap_values, t, fallback_values)
            g = rewards[t] + gamma * ((1.0 - lam) * v_next + lam * g_next)
        out[t] = g
        g_next = g
    return out


# --- off-policy correction (deferred; flag-gated, default OFF) -------------------------------
def retrace_weights(target_pi: Sequence[float], behavior_pi: Sequence[float],
                    lam=1.0, enabled: bool = False) -> Optional[List[float]]:
    """Retrace(lambda) truncated-IS weights c_s = lam * min(1, pi_target/pi_behavior).

    Deferred by design: at our scale replay is tiny and near-on-policy, and the tree-backup
    search-value bootstrap already handles off-policyness. Returns None unless explicitly enabled,
    so callers apply no correction by default. Present so the interface exists when stale replay
    is later shown to hurt.
    """
    if not enabled:
        return None
    return [lam * min(1.0, (t / b if b > 0 else 1.0)) for t, b in zip(target_pi, behavior_pi)]
