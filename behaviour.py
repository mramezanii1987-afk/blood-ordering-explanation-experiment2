"""Assumed behavioural model (not data). Used for the power and payout simulations.

first order   normal rounds: q0 = mu + beta_i (a - mu) + eps
              hint rounds:   q0 = m_I + beta_i (q*_I - m_I) + eps   (hint used partially)
              eps ~ N(0, 0.3 sd)
beta_i ~ N(0.45, 0.25), clipped to [-0.2, 1.2]                        (pull-to-centre)
weight on advice: person propensities (ignore, partial, full) ~ Dirichlet(kappa * p_group);
              per round one type is drawn; partial weights ~ Beta(3, 3)
final order   q1 = round(q0 + w (a - q0)), clipped to the allowed range
final rounds (no recommendation): q = round(mu + beta_i (q* - mu) + eps)
Group A: p = (0.30, 0.50, 0.20). Group C: (0.30 - delta, 0.50, 0.20 + delta).
"""
import numpy as np
import game

P_A = np.array([0.30, 0.50, 0.20])
R = game.ROUNDS
NORMAL = [i for i, r in enumerate(R) if r['phase'] == 'main' and r['hint'] is None]

def simulate(n, delta, rng, kappa=4.0, rounds=None, return_q0=False):
    """Returns final orders q1 (n x len(rounds)) for the given round indices; with return_q0=True
    returns (q0, q1), where q0 is the first order (equal to q1 in final rounds)."""
    rounds = list(range(len(R))) if rounds is None else rounds
    p = P_A + np.array([-delta, 0.0, delta])
    props = rng.dirichlet(kappa * p, size=n)
    beta = np.clip(rng.normal(0.45, 0.25, n), -0.2, 1.2)
    out = np.empty((n, len(rounds)), dtype=int); out0 = np.empty((n, len(rounds)), dtype=int)
    cum = np.cumsum(props, axis=1)
    for j, i in enumerate(rounds):
        r = R[i]
        eps = rng.normal(0, 0.3 * r['sd'], n)
        if r['phase'] == 'final':
            q = r['mu'] + beta * (r['qstar'] - r['mu']) + eps
            q0 = q
        else:
            if r['hint'] is None:
                q0 = r['mu'] + beta * (r['a'] - r['mu']) + eps
            else:
                q0 = r['m_i'] + beta * (r['qstar'] - r['m_i']) + eps
            u = rng.random(n)
            typ = (u[:, None] > cum).sum(axis=1)
            w = np.where(typ == 0, 0.0, np.where(typ == 2, 1.0, rng.beta(3, 3, n)))
            q = q0 + w * (r['a'] - q0)
        out[:, j] = np.clip(np.rint(q), 0, r['U']).astype(int)
        out0[:, j] = np.clip(np.rint(q0), 0, r['U']).astype(int)
    return (out0, out) if return_q0 else out

def mean_regret(orders, rounds):
    return np.mean([R[i]['regret'][orders[:, j]] for j, i in enumerate(rounds)], axis=0)
