"""Sensitivity simulations for power (reviewer point P1-1). Unlike power_sim3.py, behaviour is simulated
round by round in each participant's actual play order and demand sequence, so that it can depend on
time and on feedback. Departures from the baseline model:

  learning   first orders move towards the best order over the game: beta_t = beta + L (t/29)(1 - beta)
  fatigue    noise in first orders grows: sd_t = 0.3 sd (1 + F t/29)
  shock      after a round in which the participant's OWN realised cost was in the worst 20 per cent of the
             costs their order could have produced, the chance of full acceptance in the next round falls by
             S (moved to 'ignore'); same in all groups
  dropout    per-round hazard h_g * exp(G z - G^2/2), z = standardised running mean regret (people doing badly
             leave more often); h_g calibrated to about 10 per cent overall, optionally different in C

Assignment, analysis (completers, blocked randomisation test, fixed order H1 -> H2) as in power_sim3.py.
For drop-out scenarios the 'true' all-assigned difference (potential outcomes before drop-out) is also
recorded, to show the bias of the completer comparison.
Usage: python power_sensitivity.py out.json [reps] [seed]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game, analysis as an

R = game.ROUNDS
ORD = game.scenario_orders(R)
P_A = np.array([0.30, 0.50, 0.20])
N_BLOCKS = 85
H_BASE = 1 - 0.90 ** (1 / 30)

def simulate_version(n, delta, order, seq, rng, kappa=4.0, learn=0.0, fatigue=0.0, shock=0.0):
    """Returns per-person mean regret over normal rounds and the (n x 30) matrix of regrets in play order
    (NaN for hint rounds), for one version."""
    p = P_A + np.array([-delta, 0.0, delta])
    props = rng.dirichlet(kappa * p, size=n)
    beta = np.clip(rng.normal(0.45, 0.25, n), -0.2, 1.2)
    bad_prev = np.zeros(n, bool)
    reg_seq = np.full((n, 30), np.nan)
    for t, i in enumerate(ORD[order][:30]):
        r = R[i]
        b_t = beta + learn * (t / 29) * (1 - beta)
        eps = rng.normal(0, 0.3 * r['sd'] * (1 + fatigue * t / 29), n)
        if r['hint'] is None:
            q0 = r['mu'] + b_t * (r['a'] - r['mu']) + eps
        else:
            q0 = r['m_i'] + b_t * (r['qstar'] - r['m_i']) + eps
        pr = props.copy()
        if shock > 0:
            s = np.minimum(shock, pr[:, 2]) * bad_prev
            pr[:, 2] -= s; pr[:, 0] += s
        cum = np.cumsum(pr, axis=1)
        typ = (rng.random(n)[:, None] > cum).sum(axis=1)
        w = np.where(typ == 0, 0.0, np.where(typ == 2, 1.0, rng.beta(3, 3, n)))
        q1 = np.clip(np.rint(q0 + w * (r['a'] - q0)), 0, r['U']).astype(int)
        if r['hint'] is None:
            reg_seq[:, t] = r['regret'][q1]
        # own outcome feedback
        k, pk = game.demand_pmf(r['m_i'], r['s_i'])
        d = game.DEMAND[seq][i]
        c_real = game.cost(q1, d, r['cs'])
        c_all = game.cost(q1[:, None], k[None, :], r['cs'])
        bad_prev = (pk[None, :] * (c_all >= c_real[:, None] - 1e-12)).sum(1) <= 0.20
    return reg_seq

def one_study(rng, delta=0.12, b_share=0.0, learn=0.0, fatigue=0.0, shock=0.0, G=0.0, c_mult=1.0, P=1000):
    blocks = np.repeat(np.arange(N_BLOCKS), 6)
    group = np.tile([0, 0, 1, 1, 2, 2], N_BLOCKS)
    ver = np.array([game.version_of_block(b) for b in blocks])
    n = len(blocks); reg_seq = np.full((n, 30), np.nan)
    for (o, s) in {(0, 0), (0, 1), (1, 0), (1, 1)}:
        for g, d in ((0, 0.0), (1, delta * b_share), (2, delta)):
            idx = np.where((ver[:, 0] == o) & (ver[:, 1] == s) & (group == g))[0]
            reg_seq[idx] = simulate_version(len(idx), d, o, s, rng, learn=learn, fatigue=fatigue, shock=shock)
    full_mean = np.nanmean(reg_seq, axis=1)
    # drop-out
    keep = np.ones(n, bool)
    h = np.where(group == 2, H_BASE * c_mult, H_BASE)
    run_sum = np.zeros(n); run_n = np.zeros(n)
    for t in range(30):
        x = reg_seq[:, t]
        has = ~np.isnan(x); run_sum[has] += x[has]; run_n[has] += 1
        m = np.where(run_n > 0, run_sum / np.maximum(run_n, 1), np.nan)
        z = np.nan_to_num((m - np.nanmean(m)) / (np.nanstd(m) + 1e-9))
        haz = np.clip(h * np.exp(G * z - G * G / 2), 0, 1)        # mean hazard unchanged by G
        keep &= ~((rng.random(n) < haz) & keep)
    ps = []
    for comp in (0, 1):
        sel = keep & ((group == 2) | (group == comp))
        ps.append(an.block_perm_p(full_mean[sel], (group[sel] == 2).astype(int), blocks[sel], rng, P))
    est = full_mean[keep & (group == 0)].mean() - full_mean[keep & (group == 2)].mean()
    true = full_mean[group == 0].mean() - full_mean[group == 2].mean()
    drop_rate = {g: float(1 - keep[group == g].mean()) for g in (0, 1, 2)}
    return an.fixed_sequence(ps[0], ps[1]), est, true, drop_rate

SCENARIOS = [
    ('baseline (round-by-round code, no dynamics)', {}),
    ('learning over rounds (L = 0.4)', dict(learn=0.4)),
    ('fatigue (noise doubles by the end, F = 1)', dict(fatigue=1.0)),
    ('reliance falls after own bad outcome (S = 0.2)', dict(shock=0.2)),
    ('learning + fatigue + reliance shock', dict(learn=0.4, fatigue=1.0, shock=0.2)),
    ('drop-out depends on performance (G = 1), equal across groups', dict(G=1.0)),
    ('drop-out depends on performance, half as frequent in C', dict(G=1.0, c_mult=0.5)),
    ('drop-out depends on performance, twice as frequent in C', dict(G=1.0, c_mult=2.0)),
]

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'power_sensitivity.json'
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    rows = []
    for name, kw in SCENARIOS:
        res = [one_study(rng, **kw) for _ in range(reps)]
        rej = np.array([r[0] for r in res]); est = np.array([r[1] for r in res]); true = np.array([r[2] for r in res])
        dr = {g: float(np.mean([r[3][g] for r in res])) for g in (0, 1, 2)}
        row = dict(scenario=name, reps=reps, power_H1=float(rej[:, 0].mean()), power_H1_and_H2=float(rej[:, 1].mean()),
                   completer_diff_mean=float(est.mean()), all_assigned_diff_mean=float(true.mean()),
                   bias=float((est - true).mean()), dropout_A=dr[0], dropout_B=dr[1], dropout_C=dr[2])
        rows.append(row); print(row, flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
