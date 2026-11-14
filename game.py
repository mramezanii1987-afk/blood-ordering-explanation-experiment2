"""Game definition: scenarios, hints, demand distributions, expected cost, best orders, regret.

Conventions (to be frozen in the pre-registration):
- Demand in a round is integer-valued. For a round with mean m and SD s (after any hint),
  P(D = k) = Phi((k + 0.5 - m) / s) - Phi((k - 0.5 - m) / s) for integers k in
  [max(0, floor(m - 3 s)), ceil(m + 3 s)], renormalised to sum to 1 (truncate and renormalise).
- Allowed orders are the integers 0 .. U, with U = ceil(mu_base + 5 * sd_base).
- Cost of order q at demand d: c_s * max(d - q, 0) + c_w * max(q - d, 0), with c_w = 1 point.
- The recommendation a is the integer order with the lowest expected cost under the BASE
  distribution (the computer never sees the hint). The best order q*(I) uses the distribution
  the participant faces (base or after the hint). Regret(q) = E[cost(q) | I] - E[cost(q*(I)) | I].
- Hints: level up/down shift the mean by round(2/3 * sd_base); uncertainty up/down multiply the SD
  by 1.5 / 0.5. The "usual range" shown to participants is mean +- 1 SD, rounded.
- Demand sequences (v7): two realised-demand sequences are used. They are chosen by a fixed rule that
  depends only on the scenarios: draw 5,000 candidate sequences with seed SEQ_SEED; for each, compute
  the total realised cost of three fixed strategies (best order; follow the recommendation in main
  rounds and order the average in final rounds; always order the average); express each total as its
  distance from the candidates' median divided by the interquartile range; take the two sequences with
  the smallest largest distance (ties broken by candidate index), among candidates in which 3 or 4 of
  the 18 normal rounds are unfavourable (the expected number is about 3.5; see below). No group label
  or behavioural data enter this rule.
- Versions (v7): 4 versions = 2 scenario orders x 2 demand sequences. Randomisation blocks of six are
  assigned to versions in rotation (block b -> version b mod 4), so 85 blocks give 22/21/21/21 blocks,
  i.e. 43 blocks (86 per group) in order 1 and 42 blocks (84 per group) in order 2.
- Unfavourable round (exploratory H4, v7): a normal main round is unfavourable in a demand sequence
  when the realised cost of following the recommendation is in the worst 20 per cent of the costs the
  recommendation could have produced in that round, i.e. P(cost(a, D) >= realised cost) <= 0.20 under
  the round's demand distribution.
"""
import numpy as np
from scipy.stats import norm

C_W = 1.0
GRID_MEANS = [60, 70, 80, 90, 100, 110, 120, 130, 140]
GRID_CV = [0.20, 0.275, 0.35]
GRID_CS = [3, 4]
HINT_TYPES = ['level_up', 'level_down', 'unc_up', 'unc_down']

def demand_pmf(m, s):
    lo = max(0, int(np.floor(m - 3 * s))); hi = int(np.ceil(m + 3 * s))
    k = np.arange(lo, hi + 1)
    p = norm.cdf((k + 0.5 - m) / s) - norm.cdf((k - 0.5 - m) / s)
    return k, p / p.sum()

def expected_cost(orders, k, p, c_s):
    short = np.clip(k[None, :] - orders[:, None], 0, None)
    waste = np.clip(orders[:, None] - k[None, :], 0, None)
    return (c_s * short + C_W * waste) @ p

def build_rounds(seed=20261112):
    r = np.random.default_rng(seed)
    combos = [(m, cv, cs) for m in GRID_MEANS for cv in GRID_CV for cs in GRID_CS]
    pick = r.choice(len(combos), 36, replace=False)
    scen = [combos[i] for i in pick]
    main, final = scen[:30], scen[30:]
    hint_pos = r.choice(30, 12, replace=False)
    hint_lab = np.repeat(HINT_TYPES, 3); r.shuffle(hint_lab)
    hints = [None] * 30
    for pos, h in zip(hint_pos, hint_lab):
        hints[pos] = h
    rounds = []
    for phase, lst, hl in (('main', main, hints), ('final', final, [None] * 6)):
        for (mu, cv, cs), h in zip(lst, hl):
            sd = mu * cv
            m_i, s_i = mu, sd
            if h == 'level_up': m_i = mu + round(2 / 3 * sd)
            if h == 'level_down': m_i = mu - round(2 / 3 * sd)
            if h == 'unc_up': s_i = sd * 1.5
            if h == 'unc_down': s_i = sd * 0.5
            U = int(np.ceil(mu + 5 * sd)); orders = np.arange(0, U + 1)
            kb, pb = demand_pmf(mu, sd); ki, pi = demand_pmf(m_i, s_i)
            ec_base = expected_cost(orders, kb, pb, cs)
            ec = expected_cost(orders, ki, pi, cs)
            a = int(orders[np.argmin(ec_base)]); qstar = int(orders[np.argmin(ec)])
            regret = ec - ec.min()
            rounds.append(dict(phase=phase, mu=mu, sd=sd, cs=cs, hint=h, m_i=m_i, s_i=s_i,
                               U=U, a=a if phase == 'main' else None, qstar=qstar,
                               regret=regret, ec=ec,
                               usual_range=(round(m_i - s_i), round(m_i + s_i))))
    return rounds

def cost(q, d, cs):
    return cs * np.maximum(d - q, 0) + C_W * np.maximum(q - d, 0)

SEQ_SEED = 7_2026
N_CANDIDATES = 5000

def candidate_sequences(rounds, m=N_CANDIDATES, seed=SEQ_SEED):
    r = np.random.default_rng(seed)
    D = np.empty((m, len(rounds)), dtype=int)
    for i, x in enumerate(rounds):
        k, p = demand_pmf(x['m_i'], x['s_i'])
        D[:, i] = r.choice(k, size=m, p=p)
    return D

def fixed_strategies(rounds):
    best = np.array([x['qstar'] for x in rounds])
    follow = np.array([x['a'] if x['phase'] == 'main' else round(x['mu']) for x in rounds])
    average = np.array([round(x['m_i']) for x in rounds])
    return dict(best=best, follow=follow, average=average)

def choose_sequences(rounds, n_seq=2):
    D = candidate_sequences(rounds)
    cs = np.array([x['cs'] for x in rounds])
    score = np.zeros(len(D)); tot_best = None
    for name, q in fixed_strategies(rounds).items():
        tot = cost(q[None, :], D, cs[None, :]).sum(1)
        q25, med, q75 = np.quantile(tot, [0.25, 0.5, 0.75])
        score = np.maximum(score, np.abs(tot - med) / (q75 - q25))
        if name == 'best': tot_best, med_best = tot, med
    n_unf = count_unfavourable(rounds, D)
    score = np.where(np.isin(n_unf, (3, 4)), score, np.inf)
    pick = np.argsort(score, kind='stable')[:n_seq]
    return D[pick], tot_best[pick], med_best

def scenario_orders(rounds):
    """Order 1 = as built; order 2 = main rounds reversed (hints stay with their scenarios), final rounds unchanged."""
    idx_main = [i for i, x in enumerate(rounds) if x['phase'] == 'main']
    idx_final = [i for i, x in enumerate(rounds) if x['phase'] == 'final']
    return [idx_main + idx_final, idx_main[::-1] + idx_final]

def version_of_block(b):
    """Returns (scenario order 0/1, demand sequence 0/1) for randomisation block b (0-based)."""
    v = b % 4
    return v // 2, v % 2

def unfavourable(rounds, demand, tail=0.20):
    """Boolean per round: normal main round whose realised cost of following a is in the worst `tail`."""
    out = np.zeros(len(rounds), bool)
    for i, x in enumerate(rounds):
        if x['phase'] != 'main' or x['hint'] is not None:
            continue
        k, p = demand_pmf(x['m_i'], x['s_i'])
        c_all = cost(x['a'], k, x['cs']); c_real = cost(x['a'], demand[i], x['cs'])
        out[i] = p[c_all >= c_real - 1e-12].sum() <= tail
    return out

def count_unfavourable(rounds, D, tail=0.20):
    """Vectorised number of unfavourable normal rounds for each row of D (same rule as unfavourable())."""
    n = np.zeros(len(D), dtype=int)
    for i, x in enumerate(rounds):
        if x['phase'] != 'main' or x['hint'] is not None:
            continue
        k, p = demand_pmf(x['m_i'], x['s_i'])
        c_all = cost(x['a'], k, x['cs'])
        tailp = np.array([p[c_all >= c - 1e-12].sum() for c in c_all])     # P(cost >= cost at each k)
        bad = tailp <= tail
        n += bad[np.clip(D[:, i] - k[0], 0, len(k) - 1)]
    return n

def usable_pairs(rounds, order, flags):
    """Pairs (t, t+1) in play order where both rounds are normal main rounds; returns list of (i, j, flag_i)."""
    pairs = []
    seq = [i for i in order if rounds[i]['phase'] == 'main']
    for i, j in zip(seq[:-1], seq[1:]):
        if rounds[i]['hint'] is None and rounds[j]['hint'] is None:
            pairs.append((i, j, bool(flags[i])))
    return pairs

ROUNDS = build_rounds()
DEMAND, DEMAND_TOTAL_BEST, DEMAND_MEDIAN_BEST = choose_sequences(ROUNDS)
