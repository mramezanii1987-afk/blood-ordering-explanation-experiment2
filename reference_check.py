"""Independent reference implementation (pure Python, no numpy/scipy) and pre-specified checks
for the demand probabilities, best orders, expected costs and regret in game.py."""
import math, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import game

def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def ref_pmf(m, s):
    lo = max(0, math.floor(m - 3 * s)); hi = math.ceil(m + 3 * s)
    ks = list(range(lo, hi + 1))
    ps = [Phi((k + 0.5 - m) / s) - Phi((k - 0.5 - m) / s) for k in ks]
    tot = sum(ps)
    return ks, [p / tot for p in ps]

def ref_cost(q, ks, ps, cs):
    return sum(p * (cs * max(k - q, 0) + 1.0 * max(q - k, 0)) for k, p in zip(ks, ps))

def ref_best(U, ks, ps, cs):
    best_q, best_c = None, float('inf')
    for q in range(0, U + 1):               # enumerate every allowed integer order
        c = ref_cost(q, ks, ps, cs)
        if c < best_c - 1e-12:
            best_q, best_c = q, c
    return best_q, best_c

fails = []
def check(cond, msg):
    if not cond: fails.append(msg)

for i, r in enumerate(game.ROUNDS):
    ks, ps = ref_pmf(r['m_i'], r['s_i'])
    check(abs(sum(ps) - 1) < 1e-12, f'round {i}: probabilities do not sum to 1')
    q_ref, c_ref = ref_best(r['U'], ks, ps, r['cs'])
    check(q_ref == r['qstar'], f"round {i}: q* {r['qstar']} vs reference {q_ref}")
    check(abs(c_ref - r['ec'][r['qstar']]) < 1e-9, f'round {i}: expected cost at q* differs')
    for q in (0, r['qstar'] // 2, r['qstar'], r['U']):
        check(abs(ref_cost(q, ks, ps, r['cs']) - r['ec'][q]) < 1e-9, f'round {i}: expected cost differs at q={q}')
    check(min(r['regret']) >= 0 and r['regret'][r['qstar']] == 0, f'round {i}: regret not >= 0 or not 0 at q*')
    # recommendation under the base distribution
    if r['phase'] == 'main':
        kb, pb = ref_pmf(r['mu'], r['sd'])
        a_ref, _ = ref_best(r['U'], kb, pb, r['cs'])
        check(a_ref == r['a'], f"round {i}: recommendation {r['a']} vs reference {a_ref}")
        # direction of each hint relative to the recommendation
        h = r['hint']
        if h in ('level_up', 'unc_up'): check(r['qstar'] > r['a'], f'round {i}: {h} should raise the best order')
        if h in ('level_down', 'unc_down'): check(r['qstar'] < r['a'], f'round {i}: {h} should lower the best order')
        if h is None: check(r['qstar'] == r['a'], f'round {i}: normal round, q* should equal a')
    # agreement with the continuous percentile rule (truncation and integers allowed for)
    cr = r['cs'] / (r['cs'] + 1)
    from statistics import NormalDist
    q_cont = NormalDist(r['m_i'], r['s_i']).inv_cdf(cr)
    check(abs(q_cont - r['qstar']) <= 1.5, f"round {i}: integer q* {r['qstar']} far from continuous {q_cont:.2f}")

# textbook example: mean 100, SD 30, 4:1 -> continuous q* 125.25, expected cost 41.99
from statistics import NormalDist
ks, ps = ref_pmf(100, 30); q, c = ref_best(250, ks, ps, 4)
zc = NormalDist().inv_cdf(0.8)
cont_cost = 5 * 30 * math.exp(-zc * zc / 2) / math.sqrt(2 * math.pi)
check(abs(q - 125.25) <= 1, f'example: q* {q}')
check(abs(c - cont_cost) / cont_cost < 0.02, f'example: cost {c:.2f} vs continuous {cont_cost:.2f}')
check(abs(ref_cost(100, ks, ps, 4) - 59.84) < 0.6, f'example: cost at 100 = {ref_cost(100, ks, ps, 4):.2f}')

n_hint = {h: sum(1 for r in game.ROUNDS if r['hint'] == h) for h in game.HINT_TYPES}
print(f'rounds checked: {len(game.ROUNDS)}; hint cells: {n_hint}')
print(f'example: integer q* = {q}, expected cost = {c:.2f} (continuous {cont_cost:.2f}); cost at 100 = {ref_cost(100, ks, ps, 4):.2f}')
print('ALL CHECKS PASSED' if not fails else 'FAILURES:\n' + '\n'.join(fails))
