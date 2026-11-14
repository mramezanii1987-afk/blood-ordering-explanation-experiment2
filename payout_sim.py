"""Payout simulation for the bonus rule: bonus = max(0, S - r * total realised cost over all 36 rounds).

v7: two demand sequences (game.DEMAND), each used by half of the blocks. S and r are chosen so that
(i) the average simulated participant (group A behaviour model) receives about GBP 2.50, averaged
over the two sequences, and (ii) the bonus is still at least GBP 0.25 at the 99.9th percentile of
total cost for simulated participants across 5,000 alternative demand sequences. The calibration uses
only the behaviour model (in the pilot: all pilot participants pooled across groups), never group labels.
Usage: python payout_sim.py [output.json]
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game, behaviour as bh

R = game.ROUNDS; NR = len(R)
rng = np.random.default_rng(11)
cs = np.array([r['cs'] for r in R])

def strategy_orders(name, n=1):
    if name == 'behavioural':
        return bh.simulate(n, 0.0, rng)
    q = np.empty((n, NR), dtype=int)
    for i, r in enumerate(R):
        if name == 'best': q[:, i] = r['qstar']
        elif name == 'follow': q[:, i] = r['a'] if r['phase'] == 'main' else round(r['mu'])
        elif name == 'average': q[:, i] = round(r['m_i'])
        elif name == 'constant100': q[:, i] = 100
        elif name == 'random': q[:, i] = rng.integers(0, r['U'] + 1, n)
    return q

def total_cost(q, D):
    return game.cost(q, D, cs[None, :]).sum(1)

def calibrate(beh, seqs, alt):
    T_fixed = np.concatenate([total_cost(beh, s[None, :]) for s in seqs])
    T_alt = total_cost(beh, alt[rng.integers(0, len(alt), len(beh))])
    T_mean, T_hi = T_fixed.mean(), np.quantile(T_alt, 0.999)
    r_rate = (2.50 - 0.25) / (T_hi - T_mean)
    return 2.50 + r_rate * T_mean, r_rate, T_mean, T_hi

if __name__ == '__main__':
    alt = game.candidate_sequences(R, 5000, seed=2026_11)
    beh = strategy_orders('behavioural', 20000)
    S, r_rate, T_mean, T_hi = calibrate(beh, game.DEMAND, alt)
    out = dict(S=round(float(S), 2), rate_per_point=float(r_rate), T_beh_mean=float(T_mean),
               T_beh_999=float(T_hi), sequence_best_totals=[float(x) for x in game.DEMAND_TOTAL_BEST],
               candidate_median_best_total=float(game.DEMAND_MEDIAN_BEST), strategies={})
    for name in ('best', 'follow', 'average', 'constant100', 'random', 'behavioural'):
        q = beh if name == 'behavioural' else strategy_orders(name, 2000 if name == 'random' else 1)
        bf = [np.maximum(0, S - r_rate * total_cost(q, s[None, :])) for s in game.DEMAND]
        qa = q[:2000]
        Ta = total_cost(qa, alt[rng.integers(0, len(alt), len(qa))]) if len(qa) > 1 else total_cost(qa, alt)
        ba = np.maximum(0, S - r_rate * Ta)
        exp_cost = sum(R[i]['ec'][q[:, i]].mean() for i in range(NR))
        out['strategies'][name] = dict(expected_total_cost=round(float(exp_cost), 1),
            bonus_seq1_mean=round(float(bf[0].mean()), 2), bonus_seq2_mean=round(float(bf[1].mean()), 2),
            bonus_alt_mean=round(float(ba.mean()), 2), bonus_alt_p05=round(float(np.quantile(ba, 0.05)), 2),
            bonus_alt_p95=round(float(np.quantile(ba, 0.95)), 2), share_at_zero_alt=round(float((ba == 0).mean()), 4),
            share_at_zero_fixed=round(float(np.mean(np.concatenate(bf) == 0)), 4))
    eb = out['strategies']['best']['expected_total_cost']; ea = out['strategies']['average']['expected_total_cost']
    out['expected_gap_best_vs_average_GBP'] = round(float(r_rate * (ea - eb)), 2)
    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'payout.json', 'w'), indent=1)
