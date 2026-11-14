"""Power simulation (v7) with the blocked randomisation test and the fixed testing order H1 -> H2.

Assignment: 85 blocks of six, rotated over four versions (2 scenario orders x 2 demand sequences):
22/21/21/21 blocks, i.e. 43 blocks (86 per group) in order 1 and 42 blocks (84 per group) in order 2.
Two participants per group per block -> 170 assigned per group. Drop-out: 10 per cent at random.
Analysis: completers. Test for each contrast (C vs A = H1, C vs B = H2): analysis.block_perm_p
(one-sided, labels permuted within the original blocks among analysed members of the two groups,
ties count against). Decision: analysis.fixed_sequence (H1 at .05; H2 at .05 only if H1 rejected).
Usage: python power_sim3.py out.json [seed] [reps]
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game, behaviour as bh, analysis as an

N_BLOCKS = 85

def one_study(delta, b_share, rng, kappa=4.0, P=1000, dropout=0.10):
    blocks = np.repeat(np.arange(N_BLOCKS), 6)
    group = np.tile([0, 0, 1, 1, 2, 2], N_BLOCKS)                    # 0 = A, 1 = B, 2 = C
    n = len(blocks); reg = np.empty(n)
    for g, d in ((0, 0.0), (1, delta * b_share), (2, delta)):
        idx = np.where(group == g)[0]
        q = bh.simulate(len(idx), d, rng, kappa, an.NORMAL)
        reg[idx] = bh.mean_regret(q, an.NORMAL)
    keep = rng.random(n) >= dropout
    ps = []
    for comp in (0, 1):
        m = keep & ((group == 2) | (group == comp))
        ps.append(an.block_perm_p(reg[m], (group[m] == 2).astype(int), blocks[m], rng, P))
    return np.array(an.fixed_sequence(ps[0], ps[1])), ps

def effect(delta, rng, kappa=4.0, n=40000):
    rc = bh.mean_regret(bh.simulate(n, delta, rng, kappa, an.NORMAL), an.NORMAL)
    ra = bh.mean_regret(bh.simulate(n, 0.0, rng, kappa, an.NORMAL), an.NORMAL)
    sd = np.sqrt((rc.var(ddof=1) + ra.var(ddof=1)) / 2)
    return dict(reduction=float(1 - rc.mean() / ra.mean()), d=float((ra.mean() - rc.mean()) / sd),
                regret_A=float(ra.mean()), regret_C=float(rc.mean()))

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'power3.json'
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    v = [game.version_of_block(b) for b in range(N_BLOCKS)]
    print('blocks per version:', {k: v.count(k) for k in sorted(set(v))},
          '| order 1/2 blocks:', sum(1 for x in v if x[0] == 0), sum(1 for x in v if x[0] == 1))
    conds = [(0.0, 0.0, 4.0, 2 * reps)] + [(d, b, 4.0, reps) for d in (0.08, 0.10, 0.12, 0.15)
             for b in (0.0, 0.5, 0.9)] + [(0.12, 0.0, 2.0, reps)]
    rows = []
    for delta, b_share, kappa, nrep in conds:
        res = [one_study(delta, b_share, rng, kappa) for _ in range(nrep)]
        rej = np.array([r[0] for r in res]); ps = np.array([r[1] for r in res])
        es = effect(delta, rng, kappa) if delta > 0 else dict(reduction=0.0, d=0.0)
        row = dict(delta=delta, b_share=b_share, kappa=kappa, reps=nrep, **es,
                   power_H1=float(rej[:, 0].mean()), power_H1_and_H2=float(rej[:, 1].mean()),
                   any_reject=float(rej.any(1).mean()),
                   unconditional_p2_below_05=float((ps[:, 1] <= 0.05).mean()))
        rows.append(row); print(row, flush=True)
    json.dump(rows, open(out, 'w'), indent=1)
