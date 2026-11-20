"""Power of H1 for smaller effects (added 20 Nov 2026 after reading Joslyn & LeClerc, 2012).

Same model, test and drop-out as power_sim3.py; only the number of blocks (6 people each) and the
effect size change. Group B is assumed to be like A (b_share = 0).
Usage: python power_small.py out.json [seed] [reps]
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import power_sim3 as ps

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'power_small.json'
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    rows = []
    for delta in (0.056, 0.065):
        es = ps.effect(delta, rng)
        for nb in (85, 130, 170, 220):
            ps.N_BLOCKS = nb
            t = time.time()
            res = [ps.one_study(delta, 0.0, rng) for _ in range(reps)]
            rej = np.array([r[0] for r in res])
            row = dict(delta=delta, reduction=es['reduction'], d=es['d'], blocks=nb,
                       assigned_per_group=2 * nb, reps=reps,
                       power_H1=float(rej[:, 0].mean()), power_H1_and_H2=float(rej[:, 1].mean()),
                       seconds=round(time.time() - t, 1))
            rows.append(row); print(row, flush=True)
            json.dump(rows, open(out, 'w'), indent=1)
