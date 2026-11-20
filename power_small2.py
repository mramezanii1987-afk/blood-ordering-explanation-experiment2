"""Follow-up to power_small.py: larger numbers of blocks for a 13% reduction. Usage: python power_small2.py out.json [seed] [reps]"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import power_sim3 as ps
out = sys.argv[1]; rng = np.random.default_rng(int(sys.argv[2])); reps = int(sys.argv[3])
rows = []
for delta, nb in ((0.056, 280), (0.056, 340), (0.065, 190)):
    ps.N_BLOCKS = nb
    res = [ps.one_study(delta, 0.0, rng) for _ in range(reps)]
    rej = np.array([r[0] for r in res])
    rows.append(dict(delta=delta, blocks=nb, assigned_per_group=2 * nb, reps=reps,
                     power_H1=float(rej[:, 0].mean()), power_H1_and_H2=float(rej[:, 1].mean())))
    print(rows[-1], flush=True)
json.dump(rows, open(out, 'w'), indent=1)
