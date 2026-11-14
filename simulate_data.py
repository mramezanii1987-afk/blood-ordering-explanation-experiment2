"""Simulated data in the pre-specified three-table format (registry, rounds, survey), for dry runs of the
analysis and for tests. Assignment: blocks of six rotated over the four versions (game.version_of_block),
two per group per block in random positions; the last block can be incomplete. Some assigned people never
start; others stop during the main rounds (half technical, half voluntary).
Usage: python simulate_data.py out_prefix [n_blocks] [delta] [seed]  -> out_prefix_{registry,rounds,survey}.csv
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game, behaviour as bh

def simulate_dataset(n_blocks=85, delta=0.12, b_share=0.0, dropout=0.10, never_start=0.02, seed=1,
                     last_block_size=6, kappa=4.0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    orders = game.scenario_orders(game.ROUNDS)
    reg, rows, surv = [], [], []
    pid = 0; t0 = np.datetime64('2026-12-01T09:00')
    for b in range(n_blocks):
        o, s = game.version_of_block(b)
        labels = np.array(['A', 'A', 'B', 'B', 'C', 'C']); rng.shuffle(labels)
        size = last_block_size if b == n_blocks - 1 else 6
        for g in labels[:size]:
            d = {'A': 0.0, 'B': delta * b_share, 'C': delta}[g]
            play = orders[o]
            q0, q1 = bh.simulate(1, d, rng, kappa, play, return_q0=True)
            u = rng.random()
            if u < never_start:
                stop, status = 0, 'never_started'
            elif u < never_start + dropout:
                stop = int(rng.integers(1, 30)); status = 'technical' if rng.random() < 0.5 else 'withdrew'
            else:
                stop, status = 36, 'completed'
            assigned = t0 + np.timedelta64(pid * 3, 'm')
            reg.append(dict(pid=pid, block=b, group=g, order=o, seq=s, assigned_at=str(assigned),
                            started=stop > 0, rounds_played=stop, status=status,
                            exit_at=str(assigned + np.timedelta64(stop, 'm'))))
            for pos in range(stop):
                main = pos < 30
                rows.append(dict(pid=pid, round=play[pos], position=pos + 1, q0=int(q0[0, pos]), q1=int(q1[0, pos]),
                                 confidence=int(rng.integers(30, 101)) if main else np.nan,
                                 time_rec=float(rng.gamma(4, 2.5)) if main else np.nan))
            if status == 'completed':
                tr = lambda: int(rng.integers(2, 8))
                surv.append(dict(pid=pid, trust_10=tr(), trust_20=tr(), trust_30=tr(), accuracy_10=tr(),
                                 accuracy_20=tr(), accuracy_30=tr(), understanding=int(rng.integers(0, 6))))
            pid += 1
    return pd.DataFrame(reg), pd.DataFrame(rows), pd.DataFrame(surv)

if __name__ == '__main__':
    pre = sys.argv[1] if len(sys.argv) > 1 else 'simulated'
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 85
    dl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
    sd = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    for name, df in zip(('registry', 'rounds', 'survey'), simulate_dataset(nb, dl, seed=sd)):
        df.to_csv(f'{pre}_{name}.csv', index=False)
    print('written', pre)
