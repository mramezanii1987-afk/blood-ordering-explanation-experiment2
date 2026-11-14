"""Pre-specified analysis code (v8).

Input: three tables (see DATA_DICTIONARY.md)
  registry : one row per ASSIGNED participant, including people who never start the first round
             pid, block, group, order, seq, assigned_at, started, rounds_played, status, exit_at
  rounds   : one row per participant and round played
             pid, round, position, q0, q1 [, confidence, time_rec]
  survey   : optional, one row per participant who answered
             pid [, trust_10, trust_20, trust_30, accuracy_10, accuracy_20, accuracy_30, understanding]

Primary comparison: among participants who complete all 30 main rounds, mean expected regret over the
18 normal main rounds in the comparison group (A for H1, B for H2) minus group C (positive = C better).
Because completion happens after assignment, this is a comparison CONDITIONAL ON COMPLETION, not an
effect for everyone assigned; tipping_point() assesses how strong departures from missing at random
would have to be to change the conclusion for all assigned participants.

OUTCOME_MAP.md lists every measure in the study plan, the function that computes it and the key in the
dictionary returned by run_analysis().
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game

R = game.ROUNDS
NORMAL = [i for i, r in enumerate(R) if r['phase'] == 'main' and r['hint'] is None]
HINT = [i for i, r in enumerate(R) if r['phase'] == 'main' and r['hint'] is not None]
FINAL = [i for i, r in enumerate(R) if r['phase'] == 'final']
N_MAIN, N_ALL = 30, 36
ORDERS = game.scenario_orders(R)
STATUSES = {'completed', 'technical', 'withdrew', 'never_started'}
RESPONSE_TYPES = ['no_change', 'away', 'partial', 'exact', 'beyond']
SMALL = [i for i in NORMAL if R[i]['mu'] <= 100]
LARGE = [i for i in NORMAL if R[i]['mu'] > 100]

# ================================================================ data checks
def validate(registry, rounds, survey=None):
    """Raises ValueError listing every problem found. Impossible values never enter the analysis."""
    p = []
    reg = registry
    if reg['pid'].duplicated().any(): p.append('duplicate pid in registry')
    if not reg['group'].isin(['A', 'B', 'C']).all(): p.append('unknown group label')
    if not reg['status'].isin(STATUSES).all(): p.append('unknown status')
    for _, r in reg.iterrows():
        if (r['order'], r['seq']) != game.version_of_block(int(r['block'])):
            p.append(f"pid {r['pid']}: version does not match block {r['block']}"); break
    if (reg.groupby(['block', 'group']).size() > 2).any(): p.append('more than two of one group in a block')
    ns = reg['status'] == 'never_started'
    if ((reg['rounds_played'] == 0) != ns).any(): p.append('never_started must have rounds_played == 0 and vice versa')
    if ((reg['status'] == 'completed') != (reg['rounds_played'] == N_ALL)).any(): p.append('completed must have 36 rounds')
    if ((reg['started'].astype(bool)) == ns).any(): p.append('started flag inconsistent with status')
    if not set(rounds['pid']).issubset(set(reg['pid'])): p.append('rounds contain pid not in registry')
    cnt = rounds.groupby('pid').size()
    rp = reg.set_index('pid')['rounds_played']
    if not (cnt.reindex(rp.index, fill_value=0) == rp).all(): p.append('number of rounds differs from rounds_played')
    order_of = reg.set_index('pid')['order']
    for pid, g in rounds.groupby('pid'):
        pos = g['position'].to_numpy()
        if not np.array_equal(np.sort(pos), np.arange(1, len(pos) + 1)):
            p.append(f'pid {pid}: positions not 1..n'); break
        exp = [ORDERS[int(order_of[pid])][k - 1] for k in g['position']]
        if not np.array_equal(g['round'].to_numpy(), np.array(exp)):
            p.append(f'pid {pid}: round does not match scenario order'); break
    U = np.array([R[int(i)]['U'] if 0 <= int(i) < N_ALL else -1 for i in rounds['round']])
    for col in ('q0', 'q1'):
        q = rounds[col].to_numpy(float)
        if np.isnan(q).any() or (q != np.round(q)).any() or (q < 0).any() or (q > U).any():
            p.append(f'{col}: missing, non-integer or outside 0..U')
    fin = rounds['round'].isin(FINAL)
    if (rounds.loc[fin, 'q0'] != rounds.loc[fin, 'q1']).any(): p.append('final rounds must have q0 == q1')
    if 'confidence' in rounds and ((rounds['confidence'] < 0) | (rounds['confidence'] > 100)).any():
        p.append('confidence outside 0..100')
    if survey is not None:
        if not set(survey['pid']).issubset(set(reg['pid'])): p.append('survey contains pid not in registry')
        for c in [c for c in survey.columns if c.startswith(('trust_', 'accuracy_'))]:
            v = survey[c].dropna()
            if ((v < 1) | (v > 7)).any(): p.append(f'{c} outside 1..7')
    if p:
        raise ValueError('; '.join(p))
    return True

# ================================================================ round- and person-level measures
def round_measures(rounds):
    d = rounds.copy()
    idx = d['round'].astype(int).to_numpy()
    q0 = d['q0'].astype(int).to_numpy(); q1 = d['q1'].astype(int).to_numpy()
    d['regret'] = [R[i]['regret'][q] for i, q in zip(idx, q1)]
    d['regret0'] = [R[i]['regret'][q] for i, q in zip(idx, q0)]
    a = np.array([R[i]['a'] if R[i]['a'] is not None else -1 for i in idx])
    d['regret_rec'] = [R[i]['regret'][a_] if a_ >= 0 else np.nan for i, a_ in zip(idx, a)]
    d['norm_regret'] = d['regret'] / np.array([R[i]['ec'][R[i]['qstar']] for i in idx])
    main = a >= 0
    d['dist0'] = np.where(main, np.abs(q0 - a), np.nan); d['dist1'] = np.where(main, np.abs(q1 - a), np.nan)
    d['revision'] = d['dist0'] - d['dist1']
    mu = np.array([R[i]['mu'] for i in idx]); den = a - q0
    ok = main & (np.abs(den) >= 0.05 * mu)
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(ok, (q1 - q0) / np.where(den == 0, 1, den), np.nan)
    d['woa_raw'] = w; d['woa'] = np.clip(w, 0, 1); d['woa_na'] = np.where(main, ~ok, np.nan)
    # response type (H5)
    rt = np.full(len(d), None, dtype=object)
    sgn = np.sign(a - q0)
    rt[main & (q1 == q0)] = 'no_change'
    m = main & (q1 != q0)
    rt[m & (q1 == a) ] = 'exact'
    moved = (q1 - q0) * np.where(sgn == 0, 1, sgn)
    rt[m & (q1 != a) & (sgn == 0)] = 'away'
    rt[m & (q1 != a) & (sgn != 0) & (moved < 0)] = 'away'
    rt[m & (q1 != a) & (sgn != 0) & (moved > 0) & (np.abs(q1 - q0) < np.abs(a - q0))] = 'partial'
    rt[m & (q1 != a) & (sgn != 0) & (moved > 0) & (np.abs(q1 - q0) > np.abs(a - q0))] = 'beyond'
    d['rtype'] = rt
    # pull to centre: share of the gap between average and best order that is NOT closed (0 = best, 1 = average)
    qs = np.array([R[i]['qstar'] for i in idx]); mi = np.array([R[i]['m_i'] for i in idx])
    gap = qs - mi
    d['pull'] = np.where(np.abs(gap) >= 1, (qs - q1) / np.where(gap == 0, 1, gap), np.nan)
    d['normal'] = np.isin(idx, NORMAL); d['hintr'] = np.isin(idx, HINT); d['final'] = np.isin(idx, FINAL)
    d['small'] = np.isin(idx, SMALL); d['large'] = np.isin(idx, LARGE)
    if 'q1' in d:
        better = (np.array(d['regret']) < np.array(d['regret0'])) & (np.array(d['regret']) < np.nan_to_num(d['regret_rec'], nan=np.inf))
        d['beats_both'] = np.where(d['hintr'], better, np.nan)
    return d

def person_table(registry, rounds):
    """One row per ASSIGNED participant (left join of the registry with the rounds played)."""
    import pandas as pd
    d = round_measures(rounds)
    pt = registry.set_index('pid').copy()
    def mean_of(mask, col):
        return d[mask].groupby('pid')[col].mean().reindex(pt.index)
    nrm, hnt, fin = d['normal'], d['hintr'], d['final']
    pt['n_normal'] = d[nrm].groupby('pid').size().reindex(pt.index, fill_value=0)
    pt['n_main'] = d[~fin].groupby('pid').size().reindex(pt.index, fill_value=0)
    pt['completer'] = pt['n_main'] >= N_MAIN
    pt['regret_normal'] = mean_of(nrm, 'regret')
    pt['regret0_normal'] = mean_of(nrm, 'regret0')
    pt['reduction_normal'] = pt['regret0_normal'] - pt['regret_normal']
    pt['norm_regret_normal'] = mean_of(nrm, 'norm_regret')
    pt['regret_small'] = mean_of(nrm & d['small'], 'regret'); pt['regret_large'] = mean_of(nrm & d['large'], 'regret')
    pt['revision_normal'] = mean_of(nrm, 'revision')
    pt['dist0_normal'] = mean_of(nrm, 'dist0'); pt['dist1_normal'] = mean_of(nrm, 'dist1')
    pt['regret_hint'] = mean_of(hnt, 'regret'); pt['regret0_hint'] = mean_of(hnt, 'regret0')
    pt['regret_rec_hint'] = mean_of(hnt, 'regret_rec'); pt['beats_both_hint'] = mean_of(hnt, 'beats_both')
    pt['regret_final'] = mean_of(fin, 'regret')
    pt['woa_raw_normal'] = mean_of(nrm, 'woa_raw'); pt['woa_normal'] = mean_of(nrm, 'woa')
    pt['woa_na_main'] = mean_of(~fin, 'woa_na')
    for t in RESPONSE_TYPES:
        pt[f'share_{t}'] = (d[nrm].assign(x=(d.loc[nrm, 'rtype'] == t).astype(float))
                            .groupby('pid')['x'].mean().reindex(pt.index))
    pt['pull_normal'] = mean_of(nrm, 'pull'); pt['pull_final'] = mean_of(fin, 'pull')
    if 'confidence' in d: pt['confidence'] = mean_of(~fin, 'confidence')
    if 'time_rec' in d: pt['time_rec'] = mean_of(~fin, 'time_rec')
    pt['stratum'] = pt['order'] * 2 + pt['seq']
    return pt.reset_index()

def additive_person_effects(rounds, tol=1e-10, max_iter=1000):
    """Two-way additive model on all observed normal person-rounds: regret_ir = person_i + round_r + e,
    fitted by alternating least squares; round effects centred to mean zero, so a person effect is the
    person's predicted average regret over all 18 normal rounds."""
    d = round_measures(rounds); d = d[d['normal']]
    pid, pi = np.unique(d['pid'], return_inverse=True)
    rid, ri = np.unique(d['round'], return_inverse=True)
    y = d['regret'].to_numpy(float)
    a = np.zeros(len(pid)); g = np.zeros(len(rid))
    cnt_p = np.bincount(pi, minlength=len(pid)); cnt_r = np.bincount(ri, minlength=len(rid))
    for _ in range(max_iter):
        a_new = np.bincount(pi, y - g[ri], len(pid)) / cnt_p
        g_new = np.bincount(ri, y - a_new[pi], len(rid)) / cnt_r
        shift = g_new.mean(); g_new -= shift; a_new += shift
        done = max(np.abs(a_new - a).max(), np.abs(g_new - g).max()) < tol
        a, g = a_new, g_new
        if done: break
    return dict(zip(pid, a))

# ================================================================ tests and intervals
def _perm_labels(labs, bidx, rng, P):
    keys = rng.random((P, len(labs))) + bidx[None, :] * 10.0     # blocks stay contiguous after sorting
    order = np.argsort(keys, axis=1)
    perm = np.empty_like(order); perm[np.arange(P)[:, None], order] = labs[None, :]
    return perm

def block_perm_stats(vals, labs, blocks, rng, P=10000, stat='diff'):
    """Randomisation distribution under within-block re-assignment (label counts per block fixed).
    labs: 1 = C, 0 = comparison. stat='diff': mean(comparison) - mean(C); stat='welch': that difference
    divided by its Welch standard error (studentized). Returns (observed, permutation statistics)."""
    vals = np.asarray(vals, float); labs = np.asarray(labs, int); blocks = np.asarray(blocks)
    o = np.lexsort((labs, blocks)); vals, labs, blocks = vals[o], labs[o], blocks[o]
    _, bidx = np.unique(blocks, return_inverse=True)
    nC = labs.sum(); nO = len(labs) - nC
    if nC < 1 or nO < 1:
        raise ValueError('both groups need at least one analysed participant')
    def f(sC, sO, qC, qO):
        d = sO / nO - sC / nC
        if stat == 'diff': return d
        vC = (qC - sC ** 2 / nC) / max(nC - 1, 1); vO = (qO - sO ** 2 / nO) / max(nO - 1, 1)
        return d / np.sqrt(vC / nC + vO / nO)
    tot, tot2 = vals.sum(), (vals ** 2).sum()
    obs = f(vals[labs == 1].sum(), vals[labs == 0].sum(), (vals[labs == 1] ** 2).sum(), (vals[labs == 0] ** 2).sum())
    perm = _perm_labels(labs, bidx, rng, P)
    sC = perm @ vals; qC = perm @ (vals ** 2)
    return obs, f(sC, tot - sC, qC, tot2 - qC)

def block_perm_p(vals, labs, blocks, rng, P=10000, stat='diff'):
    """One-sided p for 'C lower than comparison' = (1 + #{perm >= obs}) / (P + 1); ties count against."""
    obs, stats = block_perm_stats(vals, labs, blocks, rng, P, stat)
    return (1 + np.sum(stats >= obs - 1e-12)) / (P + 1)

def block_bootstrap_ci(vals, labs, blocks, strata, rng, B=10000, level=0.95):
    """Percentile CI for mean(comparison) - mean(C), resampling whole blocks with replacement within
    each stratum (version = scenario order x demand sequence)."""
    vals = np.asarray(vals, float); labs = np.asarray(labs, int); blocks = np.asarray(blocks); strata = np.asarray(strata)
    ub = np.unique(blocks)
    sums = np.array([(vals[(blocks == b) & (labs == 1)].sum(), ((blocks == b) & (labs == 1)).sum(),
                      vals[(blocks == b) & (labs == 0)].sum(), ((blocks == b) & (labs == 0)).sum()) for b in ub])
    b_str = np.array([strata[blocks == b][0] for b in ub])
    tot = np.zeros((B, 4))
    for s in np.unique(b_str):
        a = sums[b_str == s]
        tot += a[rng.integers(0, len(a), (B, len(a)))].sum(1)
    ok = (tot[:, 1] > 0) & (tot[:, 3] > 0)
    d = tot[ok, 2] / tot[ok, 3] - tot[ok, 0] / tot[ok, 1]
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))

def holm(ps, alpha=0.05):
    ps = np.asarray(ps, float); o = np.argsort(ps); rej = np.zeros(len(ps), bool)
    for k, i in enumerate(o):
        if ps[i] <= alpha / (len(ps) - k): rej[i] = True
        else: break
    return rej

def fixed_sequence(p1, p2, alpha=0.05):
    """H1 first at alpha; H2 at alpha only if H1 is rejected. Controls the familywise error at alpha."""
    r1 = p1 <= alpha
    return bool(r1), bool(r1 and p2 <= alpha)

def _pair(pt, value, comp, subset, ref='C'):
    m = pt['group'].isin([ref, comp]) & pt[value].notna()
    if subset is not None: m &= subset
    d = pt[m]
    return d, (d['group'] == ref).astype(int).to_numpy()

def compare(pt, value, comp, rng, P=10000, B=10000, subset=None, two_sided=False, stat='diff', ref='C'):
    """Blocked randomisation test and block-bootstrap CI for mean(comp) - mean(ref)."""
    d, labs = _pair(pt, value, comp, subset, ref)
    obs, stats = block_perm_stats(d[value], labs, d['block'], rng, P, stat)
    p = (1 + np.sum(stats >= obs - 1e-12)) / (P + 1)
    if two_sided:
        p = min(1.0, 2 * min(p, (1 + np.sum(stats <= obs + 1e-12)) / (P + 1)))
    diff = d.loc[labs == 0, value].mean() - d.loc[labs == 1, value].mean()
    lo, hi = block_bootstrap_ci(d[value], labs, d['block'], d['stratum'], rng, B)
    return dict(diff=float(diff), ci=(lo, hi), p=float(p), n_ref=int(labs.sum()), n_comp=int((1 - labs).sum()))

def estimate(pt, value, comp, rng, B=10000, subset=None, ref='C'):
    """Descriptive contrast (no test): mean(comp) - mean(ref) with block-bootstrap CI."""
    d, labs = _pair(pt, value, comp, subset, ref)
    diff = d.loc[labs == 0, value].mean() - d.loc[labs == 1, value].mean()
    lo, hi = block_bootstrap_ci(d[value], labs, d['block'], d['stratum'], rng, B)
    return dict(diff=float(diff), ci=(lo, hi), n_ref=int(labs.sum()), n_comp=int((1 - labs).sum()))

def mann_whitney(pt, value, comp, subset=None):
    from scipy.stats import mannwhitneyu
    d, labs = _pair(pt, value, comp, subset)
    res = mannwhitneyu(d.loc[labs == 0, value], d.loc[labs == 1, value], alternative='greater')
    return dict(U=float(res.statistic), p=float(res.pvalue))

# ================================================================ drop-out
def bounds_values(pt, comp, q_hi=0.90, q_lo=0.10):
    """Pessimistic scenario: every non-completer (including never-started) in C gets the 90th percentile of
    completers' regret; every non-completer in the comparison group gets the 10th percentile."""
    v = pt['regret_normal'].copy()
    ref = pt.loc[pt['completer'], 'regret_normal']
    v[(~pt['completer']) & (pt['group'] == 'C')] = ref.quantile(q_hi)
    v[(~pt['completer']) & (pt['group'] == comp)] = ref.quantile(q_lo)
    return v

def imputed_all_assigned(pt, rounds, comp, delta):
    """Value for every assigned member of C and comp: observed normal-round regrets plus, for missing
    normal rounds, the mean regret of that round among completers of the same group (missing at random
    given group and round), plus `delta` points per missing round in group C (departure from MAR)."""
    d = round_measures(rounds); d = d[d['normal']].merge(pt[['pid', 'group', 'completer']], on='pid')
    rm = d[d['completer']].groupby(['group', 'round'])['regret'].mean()
    obs = d.groupby('pid')['regret'].apply(lambda s: s.to_numpy()).to_dict()
    seen = d.groupby('pid')['round'].apply(set).to_dict()
    out = {}
    for _, r in pt[pt['group'].isin(['C', comp])].iterrows():
        g = r['group']; s = seen.get(r['pid'], set())
        miss = [rm[(g, i)] + (delta if g == 'C' else 0.0) for i in NORMAL if i not in s]
        vals = np.concatenate([obs.get(r['pid'], np.array([])), np.array(miss)])
        out[r['pid']] = vals.mean()
    return pt['pid'].map(out)

def tipping_point(pt, rounds, comp, rng, deltas=None, P=2000, alpha=0.05):
    """Tipping-point analysis for ALL assigned participants. Returns the grid (delta, difference, p) and
    the smallest delta at which the one-sided p exceeds alpha (None if never, or if p > alpha at 0)."""
    deltas = np.arange(0, 6.01, 0.25) if deltas is None else deltas
    grid = []
    for dl in deltas:
        v = imputed_all_assigned(pt, rounds, comp, dl)
        tmp = pt.assign(_v=v)
        d, labs = _pair(tmp, '_v', comp, None)
        p = block_perm_p(d['_v'], labs, d['block'], rng, P)
        grid.append(dict(delta=float(dl), diff=float(d.loc[labs == 0, '_v'].mean() - d.loc[labs == 1, '_v'].mean()), p=float(p)))
    tip = None
    if grid[0]['p'] <= alpha:
        tip = next((g['delta'] for g in grid if g['p'] > alpha), None)
    return dict(grid=grid, tipping_delta=tip,
                sd_person_regret=float(pt.loc[pt['completer'], 'regret_normal'].std()))

def attrition_table(pt):
    t = pt.groupby(['group', 'order', 'seq']).agg(
        assigned=('pid', 'size'), started=('started', 'sum'), completed=('completer', 'sum'),
        never_started=('status', lambda s: (s == 'never_started').sum()),
        technical=('status', lambda s: (s == 'technical').sum()),
        withdrew=('status', lambda s: (s == 'withdrew').sum()))
    t['dropped_after_start'] = t['started'] - t['completed']
    curve = pt[pt['started'].astype(bool) & ~pt['completer']].groupby('group')['rounds_played'].describe()
    return t, curve

# ================================================================ secondary and exploratory summaries
def revision_by_round(rounds, registry):
    d = round_measures(rounds).merge(registry[['pid', 'group']], on='pid'); d = d[d['normal']]
    return d.groupby(['group', 'position'])[['dist0', 'dist1', 'revision']].mean()

def h4_analysis(rounds, registry, rng, B=5000):
    """Exploratory H4. WOA (0-1) in a normal round that directly follows a normal round, by whether the
    earlier round was unfavourable in the participant's demand sequence. Within each group: mean WOA
    after favourable minus after unfavourable, with a CI from resampling participants; C - A contrast of
    that drop, also by resampling participants within groups. Counts by version are reported."""
    import pandas as pd
    d = round_measures(rounds).merge(registry[['pid', 'group', 'order', 'seq']], on='pid').sort_values(['pid', 'position'])
    flags = {s: game.unfavourable(R, game.DEMAND[s]) for s in (0, 1)}
    d['unfav'] = [bool(flags[int(s)][int(i)]) for s, i in zip(d['seq'], d['round'])]
    d['prev_normal'] = d.groupby('pid')['normal'].shift(1, fill_value=False).astype(bool)
    d['prev_unfav'] = d.groupby('pid')['unfav'].shift(1, fill_value=False).astype(bool)
    pairs = d[d['normal'] & d['prev_normal'] & d['woa'].notna()]
    counts = pairs.groupby(['order', 'seq', 'group']).agg(persons=('pid', 'nunique'), pairs=('pid', 'size'),
                                                       pairs_after_unfav=('prev_unfav', 'sum'))
    per = pairs.groupby(['pid', 'group', 'prev_unfav'])['woa'].mean().unstack('prev_unfav')
    per = per.dropna()                                    # persons with both kinds of pairs
    per['drop'] = per[False] - per[True]
    out = {}
    drops = {}
    for g in ('A', 'B', 'C'):
        x = per.xs(g, level='group')['drop'].to_numpy() if g in per.index.get_level_values('group') else np.array([])
        drops[g] = x
        if len(x) > 1:
            bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
            out[g] = dict(mean_drop=float(x.mean()), ci=(float(np.quantile(bs, .025)), float(np.quantile(bs, .975))), persons=int(len(x)))
    if len(drops['A']) > 1 and len(drops['C']) > 1:
        a, c = drops['A'], drops['C']
        bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1) - c[rng.integers(0, len(c), (B, len(c)))].mean(1)
        out['A_minus_C'] = dict(diff=float(a.mean() - c.mean()), ci=(float(np.quantile(bs, .025)), float(np.quantile(bs, .975))))
    return dict(counts=counts, by_group=out)

def survey_summary(pt, survey, rng, B=10000):
    cols = [c for c in survey.columns if c != 'pid']
    m = pt.merge(survey, on='pid', how='left')
    res = {}
    for c in cols:
        res[c] = dict(means=m.groupby('group')[c].mean().to_dict(),
                      A_minus_C=estimate(m, c, 'A', rng, B), B_minus_C=estimate(m, c, 'B', rng, B))
    return res

# ================================================================ full analysis
def run_analysis(registry, rounds, survey=None, seed=20261112, P=10000, B=10000, tipping=True):
    validate(registry, rounds, survey)
    rng = np.random.default_rng(seed)
    pt = person_table(registry, rounds)
    comp_ = pt['completer']
    res = dict(n_assigned=int(len(pt)))
    # --- confirmatory (conditional on completion)
    for h, comp in (('H1', 'A'), ('H2', 'B')):
        res[h] = compare(pt, 'regret_normal', comp, rng, P, B, subset=comp_)
    res['decision'] = dict(zip(('H1', 'H2'), fixed_sequence(res['H1']['p'], res['H2']['p'])))
    # --- sensitivity
    pt['additive'] = pt['pid'].map(additive_person_effects(rounds))
    sens = {}
    for h, comp in (('H1', 'A'), ('H2', 'B')):
        pt[f'bounds_{comp}'] = bounds_values(pt, comp)
        s = dict(
            at_least_1_normal=compare(pt, 'regret_normal', comp, rng, P, B, subset=pt['n_normal'] >= 1),
            at_least_9_normal=compare(pt, 'regret_normal', comp, rng, P, B, subset=pt['n_normal'] >= 9),
            additive_model=compare(pt, 'additive', comp, rng, P, B, subset=pt['n_normal'] >= 1),
            pessimistic_bounds=compare(pt, f'bounds_{comp}', comp, rng, P, B),
            studentized=compare(pt, 'regret_normal', comp, rng, P, B, subset=comp_, stat='welch'),
            mann_whitney=mann_whitney(pt, 'regret_normal', comp, subset=comp_),
            normalised_regret=compare(pt, 'norm_regret_normal', comp, rng, P, B, subset=comp_),
            smaller_scenarios=estimate(pt, 'regret_small', comp, rng, B, subset=comp_),
            larger_scenarios=estimate(pt, 'regret_large', comp, rng, B, subset=comp_),
            by_sequence={s_: estimate(pt, 'regret_normal', comp, rng, B, subset=comp_ & (pt['seq'] == s_)) for s_ in (0, 1)})
        if tipping:
            s['tipping_point'] = tipping_point(pt, rounds, comp, rng, P=min(P, 2000))
        sens[h] = s
    res['sensitivity'] = sens
    # --- secondary: M (C larger revision; one-sided) and H3 (two-sided), Holm within family
    pt['neg_revision'] = -pt['revision_normal']
    res['M'] = {c: compare(pt, 'neg_revision', c, rng, P, B, subset=comp_) for c in ('A', 'B')}
    res['H3'] = compare(pt, 'regret_hint', 'A', rng, P, B, subset=comp_, two_sided=True)
    res['secondary_holm_reject'] = holm([res['M']['A']['p'], res['M']['B']['p'], res['H3']['p']]).tolist()
    res['M_interpretation'] = dict(
        distances_by_group=pt[comp_].groupby('group')[['dist0_normal', 'dist1_normal', 'revision_normal']].mean(),
        by_round=revision_by_round(rounds, registry))
    # --- exploratory and additional comparisons (estimates and intervals only)
    ex = {}
    ex['B_vs_A_regret'] = compare(pt, 'regret_normal', 'A', rng, P, B, subset=comp_, two_sided=True, ref='B')
    ex['reduction_first_to_final'] = {c: estimate(pt, 'reduction_normal', c, rng, B, subset=comp_) for c in ('A', 'B')}
    ex['final_rounds'] = {c: estimate(pt, 'regret_final', c, rng, B, subset=comp_) for c in ('A', 'B')}
    ex['hint_value'] = dict(
        means=pt[comp_].groupby('group')[['regret0_hint', 'regret_rec_hint', 'regret_hint', 'beats_both_hint']].mean(),
        beats_both_A_minus_C=estimate(pt, 'beats_both_hint', 'A', rng, B, subset=comp_))
    ex['H4'] = h4_analysis(rounds, registry, rng, min(B, 5000))
    ex['H5'] = dict(shares=pt[comp_].groupby('group')[[f'share_{t}' for t in RESPONSE_TYPES]].mean(),
                    **{f'{t}_{c}_minus_C': estimate(pt, f'share_{t}', c, rng, B, subset=comp_)
                       for t in RESPONSE_TYPES for c in ('A', 'B')})
    ex['woa_descriptive'] = pt[comp_].groupby('group')[['woa_raw_normal', 'woa_normal', 'woa_na_main']].mean()
    ex['pull_to_centre'] = pt[comp_].groupby('group')[['pull_normal', 'pull_final']].mean()
    for col in ('confidence', 'time_rec'):
        if col in pt:
            ex[col] = dict(means=pt[comp_].groupby('group')[col].mean().to_dict(),
                           **{f'{c}_minus_C': estimate(pt, col, c, rng, B, subset=comp_) for c in ('A', 'B')})
    if survey is not None:
        ex['survey'] = survey_summary(pt[comp_], survey, rng, B)
    res['exploratory'] = ex
    res['attrition'], res['dropout_curve'] = attrition_table(pt)
    return res

# ================================================================ payment
def payment(rounds_played, total_cost, status, S, r, base_full, n_rounds=N_ALL):
    """Single rule: the share of the study completed is the share of the 36 rounds played.
    completed: full base + max(0, S - r*cost). technical stop: base x share, bonus max(0, S x share - r*cost).
    withdrew (voluntary): base x share, no bonus (to be confirmed with the ethics committee and Prolific).
    never_started (assigned but left before round 1): paid under Prolific's rules for the time spent."""
    share = rounds_played / n_rounds
    if status == 'completed':
        return base_full, max(0.0, S - r * total_cost)
    if status == 'technical':
        return base_full * share, max(0.0, S * share - r * total_cost)
    if status == 'withdrew':
        return base_full * share, 0.0
    if status == 'never_started':
        return 0.0, 0.0
    raise ValueError(status)
