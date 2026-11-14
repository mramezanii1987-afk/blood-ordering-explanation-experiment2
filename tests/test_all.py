"""Unit tests (run with: python -m pytest -q tests)."""
import os, sys, itertools, subprocess
import numpy as np
import pytest
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import game, analysis as an, simulate_data as sd

R = game.ROUNDS

# ------------------------------------------------------------------ game definition
def test_36_unique_scenarios_and_phases():
    keys = [(r['mu'], r['sd'] / r['mu'], r['cs']) for r in R]
    assert len(R) == 36 and len(set(keys)) == 36
    assert sum(r['phase'] == 'main' for r in R) == 30 and sum(r['phase'] == 'final' for r in R) == 6

def test_hint_design_two_by_two_three_per_cell():
    from collections import Counter
    c = Counter(r['hint'] for r in R if r['phase'] == 'main')
    assert c[None] == 18 and all(c[h] == 3 for h in game.HINT_TYPES)
    assert all(r['hint'] is None for r in R if r['phase'] == 'final')

@pytest.mark.parametrize('i', range(36))
def test_each_round_regret_and_best_order(i):
    r = R[i]
    assert np.all(r['regret'] >= 0) and r['regret'][r['qstar']] == 0
    if r['hint'] in (None, 'level_up', 'unc_up', 'unc_down'):
        assert r['qstar'] > r['m_i']                     # best order above the (hint) average
    if r['phase'] == 'main' and r['hint'] is None:
        assert r['a'] == r['qstar']                      # recommendation is best in normal rounds
    k, p = game.demand_pmf(r['m_i'], r['s_i'])
    assert abs(p.sum() - 1) < 1e-12 and k.min() >= 0

def test_independent_reference_implementation_agrees():
    out = subprocess.run([sys.executable, os.path.join(HERE, 'reference_check.py')], capture_output=True, text=True)
    assert 'ALL CHECKS PASSED' in out.stdout, out.stdout + out.stderr

def test_example_scenario_numbers():
    k, p = game.demand_pmf(100, 30)
    ec = game.expected_cost(np.arange(0, 251), k, p, 4)
    assert int(np.argmin(ec)) == 125 and round(ec[125], 2) == 41.56 and round(ec[100], 2) == 59.36

def test_scenario_orders_contain_same_rounds():
    o1, o2 = game.scenario_orders(R)
    assert sorted(o1) == sorted(o2) == list(range(36)) and o1[30:] == o2[30:]

# ------------------------------------------------------------------ demand sequences and versions
def test_demand_sequence_rule_is_deterministic_and_typical():
    D1, t1, m1 = game.choose_sequences(R); D2, t2, m2 = game.choose_sequences(R)
    assert np.array_equal(D1, D2) and D1.shape == (2, 36)
    for i, r in enumerate(R):
        k, _ = game.demand_pmf(r['m_i'], r['s_i'])
        assert k.min() <= D1[0, i] <= k.max() and k.min() <= D1[1, i] <= k.max()
    for d in D1:
        assert game.unfavourable(R, d).sum() in (3, 4)

def test_vectorised_unfavourable_count_matches_loop():
    D = game.candidate_sequences(R, 200, seed=3)
    assert np.array_equal(game.count_unfavourable(R, D), [game.unfavourable(R, d).sum() for d in D])

def test_unfavourable_only_in_normal_rounds():
    for d in game.DEMAND:
        f = game.unfavourable(R, d)
        assert all(R[i]['phase'] == 'main' and R[i]['hint'] is None for i in np.where(f)[0])

def test_blocks_split_42_43_by_order():
    v = [game.version_of_block(b) for b in range(85)]
    assert v.count((0, 0)) == 22 and v.count((0, 1)) == 21 and v.count((1, 0)) == 21 and v.count((1, 1)) == 21
    assert sum(x[0] == 0 for x in v) * 2 == 86 and sum(x[0] == 1 for x in v) * 2 == 84

# ------------------------------------------------------------------ randomisation test
def exact_block_p(vals, labs, blocks):
    """Full enumeration of all within-block re-assignments (small examples only)."""
    vals = np.asarray(vals, float); labs = np.asarray(labs); blocks = np.asarray(blocks)
    obs = vals[labs == 0].mean() - vals[labs == 1].mean()
    per_block = []
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]; k = labs[idx].sum()
        per_block.append([(idx, set(c)) for c in itertools.combinations(idx, k)])
    stats = []
    for combo in itertools.product(*per_block):
        lab = np.zeros(len(vals), int)
        for idx, cset in combo:
            lab[list(cset)] = 1
        stats.append(vals[lab == 0].mean() - vals[lab == 1].mean())
    stats = np.array(stats)
    return np.mean(stats >= obs - 1e-12)

def test_monte_carlo_matches_exact_enumeration():
    rng = np.random.default_rng(0)
    vals = np.array([5., 3., 4., 1., 6., 2., 7., 2., 3., 5.])
    labs = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    blocks = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    p_exact = exact_block_p(vals, labs, blocks)
    p_mc = an.block_perm_p(vals, labs, blocks, rng, P=40000)
    assert abs(p_mc - p_exact) < 0.01

def test_permutations_keep_label_counts_within_blocks():
    rng = np.random.default_rng(1)
    vals = np.arange(9.); labs = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1]); blocks = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    o = np.lexsort((labs, blocks)); b_sorted = blocks[o]; l_sorted = labs[o]
    keys = rng.random((200, 9)) + b_sorted[None, :] * 10.0
    order = np.argsort(keys, axis=1); perm = np.empty_like(order)
    perm[np.arange(200)[:, None], order] = l_sorted[None, :]
    for b in range(3):
        assert np.all(perm[:, b_sorted == b].sum(1) == l_sorted[b_sorted == b].sum())

def test_ties_count_against_hypothesis():
    rng = np.random.default_rng(2)
    p = an.block_perm_p(np.ones(12), np.tile([0, 1], 6), np.repeat(range(3), 4), rng, P=500)
    assert p == 1.0

def test_incomplete_blocks_and_single_group_blocks():
    rng = np.random.default_rng(3)
    vals = np.array([1., 2., 3., 4., 5., 6., 7.])
    labs = np.array([1, 1, 0, 1, 0, 0, 1])               # block 0 only C, block 3 has one person
    blocks = np.array([0, 0, 1, 1, 2, 2, 3])
    p = an.block_perm_p(vals, labs, blocks, rng, P=2000)
    assert 0 < p <= 1
    assert abs(p - exact_block_p(vals, labs, blocks)) < 0.03

def test_type_one_error_under_null():
    rng = np.random.default_rng(4)
    blocks = np.repeat(np.arange(40), 4); labs = np.tile([0, 0, 1, 1], 40)
    rej = [an.block_perm_p(rng.gamma(2, 1, 160), labs, blocks, rng, P=400) <= 0.05 for _ in range(400)]
    assert np.mean(rej) < 0.075

def test_fixed_sequence():
    assert an.fixed_sequence(0.01, 0.20) == (True, False)
    assert an.fixed_sequence(0.20, 0.01) == (False, False)
    assert an.fixed_sequence(0.04, 0.05) == (True, True)

def test_holm():
    assert an.holm([0.01, 0.04, 0.03]).tolist() == [True, False, False]
    assert an.holm([0.01, 0.02, 0.03]).tolist() == [True, True, True]

def test_bootstrap_ci_contains_difference_and_uses_strata():
    rng = np.random.default_rng(5)
    blocks = np.repeat(np.arange(60), 4); labs = np.tile([0, 0, 1, 1], 60); strata = blocks % 4
    vals = rng.normal(0, 1, 240) + (labs == 0) * 0.5
    lo, hi = an.block_bootstrap_ci(vals, labs, blocks, strata, rng, B=3000)
    d = vals[labs == 0].mean() - vals[labs == 1].mean()
    assert lo < d < hi

# ------------------------------------------------------------------ data format, registry, drop-out
@pytest.fixture(scope='module')
def data():
    return sd.simulate_dataset(n_blocks=12, delta=0.12, dropout=0.3, never_start=0.1, seed=7, last_block_size=3)

def test_validate_accepts_simulated_data(data):
    assert an.validate(*data)

@pytest.mark.parametrize('breaker', ['q_out_of_range', 'q_non_integer', 'wrong_round', 'pid_unknown',
                                     'count_mismatch', 'version_mismatch', 'final_q0_differs', 'bad_status'])
def test_validate_rejects_impossible_data(data, breaker):
    reg, rd, sv = (x.copy() for x in data)
    if breaker == 'q_out_of_range': rd.loc[rd.index[0], 'q1'] = 10_000
    if breaker == 'q_non_integer': rd['q1'] = rd['q1'].astype(float); rd.loc[rd.index[0], 'q1'] += 0.5
    if breaker == 'wrong_round': rd.loc[rd.index[0], 'round'] = (rd.loc[rd.index[0], 'round'] + 1) % 36
    if breaker == 'pid_unknown': rd.loc[rd.index[0], 'pid'] = 99_999
    if breaker == 'count_mismatch': rd = rd.drop(rd.index[-1])
    if breaker == 'version_mismatch': reg.loc[reg.index[0], 'seq'] = 1 - reg.loc[reg.index[0], 'seq']
    if breaker == 'final_q0_differs':
        i = rd.index[rd['round'].isin(an.FINAL)][0]; rd.loc[i, 'q0'] = max(0, rd.loc[i, 'q1'] - 1) if rd.loc[i, 'q1'] > 0 else 1
    if breaker == 'bad_status': reg.loc[reg.index[0], 'status'] = 'gone'
    with pytest.raises(ValueError):
        an.validate(reg, rd, sv)

def test_registry_keeps_people_who_never_started(data):
    reg, rd, sv = data
    pt = an.person_table(reg, rd)
    ns = reg.loc[reg['status'] == 'never_started', 'pid']
    assert len(ns) > 0 and set(ns) <= set(pt['pid'])
    assert (pt.set_index('pid').loc[ns, 'n_normal'] == 0).all()
    t, _ = an.attrition_table(pt)
    assert t['assigned'].sum() == len(reg) == 12 * 6 - 3
    assert t['never_started'].sum() == len(ns)

def test_never_started_enter_bounds_and_tipping(data):
    reg, rd, sv = data
    pt = an.person_table(reg, rd)
    v = an.bounds_values(pt, 'A')
    ns_c = pt[(pt['status'] == 'never_started') & pt['group'].isin(['A', 'C'])]
    assert v[ns_c.index].notna().all()
    imp = an.imputed_all_assigned(pt, rd, 'A', 0.0)
    assert imp[pt['group'].isin(['A', 'C'])].notna().all()

def test_early_stopping_handled(data):
    reg, rd, sv = data
    pt = an.person_table(reg, rd)
    nc = pt[pt['started'].astype(bool) & ~pt['completer']]
    assert len(nc) > 0 and (nc['n_main'] < 30).all()
    assert (pt.loc[pt['completer'], 'n_normal'] == 18).all()

def test_tipping_point_is_monotone_and_finds_tip():
    reg, rd, sv = sd.simulate_dataset(n_blocks=30, delta=0.15, dropout=0.4, seed=11)
    pt = an.person_table(reg, rd)
    tp = an.tipping_point(pt, rd, 'A', np.random.default_rng(0), deltas=np.arange(0, 30.1, 2.0), P=400)
    diffs = [g['diff'] for g in tp['grid']]
    assert all(x >= y - 1e-9 for x, y in zip(diffs, diffs[1:]))            # larger delta -> smaller C advantage
    if tp['grid'][0]['p'] <= 0.05:
        assert tp['tipping_delta'] is not None

def test_additive_model_equals_means_on_complete_data(data):
    reg, rd, sv = data
    full = reg[reg['status'] == 'completed']
    rdf = rd[rd['pid'].isin(full['pid'])]
    eff = an.additive_person_effects(rdf)
    pt = an.person_table(full, rdf).set_index('pid')
    diffs = np.array([eff[p] - pt.loc[p, 'regret_normal'] for p in eff])
    assert np.allclose(diffs, diffs[0], atol=1e-6)

# ------------------------------------------------------------------ measures
def test_response_types_hand_cases():
    import pandas as pd
    i = an.NORMAL[0]; a = R[i]['a']
    rd = pd.DataFrame(dict(pid=range(6), round=[i] * 6, position=[1] * 6,
                           q0=[a - 20, a - 20, a - 20, a - 20, a - 20, a],
                           q1=[a - 20, a - 30, a - 10, a, a + 5, a + 3]))
    t = an.round_measures(rd)['rtype'].tolist()
    assert t == ['no_change', 'away', 'partial', 'exact', 'beyond', 'away']

def test_regret_of_recommendation_zero_in_normal_rounds(data):
    reg, rd, sv = data
    d = an.round_measures(rd)
    assert (d.loc[d['normal'], 'regret_rec'] == 0).all()
    assert (d.loc[d['hintr'], 'regret_rec'] > 0).all()

def test_studentized_test_null_size():
    rng = np.random.default_rng(8)
    blocks = np.repeat(np.arange(40), 4); labs = np.tile([0, 0, 1, 1], 40)
    rej = []
    for _ in range(300):
        v = np.where(labs == 1, rng.normal(0, 3, 160), rng.normal(0, 1, 160))   # unequal variances, equal means
        rej.append(an.block_perm_p(v, labs, blocks, rng, P=400, stat='welch') <= 0.05)
    assert np.mean(rej) < 0.085

def test_h4_reports_counts_and_participant_level_intervals(data):
    reg, rd, sv = data
    h4 = an.h4_analysis(rd, reg, np.random.default_rng(1), B=500)
    assert {'persons', 'pairs', 'pairs_after_unfav'} <= set(h4['counts'].columns)
    for g, v in h4['by_group'].items():
        if g in 'ABC': assert v['ci'][0] <= v['mean_drop'] <= v['ci'][1]

def test_full_analysis_returns_every_mapped_output(data):
    res = an.run_analysis(*data, P=300, B=300)
    assert set(res['decision']) == {'H1', 'H2'}
    need_sens = {'at_least_1_normal', 'at_least_9_normal', 'additive_model', 'pessimistic_bounds', 'studentized',
                 'mann_whitney', 'normalised_regret', 'smaller_scenarios', 'larger_scenarios', 'by_sequence', 'tipping_point'}
    assert need_sens <= set(res['sensitivity']['H1']) and need_sens <= set(res['sensitivity']['H2'])
    need_ex = {'B_vs_A_regret', 'reduction_first_to_final', 'final_rounds', 'hint_value', 'H4', 'H5',
               'woa_descriptive', 'pull_to_centre', 'confidence', 'time_rec', 'survey'}
    assert need_ex <= set(res['exploratory'])
    assert {'M', 'H3', 'M_interpretation', 'attrition', 'dropout_curve'} <= set(res)
    assert res['n_assigned'] == 12 * 6 - 3

def test_outcome_map_keys_exist(data):
    """Every key listed in OUTCOME_MAP.md is present in run_analysis output."""
    import re
    res = an.run_analysis(*data, P=200, B=200, tipping=True)
    txt = open(os.path.join(HERE, 'OUTCOME_MAP.md')).read()
    for key in re.findall(r'`res((?:\[\'[^\']+\'\])+)`', txt):
        obj = res
        for k in re.findall(r"\['([^']+)'\]", key):
            obj = obj[int(k)] if isinstance(obj, dict) and k.isdigit() and int(k) in obj else obj[k]

# ------------------------------------------------------------------ payment
def test_bonus_floor_and_partial_payment_single_rule():
    S, r, base = 5.82, 0.00225, 4.50
    assert an.payment(36, 4000, 'completed', S, r, base) == (base, 0.0)
    b, bon = an.payment(36, 1300, 'completed', S, r, base); assert b == base and abs(bon - (S - r * 1300)) < 1e-12
    b, bon = an.payment(18, 700, 'technical', S, r, base)
    assert abs(b - base / 2) < 1e-12 and abs(bon - (S / 2 - r * 700)) < 1e-12
    b, bon = an.payment(18, 700, 'withdrew', S, r, base); assert abs(b - base / 2) < 1e-12 and bon == 0.0
    assert an.payment(3, 5000, 'technical', S, r, base)[1] == 0.0
    assert an.payment(0, 0, 'never_started', S, r, base) == (0.0, 0.0)
    with pytest.raises(TypeError):
        an.payment(18, 700, 'technical', S, r, base, share_time=0.7)    # no second definition of 'share'
