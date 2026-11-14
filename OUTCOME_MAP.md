# Outcome map: study plan → code → output

Every measure in Table 5 and every analysis in Section 7 of the study plan, with the function that computes it, the input columns, the key in the dictionary `res = analysis.run_analysis(registry, rounds, survey)` and the type of interval. The automated test `test_outcome_map_keys_exist` checks that every key below exists.

Interval types: **BB** = 95% percentile bootstrap resampling whole blocks within each of the four versions; **PB** = 95% percentile bootstrap resampling participants within group (H4 only). Tests: **RT** = blocked randomisation test (labels permuted within the original blocks, ties count against), 10,000 permutations.

## Confirmatory (conditional on completing all 30 main rounds)

| Plan | Measure | Function | Input columns | Output key | Interval / test |
|---|---|---|---|---|---|
| H1 | Average regret, normal rounds, A − C | `person_table`, `compare` | rounds.q1, round | `res['H1']` | BB; RT one-sided |
| H2 | Average regret, normal rounds, B − C | `person_table`, `compare` | rounds.q1, round | `res['H2']` | BB; RT one-sided |
| H1→H2 | Fixed testing order | `fixed_sequence` | p-values | `res['decision']` | α = 0.05 each, H2 only if H1 |

## Drop-out and other sensitivity analyses (for H1 and H2; shown for H1)

| Analysis | Function | Output key | Interval / test |
|---|---|---|---|
| ≥ 1 normal round | `compare` | `res['sensitivity']['H1']['at_least_1_normal']` | BB; RT |
| ≥ 9 normal rounds | `compare` | `res['sensitivity']['H1']['at_least_9_normal']` | BB; RT |
| Two-way additive model (person + round), all observed normal rounds | `additive_person_effects` | `res['sensitivity']['H1']['additive_model']` | BB; RT |
| Pessimistic bounds (all assigned, incl. never started) | `bounds_values` | `res['sensitivity']['H1']['pessimistic_bounds']` | BB; RT |
| Tipping point, all assigned (MAR given group and round, shifted by δ in C) | `tipping_point` | `res['sensitivity']['H1']['tipping_point']` | RT for each δ |
| Studentized (Welch) randomisation test | `block_perm_stats(stat='welch')` | `res['sensitivity']['H1']['studentized']` | BB; RT |
| Mann–Whitney | `mann_whitney` | `res['sensitivity']['H1']['mann_whitney']` | one-sided p |
| Regret ÷ expected cost of best order | `compare` | `res['sensitivity']['H1']['normalised_regret']` | BB; RT |
| Smaller scenarios (mean ≤ 100) | `estimate` | `res['sensitivity']['H1']['smaller_scenarios']` | BB |
| Larger scenarios (mean > 100) | `estimate` | `res['sensitivity']['H1']['larger_scenarios']` | BB |
| By demand sequence | `estimate` | `res['sensitivity']['H1']['by_sequence']['0']`, `res['sensitivity']['H1']['by_sequence']['1']` | BB |
| Drop-out by group × order × sequence; stopping round | `attrition_table` | `res['attrition']`, `res['dropout_curve']` | counts |

## Secondary (Holm within family)

| Plan | Measure | Function | Output key | Interval / test |
|---|---|---|---|---|
| M | Within-round revision, A − C and B − C | `compare` on −revision | `res['M']['A']`, `res['M']['B']` | BB; RT one-sided |
| M | Distances of first and final orders from the recommendation, by group and by round | `revision_by_round` | `res['M_interpretation']` | descriptive |
| H3 | Regret in hint rounds, A − C | `compare(two_sided=True)` | `res['H3']` | BB; RT two-sided |
| — | Holm decisions (M vs A, M vs B, H3) | `holm` | `res['secondary_holm_reject']` | — |

## Exploratory and additional comparisons (estimates and intervals only)

| Table 5 row | Function | Output key | Interval |
|---|---|---|---|
| Reduction in regret from first to final order | `estimate` | `res['exploratory']['reduction_first_to_final']` | BB |
| B vs A, regret | `compare(ref='B', two_sided=True)` | `res['exploratory']['B_vs_A_regret']` | BB (p reported, not a claim) |
| Regret without the computer (final rounds) | `estimate` | `res['exploratory']['final_rounds']` | BB |
| Regret of first order, recommendation, final order; share where final beats both | `person_table` | `res['exploratory']['hint_value']` | means; BB for share |
| WOA after unfavourable vs favourable round; persons and pairs by version | `h4_analysis` | `res['exploratory']['H4']` | PB |
| Shares of response types (no change, away, partial, exact, beyond) | `round_measures`, `estimate` | `res['exploratory']['H5']` | BB |
| WOA raw and limited to 0–1; share where WOA cannot be calculated | `person_table` | `res['exploratory']['woa_descriptive']` | means |
| Pull towards average demand (normal and final rounds) | `round_measures` | `res['exploratory']['pull_to_centre']` | means |
| Confidence (0–100) | `estimate` | `res['exploratory']['confidence']` | BB |
| Time on the recommendation screen | `estimate` | `res['exploratory']['time_rec']` | BB |
| Trust and perceived accuracy after rounds 10, 20, 30 | `survey_summary` | `res['exploratory']['survey']['trust_10']`, `res['exploratory']['survey']['accuracy_30']` | BB |
| Understanding of the trade-off | `survey_summary` | `res['exploratory']['survey']['understanding']` | BB |

Not implemented as code: the reader test of the explanation texts and the pilot report, which use separate small data sets and are described in the plan.
