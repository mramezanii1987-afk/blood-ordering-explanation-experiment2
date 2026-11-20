# Blood-ordering explanation experiment: analysis and simulation code

Code supplement to the study plan *From Optimisation to Decisions* (draft v9). The study is an incentivised online experiment in which participants play a simulated ordering game with a short-shelf-life product and receive a computer recommendation with no explanation (group A), a calculation (group B) or an economic reason (group C). Regret, the extra expected cost of an order compared with the best order, is the main outcome.

This repository contains the game definition, the pre-specified analysis, the simulations used to plan the sample size and the bonus, and automated tests. **It contains no participant data:** all data used here are simulated.

## Installation

```
pip install -r requirements.txt
```


Environment used: Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, pytest 9.1.1 (Linux, 2 cores). All random seeds are fixed in the code or in the commands below.

## Files

| File | What it does |
|---|---|
| `game.py` | Scenarios, hints, demand probabilities, recommendation, best order, regret; fixed rule choosing the two demand sequences; four versions and rotation of blocks over versions (85 blocks → 22/21/21/21; 43 blocks in order 1, 42 in order 2); unfavourable-round labels (H4). |
| `reference_check.py` | Independent pure-Python reference implementation and pre-set checks. |
| `analysis.py` | Pre-specified analysis on three tables (assignment register, rounds, questionnaire): data checks, main tests with fixed order H1 → H2, five drop-out sensitivity analyses (incl. tipping point for all assigned), studentized and Mann–Whitney checks, secondary and exploratory outputs, payment rule. `run_analysis(registry, rounds, survey)` runs everything. |
| `OUTCOME_MAP.md` | Every measure and analysis in the plan → function → input columns → output key → interval type. |
| `DATA_DICTIONARY.md` | Columns, types and allowed values of the three tables; rules enforced by `analysis.validate`. |
| `simulate_data.py` | Simulated data in the three-table format (people who never start, drop-outs, incomplete last block). |
| `behaviour.py` | Assumed behavioural model for the simulations (not data). |
| `power_sim3.py` | Main power simulation (static behaviour, random drop-out) using the analysis functions. |
| `power_sensitivity.py` | Sensitivity simulations: learning, fatigue, reliance shock after own bad outcome, performance-dependent drop-out. |
| `payout_sim.py` | Bonus calibration and bonus by strategy. |
| `tests/test_all.py` | 74 automated tests. |

## Commands, outputs and running times (on the machine above)

| Command | Output | Time |
|---|---|---|
| `python -m pytest -q tests` | 74 passed | about 10 s |
| `python reference_check.py` | ALL CHECKS PASSED | about 1 s |
| `python payout_sim.py payout.json` | `payout.json` | about 1 s |
| `python power_sim3.py power3.json 1 1000` | `power3.json` (reproduced exactly) | about 3.5 min |
| `python power_sensitivity.py power_sensitivity.json 500 5` | `power_sensitivity.json` | about 7 min |
| `python simulate_data.py sim 85 0.12 1` then `analysis.run_analysis(...)` | dry run of the full analysis | about 10 s with P = B = 2,000 |

## Main results (simulated)

- False-positive rate with no true differences: 4.9% (2,000 studies). Power for H1: 80% at a 23% regret reduction, 93% at 28%. H1 and H2 together: 89% at 28% if B is like A.
- Sensitivity (500 studies each, 28% reduction, B like A): H1 85–97% with learning, fatigue, reliance shock or all together; 84–97% with performance-dependent drop-out, where the completer comparison understated the all-assigned difference by 10–32%.
- Provisional bonus: S = £5.82, r = £0.00225 per point; average simulated bonus £2.50 with either demand sequence.

## SHA-256 checksums
    fa9334023656180d5ccdd59367f27250179ed26f4757d5787a36f3a1abb08411  analysis.py
    9f476972f8e12ea0e80ec8fb110a08708017c34e63b6d7bb75ea88fd37cc7f8a  behaviour.py
    13a9f3ade72e146e0b75ba942fc5c965c1aa24706410657013d915120a162594  game.py
    b1c64235c433f0a39e658fe2c8a121ae2b3a2e0a7c3583adf3185922923a2a46  payout_sim.py
    5c0d5aa65daa40df0c6e43239df013fb663b68bab45893100bee5e97a879dde4  power_sensitivity.py
    00eb8275ed103be99b3dcd90327d241d95b51aeb5b73dcf41e91aaac22b0915e  power_sim3.py
    31ccdc979cd28403fc68dc4f6b84c2b7215064f115dc7ed34b3131a3ddb1f8c6  reference_check.py
    396b5b131687e40d75ec92e7116c5e67e5b50ad89072b14bf5e5f386ffbea1cb  simulate_data.py
    fb03730c22c0655e08fceffbeb8c712d4b3637002f841dcf371a88c7e552a562  tests/test_all.py
    cb98b92579c737379f855abd98da81f3b94b2d859bab7262b86db123c24b92b2  OUTCOME_MAP.md
    0acfc6bf68ad71cfe4968fd5497ba34a4ae933ea3fe1dd7dc124ae5810f6c58d  DATA_DICTIONARY.md
    052c96122ca49d8c4f41a35f6a78b665d86b2e9654583a363efb749efd9392a1  payout.json
    d2d049c3ad271cc558b6ad5f7c026930bd3d4d6955d1a6d5e100959a0844c585  power3.json
    908a4161861adbe371416751fd78ca59bff4930f4b726af93a463553c50232f1  power_sensitivity.json

## Added 20 November 2026: power for smaller effects

After reading Joslyn & LeClerc (2012), whose explanations reduced expected loss by only about 5 per cent compared with advice alone, H1 power was simulated for smaller effects with the same model, test and drop-out (group B like A, 500 studies per row).

| Command | Output | Time |
|---|---|---|
| `python power_small.py power_small.json 7 500` | power_small.json | about 2 min |
| `python power_small2.py power_small2.json 11 500` | power_small2.json | about 2 min |

| Reduction in regret | Per group | Power H1 |
|---|---|---|
| about 13% | 170 | 37% |
| about 13% | 560 | 78% |
| about 13% | 680 | 84% |
| about 15% | 170 | 47% |
| about 15% | 380 | 76% |
| about 15% | 440 | 81% |

The reductions (12.8% and 14.7%) were estimated separately with 3 × 100,000 simulated participants per setting; the `reduction` values stored in power_small.json come from a single smaller run and are noisier.

## Licence

MIT; see `LICENSE`.
