# Data dictionary

The analysis reads three tables. `analysis.validate()` checks every rule below and stops with a list of problems; impossible values never reach the regret calculation.

## 1. `registry` — one row per assigned participant

Written by the experiment software at the moment of group assignment, before the first round. People who are assigned but never play a round stay in this table.

| Column | Type | Allowed values / rule |
|---|---|---|
| `pid` | integer | unique |
| `block` | integer | randomisation block, 0-based |
| `group` | text | `A`, `B`, `C`; at most two of each per block |
| `order` | 0/1 | scenario order; must equal `game.version_of_block(block)[0]` |
| `seq` | 0/1 | demand sequence; must equal `game.version_of_block(block)[1]` |
| `assigned_at` | timestamp | time of assignment |
| `started` | boolean | played at least one round; false only if `status == never_started` |
| `rounds_played` | integer 0–36 | must equal the number of rows in `rounds`; 36 if and only if `completed`; 0 if and only if `never_started` |
| `status` | text | `completed`, `technical` (stopped by a technical problem), `withdrew` (chose to stop), `never_started` |
| `exit_at` | timestamp | time of completion or exit |

## 2. `rounds` — one row per participant and round played

| Column | Type | Allowed values / rule |
|---|---|---|
| `pid` | integer | must exist in `registry` |
| `round` | integer 0–35 | scenario index in `game.ROUNDS`; must equal `game.scenario_orders()[order][position − 1]` |
| `position` | integer | play position 1..`rounds_played`, consecutive |
| `q0` | integer | first order, 0 ≤ q0 ≤ U of the round (U = ceil(mean + 5 SD)); in final rounds equal to `q1` |
| `q1` | integer | final order, 0 ≤ q1 ≤ U; determines points |
| `confidence` | 0–100 | main rounds only; optional column |
| `time_rec` | seconds | time on the recommendation screen, main rounds only; optional column |

The experiment software must enforce the same limits on input (whole numbers from 0 to U), so that invalid orders cannot be entered.

## 3. `survey` — one row per participant who answered (optional)

| Column | Type | Allowed values |
|---|---|---|
| `pid` | integer | must exist in `registry` |
| `trust_10`, `trust_20`, `trust_30` | 1–7 | “How much do you trust the system’s recommendations?” after rounds 10, 20, 30 |
| `accuracy_10`, `accuracy_20`, `accuracy_30` | 1–7 | “How accurate do you think they are?” |
| `understanding` | integer | number of correct answers on the trade-off questions after the game |

## Derived measures (computed, not stored)

Regret, regret of the first order and of the recommendation, regret ÷ expected cost of the best order, distance of first and final orders from the recommendation, within-round revision, weight on advice (raw, limited to 0–1, and whether it can be calculated), response type, pull towards average demand, unfavourable-round flag. Definitions are in `analysis.round_measures()` and `game.py`.
