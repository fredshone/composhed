# How the DaySim and ActivitySim variants differ from the source models

This page compares the [DaySim](daysim.md) and [ActivitySim](activitysim.md)
variants with the published implementations, and lists every remaining
difference. It was written in October 2026, after a review found that both
variants had been fitted to NTS in ways the sources are not. The models were
then rebuilt from the sources' own terms (see
[What changed](#what-changed-in-october-2026)).

**Sources checked:**

* **ActivitySim** v1.6.0, `activitysim/examples/prototype_mtc/configs`. The
  CDAP, tour-frequency and stop-frequency spec and coefficient files are
  vendored in `composhed/activitysim/specs/` and read as they are. Also
  checked:
  * `tour_scheduling_*.csv` (transcribed in `activitysim/tod_spec.py`);
  * `trip_purpose_probs.csv`;
  * `trip_scheduling_probs.csv`;
  * `tour_departure_and_duration_alternatives.csv`.
* **DaySim**, RSGInc/DaySim on GitHub, `DaySim/ChoiceModels/Default/Models`:
  * `IndividualPersonDayPatternModel`;
  * `PersonExactNumberOfToursModel`;
  * `WorkTourTimeModel`, `SchoolTourTimeModel` and
    `OtherHomeBasedTourTimeModel`;
  * `IntermediateStopGenerationModel`;
  * `TripTimeModel`;
  * `ChoiceModelRunner`.

  Their terms are transcribed in `daysim/models.py` and `daysim/tod_spec.py`,
  with DaySim's term numbers in the comments.

## Summary

The variants now take three things from the sources:

* **structure:** the chain of models, its order and the alternatives;
* **utility terms:** each term on the alternatives the source gives it;
* **availability rules.**

Where NTS has a source variable, a counterpart is built from the NTS labels.
NTS is often coarser than the source, for example income bands for income in
dollars. Where NTS has nothing, the term is left out. **Day of week is the only
variable the sources do not have.** Every coefficient is re-estimated on NTS,
and nothing is calibrated.

The labels are no longer covariates in their own right. A label enters a model
only through the variables built from it. For example, `hh_zone` enters as
"home is urban", and only in the terms where the source uses urban homes.

## What changed in October 2026

| # | Difference found | Who | Now |
|---|---|---|---|
| 1 | All 8 NTS labels (35 dummies) as covariates in every choice model | both | **Fixed.** Each model has the source's variables only, built from the labels, plus day of week (below) |
| 2 | Separate CDAP and tour-frequency models per person type | ActivitySim | **Fixed.** As `prototype_mtc`, only non-mandatory tour frequency has one coefficient set per person type; CDAP and mandatory frequency are one model each with person-type terms |
| 3 | Covariates on every alternative; 3–20 times the source's parameter count | both | **Fixed.** Each term acts on the alternatives the source gives it; counts are now at or below the source's (table below) |
| 4 | Saturday and Sunday terms in timing | both | **Kept, as the one allowed addition**, and used the same way everywhere: Saturday and Sunday dummies on the alternatives of each choice model, and as shift terms in timing |
| 5 | Person-type availability rules dropped | both | **Restored.** ActivitySim: no M days for non-working adults and retirees, and the mandatory-frequency availability rows. DaySim: work tours only for workers, school tours only for students |
| 6 | Lookup tables (trip purpose, trip departure, tour-frequency extension) estimated on NTS | ActivitySim | **Unchanged**, as decided. This is how the source tables were made, for MTC's survey |
| 7 | Rules added for the minute-level output | both | **Removed:** the neighbouring-purpose exclusion for stops, and the 10-minute home buffer between tours (now 1 minute). DaySim's nearest-feasible mapping of out-of-range days was replaced by leaving them out of estimation. DaySim re-simulates failed person-days, as its source does |
| — | `access_egress_distance` looked like an outcome (85% of its `unknown` days stayed at home) | all models | **Replaced** by `wt_travel_time_to_cities`, which stands in for the sources' accessibility terms |
| — | Calibration (off in the reported runs, but in the code) | all models | **Removed** from every model, including MDCEV's constant adjustment |

## Variables: source to NTS

**ActivitySim** (`activitysim/sourcevars.py`):

| ActivitySim variable | From NTS |
|---|---|
| `ptype` | person type from age and employment (no part-time workers) |
| `age` | midpoint of the age band |
| `sex`, `female` | `sex` |
| `income_in_thousands` and income dummies | `hh_income` band rank → 10, 30, 45, 75, 150 |
| `auto_ownership` against workers / drivers | `veh_per_adult`: 0, 0.5, 1.5 cars against 1 |
| `home_is_urban` | `hh_zone == urban` |
| `auPkTotal`, `auOpRetail` | `wt_travel_time_to_cities` (3 − band rank) |
| joint tours, at-work subtours | 0 |

**DaySim** (`daysim/sourcevars.py`):

| DaySim term | From NTS |
|---|---|
| person type flags | person type |
| income 0–25k, 25–45k, 75k+ | `hh_income` band rank 0, 1, 4 |
| cars per driver | `veh_per_adult`: 0, 0.5, 1 |
| female adult | female, 18+ (DaySim splits it by household children) |
| age 18–25, 26–35, 51–65 | bands 18-27, 27-35, 48-62 |
| mixed-use density | `hh_zone == urban` |
| purpose and composite logsums | `wt_travel_time_to_cities` |
| adult male | male, 18+ |

**Both:** `saturday` and `sunday` from `day`.

**Dropped, for lack of an NTS counterpart:**

* household size and composition, children and household interactions;
* distances and travel times to work or school;
* tour mode;
* walk and transit accessibility;
* parcel densities other than the urban flag;
* works at home, transit pass, parking;
* joint tours.

ActivitySim rows are dropped automatically and listed per model in
`validation.json` (`dropped_rows`).

## Parameter counts

Estimated coefficients, NTS 2023 seed 0. "Source" counts the distinct
coefficients a spec file uses (non-zero ones for non-mandatory frequency), or
the utility terms in the DaySim code. "Before" is the version reviewed in
October 2026.

| Model | Source | Before | Now |
|---|---|---|---|
| ActivitySim CDAP | 54 (plus household interactions) | 504 | 40 |
| ActivitySim mandatory tour frequency | 51 | 1,008 | 31 |
| ActivitySim non-mandatory tour frequency, per person type | about 100 | 468 | 24–69 |
| ActivitySim tour scheduling, work / school / non-mandatory | 60 / 53 / 77 | 62 / 56 / 87 | 62 / 56 / 87 |
| ActivitySim stop frequency, per purpose | about 27 | 138 | 13–19 |
| DaySim day pattern | about 400 | 661 | 258 |
| DaySim exact number of tours, per purpose | about 44 | 108 | 33–34 |
| DaySim tour time, work / school / other | 128 / 128 / 132 terms (many need mode or logsums) | 76 / 76 / 92 | 76 / 76 / 92 |
| DaySim stop generation | 106 | 350 | 84 |
| DaySim trip time | 75 | 62 | 62 |

Every model is now at or below its source's size. The gap is the terms whose
variables NTS lacks.

## Remaining differences

**Forced by the data:**

* No locations, modes, travel times or logsums. Moving takes no time.
* One person-day: no household interactions, joint tours or school escorting.
* Person types from binned age and employment. There are no part-time
  workers, and the 0–11 and 11–18 bands straddle the sources' age limits.
* Coarser variables, so source thresholds fall inside NTS bands: age, income,
  and cars per adult rather than per worker or driver. One accessibility index
  stands in for every logsum and accessibility measure.
* No eat-out (ActivitySim) or meal (DaySim) purpose. In DaySim, NTS `visit`
  takes the meal slot.
* No work or school stops and no at-work subtours. Work interrupted by other
  activities is merged into one block ([tour_rules.md](tour_rules.md)).

**Choices made here:**

* Day of week added (Saturday and Sunday dummies). Both sources model an
  average weekday; NTS covers every day.
* Days whose observed choice the source would not allow are left out of that
  model's estimation (counted in `validation.json`). For example, 321 NTS
  days of non-working adults or retirees with work or education are left out
  of ActivitySim's CDAP. 1,245 days are left out of DaySim's day pattern:
  1,198 break its worker / student rule and 47 its pattern limits. Such days
  can no longer be generated.
* ActivitySim's trip purpose, trip departure and extension tables are
  tabulated from NTS (point 6 above).
* Minutes are drawn uniformly within the chosen period, with at least 1 minute
  at home between tours.
* ActivitySim stops that do not fit their tour at whole-minute resolution are
  dropped and counted (about 300 per run). ActivitySim's own fail-fix handles
  departures.
* Two neighbouring episodes of one type are written as separate episodes,
  as the sources would keep them (at different places). Nothing is merged.

## In-sample fit

A logit with alternative-specific dummies reproduces, in its estimation data,
each alternative's share within every value of those dummies. Before the
rebuild this held for every value of every label. Now it holds only for the
variables the source uses:

* person type × each alternative, where the source has person-type
  constants;
* Saturday and Sunday;
* a few income and car bands, where a source term has them alone.

It no longer holds for zone, access, or the age and income bands inside a
person type.
