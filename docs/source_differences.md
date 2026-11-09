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
| 6 | Lookup tables (trip purpose, trip departure, tour-frequency extension) estimated on NTS | ActivitySim | **Kept on NTS**, as decided. This is how the source tables were made, for MTC's survey. Their keys are now the source tables' (November 2026, below) |
| 7 | Rules added for the minute-level output | both | **Removed:** the neighbouring-purpose exclusion for stops, and the 10-minute home buffer between tours (now 1 minute). DaySim's nearest-feasible mapping of out-of-range days was replaced by leaving them out of estimation. DaySim re-simulates failed person-days, as its source does |
| — | `access_egress_distance` looked like an outcome (85% of its `unknown` days stayed at home) | all models | **Replaced** by `wt_travel_time_to_cities`, which stands in for the sources' accessibility terms |
| — | Calibration (off in the reported runs, but in the code) | all models | **Removed** from every model, including MDCEV's constant adjustment |
| — | Post-processing of the output: neighbouring episodes of one type merged, zero-length episodes dropped, ActivitySim stops that did not fit at whole minutes dropped | all models (merging); source-aligned (the rest) | **Removed.** Every generated episode is written |
| — | Time precision finer than the source: ActivitySim drew a minute inside each hourly period | ActivitySim | **Removed.** Each variant keeps its source's precision (below) |
| — | Minimum home stay between tours (10 minutes CEMDAP, 1 minute DaySim and ActivitySim) and window margins at the day's edges | source-aligned | **Removed.** None of the sources has one |
| — | CEMDAP drew the work start before the work duration, measured from 03:00 and clipped earlier starts | CEMDAP | **Fixed.** As CEMDAP, the work-based duration is drawn first and the start given it, from the start of the day |
| — | Work or education interrupted by other activities was merged into one block before estimation | source-aligned | **Removed.** DaySim and ActivitySim read the other episodes as work / school stops, which both have; CEMDAP reads them as work-based tours, which it has. DaySim's work and school stop purposes and its stop-availability rules by tour purpose were restored with them |

## ActivitySim, checked against its code in November 2026

The ActivitySim variant was checked against the v1.6.0 code (`core/timetable.py`,
`abm/models/trip_scheduling.py`, `trip_purpose.py`,
`non_mandatory_tour_frequency.py`) as well as the `prototype_mtc` configs.
Every difference found made the variant fit NTS more closely than the source
would:

| Difference found | Now |
|---|---|
| An extra 00:00–05:00 period (20 periods, 210 pairs) | `prototype_mtc`'s 19 hourly periods 5–23 (190 pairs); earlier times fall in the 05:00 period, as in ActivitySim |
| Every person type had departure and duration shifts in all three scheduling models (about 30 terms more than the spec files) | Only the person-type rows the spec files have |
| Wrong reference constants: school duration, work arrival, both non-mandatory sets | The constants the coefficient files fix at 0 (10 of the escort set's 27) |
| Home-made timetable terms: gaps to the adjacent tours in fractions of a day, a look-ahead term for every tour; a tour ending where the next began was missed | ActivitySim's rows on its period timetable: previous tour ends / begins in the period, free periods adjacent before / after for the first of 2+ and later tours, tours still to schedule / periods left as each spec defines it |
| Trip departures: probability outside the feasible periods renormalised | As `trip_scheduling.py`: it is a failure, the leg is drawn again up to 100 times, then `choose_most_initial`. `logic_version: 2`, which fixes a version-1 bug (`prototype_mtc` sets no version, so ActivitySim falls back to 1) |
| Trip departure keys: a mandatory / non-mandatory class key, inbound trips numbered from the tour end | `trip_scheduling_probs.csv`'s keys: purpose (with `univ`), direction, tour hour, trip number from the start of each leg |
| Trip purpose keys: three hour bands for every purpose, inbound keyed on the tour end | `trip_purpose_probs.csv`'s keys: only work tours split by hour (5–8 / 9–23 outbound, 5–14 / 15–23 inbound), always on the tour start |
| Extension step applied to every purpose with a tour | As `extend_tour_counts`: only purposes at their maximum, for people with 1–4 tours |
| Residual window for non-mandatory frequency in free minutes (including 00:00–05:00) | `log1p` of the longest run of free periods (`max_time_block_available`) |
| Adjacent-window terms counted the free periods next to the tour | A 0/1 flag, as the specs define them (`adjacent_window_before(...) > 0`) |
| A second mandatory tour, or a second tour of one non-mandatory purpose, could be placed before the first | As the specs' "subsequent tour must start after previous tour ends" rows (−100 / −999): it cannot start before the previous tour's end period. 1 NTS day breaks the rule; its observed choice is kept and counted (`forced_available`) |

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
| ActivitySim tour scheduling, work / school / non-mandatory | 60 / 53 / 77 | 62 / 56 / 87 | 53 / 52 / 74 |
| ActivitySim stop frequency, per purpose | about 27 | 138 | 13–19 |
| DaySim day pattern | about 400 | 661 | 277 |
| DaySim exact number of tours, per purpose | about 44 | 108 | 35–36 |
| DaySim tour time, work / school / other | 128 / 128 / 132 terms (many need mode or logsums) | 76 / 76 / 92 | 76 / 76 / 92 |
| DaySim stop generation | 106 | 350 | 102 |
| DaySim trip time | 75 | 62 | 66 |

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
* No at-work subtours in DaySim and ActivitySim: an interrupted work episode
  is a work stop of the work tour instead ([tour_rules.md](tour_rules.md)).
  CEMDAP has its work-based tours.

**Choices made here:**

* Day of week added (Saturday and Sunday dummies). Both sources model an
  average weekday; NTS covers every day.
* Days whose observed choice the source would not allow are left out of that
  model's estimation (counted in `validation.json`). For example, 321 NTS
  days of non-working adults or retirees with work or education are left out
  of ActivitySim's CDAP. 1,253 days are left out of DaySim's day pattern:
  1,202 break its worker / student rule and 51 its pattern limits. Such days
  can no longer be generated. Likewise 2,281 observed stops on tours that
  DaySim's stop rules do not allow them on (mostly escort stops on shopping or
  social tours) are left out of DaySim's stop-generation estimation.
* ActivitySim's trip purpose, trip departure and extension tables are
  tabulated from NTS (point 6 above), with the source tables' keys; cells with
  fewer than 20 NTS observations fall back to coarser keys.
* Time precision is the source's: CEMDAP continuous (rounded to whole
  minutes when written), DaySim half-hour choices then a uniform free minute
  (as its `GetAvailableMinute`), ActivitySim hourly periods 05:00–23:00 with
  nothing finer. ActivitySim's periods are written at hh:30, so events in one
  period are zero minutes apart.
* No minimum home stay between tours; a zero-length home stay is still
  written between two tours.
* Every generated episode is written: zero-length ones, and two neighbouring
  episodes of one type, as the sources would keep them (at different places).
  ActivitySim's own fail-fix handles infeasible departures.

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
