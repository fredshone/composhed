# DaySim-aligned variant

**Code:** `composhed/daysim/`, which builds on `composhed/common/` (see [what the source-aligned variants share](source_aligned.md))

DaySim (Bowman & Ben-Akiva, 2001; Bradley, Bowman & Griesenbeck, 2010) plans a
day around tours: trips that leave home and come back. This variant follows the
person-day part of DaySim's model chain. Each component's form was taken from
the DaySim source code (RSGInc/DaySim on GitHub, `DaySim/ChoiceModels/Default`;
checked October 2026). The utility terms are DaySim's own, transcribed from
the C# code; terms whose variables NTS lacks are left out, day of week is the
only addition, and every coefficient is re-estimated on NTS without
calibration ([how the variants differ from the sources](source_differences.md)).
The chain runs in DaySim's order (`ChoiceModelRunner`):

1. **Day pattern.** One choice says which tour purposes the day has and which
   stop purposes it has.
2. **Exact number of tours.** For each tour purpose in the pattern: 1, 2 or 3
   tours.
3. **For each tour, in priority order** (work, education, then the rest):
   1. **Tour time.** The arrival and departure half-hour **at the primary
      destination**, chosen together from the time earlier tours left free.
   2. **Stops**, one half-tour at a time, working outwards from the primary
      destination: the next stop's purpose (or no more stops), then the
      half-hour in which that stop begins (outbound) or ends (inbound).
   3. The whole tour is then marked as taken in the time window.

Any time not spent on a tour is home. Exact minutes are drawn uniformly from
the free minutes of each chosen half-hour, as DaySim's
`TimeWindow.GetAvailableMinute` does, so DaySim's own precision is kept:
half-hour choices, then minutes. A tour may start in the minute the previous
one ends (a zero-length home stay). As in DaySim, a person-day in which a tour
or stop cannot be placed is simulated again.

## Running it

```bash
uv run composhed-train-daysim --attributes A.parquet --schedules S.parquet --output-dir models/ --seed 0
uv run composhed-generate-daysim --attributes A.parquet --models models/daysim_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

Options beyond the [shared ones](source_aligned.md#running-them):

| Option | Default | Effect |
|---|---|---|
| `--time-resolution {15,30,60}` | 30 | Length of a time period in minutes (DaySim: 30) |

Training takes about 25 minutes on NTS 2023 when run alone, most of it in the
tour-time models; `scripts/time_models.py` measures it.

## Inputs

DaySim's own person terms, built from the NTS labels where NTS has a
counterpart (`sourcevars.py`):

| DaySim term | From NTS |
|---|---|
| person type (8 flags; full-time worker reference) | person type (7 types, see the [ActivitySim page](activitysim.md#components); no part-time workers) |
| household income 0–25k, 25–45k, 75k+ | `hh_income` band rank 0, 1, 4 (ranks 2–3 are the reference) |
| cars per driver | `veh_per_adult`: 0, 0.5, 1 |
| female adult (DaySim splits it by children in the household) | female, 18 or over (NTS has no household children: one term) |
| age 18–25, 26–35, 51–65 | age bands 18-27, 27-35, 48-62 |
| mixed-use density | `hh_zone == urban` |
| purpose and composite logsums | `wt_travel_time_to_cities`: 3 − band rank |
| adult male (stop generation) | male, 18 or over |
| Saturday, Sunday | `day` (added: DaySim models a weekday) |

Left out: only adult, only worker, works at home, intersection density,
transit pass, parking, household size and children, tour mode. The labels
themselves are not covariates. Workers (`employment = employed`) and students
(`employment = student`, or a school-age child) define which tour purposes
are available, as in DaySim.

## Components

"Fidelity" says how closely each component follows DaySim:

* **full:** same structure and model form;
* **simplified:** same idea, reduced form;
* **absent:** not modelled.

| DaySim model | What it decides | Model here | Code | Fidelity | Differences |
|---|---|---|---|---|---|
| `IndividualPersonDayPatternModel` | Whether the day has a tour of each of the 7 purposes and a stop of each of the same 7 purposes | Logit over all 2,080 feasible patterns, as DaySim (≤ 3 tour purposes, ≤ 4 stop purposes, ≤ 5 in all, stops only with a tour, work or school stops only with a work or school tour and at most one of them), with DaySim's terms: tour and stop constants; tour–stop (every tour purpose with every stop purpose but social / recreation; work and school stops only with work and school tours), tour–tour, stop–stop and tour-count × stop-count constants; the person terms on each purpose's presence; one person vector × log(tour purposes) when 2+ and one × log(stop purposes) when 2+; the non-mandatory-tour and non-mandatory-stop components (cars, density, density × child 5–15, logsum); logsum on each tour and stop. Saturday and Sunday on each purpose's presence. DaySim's availability: work tours and stops only for workers, school tours and stops only for students. | `models.DaysimModels.pattern` | simplified | NTS days outside DaySim's choice set (the worker / student rule or its limits) are left out of this model's estimation and counted in `validation.json`; their tours are used by the later models. |
| `PersonExactNumberOfToursModel` | 1, 2 or 3+ tours of each purpose in the pattern | Logit per purpose with DaySim's terms: the person terms and the day's tours and stops on 2 and 3+ (one coefficient for both), a constant and a logsum term for each; Saturday and Sunday added | `.exact` | full | Tours beyond 3 of one purpose are dropped from the training data (counted). |
| Tour priority | Order in which tours are scheduled | Fixed: work → education → escort → other (personal business) → shop → visit → leisure (social / recreation; DaySim's meal purpose has no NTS equivalent) | `data.priority_order` | full | |
| `WorkTourTimeModel`, `SchoolTourTimeModel`, `OtherHomeBasedTourTimeModel` | Arrival and departure period at the primary destination (1,176 pairs at 30 minutes) | Conditional logit with DaySim's grouped arrival, departure and duration constants; arrival and duration shift terms by person type, tour purpose, income, the four priority flags, stop purposes in the pattern and escort stops; time-band interactions; and the time-window terms (log free minutes in the arrival and departure periods, adjacent free time for later tours, stops and tours still to place per free hour). Separate models for work, education and other tours. | `common.tod.TourTOD` with `daysim.tod_spec` | simplified | No mode impedances, transit or subtour terms. Saturday and Sunday shift terms added (DaySim models a weekday). |
| Time windows | Which pairs and periods are still possible | Time taken by earlier tours is removed from the choice set; a tour may start in the minute another ends | `common.timewindow.TimeWindow` | full | |
| `IntermediateStopGenerationModel` | The next stop's purpose, or no more stops | Logit over none + work, education (school), escort, other (personal business), shop, visit (DaySim's meal slot), leisure (social / recreation). As DaySim, a purpose is available only if the pattern has it and the tour allows it: work and school stops only on work and school tours, escort stops only on work, school and escort tours. Observed stops the source would not allow are left out of estimation (counted). DaySim's terms, each on the alternatives it has there: trip number × half-tour, tours in the day, tours already simulated, before a mandatory destination and school tour on "none"; per stop purpose its tour-purpose flags, half-tour, stops of that purpose so far, tours remaining, free hours and time-of-day bands; one-trip, adult-male and person-type terms where DaySim has them. Saturday and Sunday per purpose added. | `.stopgen` | simplified | No tour-mode, household-children, one-person-household or parcel terms. At most 4 stops per half-tour. |
| `TripTimeModel` | The half-hour of each stop's outer boundary | Conditional logit over the free periods between the anchor and the window edge: direction-specific period constants, a period shift and duration-from-anchor constants and shifts by person type, direction × work tour and stop purpose, the log free share of the period, and the trip-remaining and tours-remaining window terms | `models.TripTimeModel` | simplified | Zero travel time, so a stop lasts from its own boundary to the next one. No mode or impedance terms. |
| Home | Time between tours | Gaps are home | `common.schedule.from_episodes` | full | |
| Infeasible tours and stops | | As DaySim, the person-day is simulated again (up to `--max-resample` times, counted as `person_day_resimulated`); after that the tour or stop is dropped (counted). | `daysim.chain` | full | |
| Location, mode, travel time, work-based subtours, household models | | — | — | absent | Not in the data. |

**Stops next to an activity of the same type:** DaySim has no rule against
them, and neither does this variant. They are written as separate episodes
(for example `shop, shop`), as in the observed data, which has 4,682 such
pairs. Nothing is merged.

**Estimation:** each observed tour is estimated against the time its day's
higher-priority tours left free. An observed choice that is not available is
kept available and counted (`forced_available` in `validation.json`).

## Deviations from DaySim

* **No locations, modes or travel times.** No minimum home stay between
  tours.
* **Coefficients are re-estimated on NTS**, with DaySim's terms that have an
  NTS counterpart (approximated where NTS is coarser), plus Saturday and
  Sunday. No calibration. See [how the variants differ from the
  sources](source_differences.md).
* **One person-day**: no household or joint tours.
* **No tours from work.** Work interrupted by other activities is a work tour
  with work stops; DaySim would read most of these as work-based subtours
  ([tour_rules.md](tour_rules.md)).
* **Day starts at midnight**, not 03:00.

## Ablations

| Switch | Effect |
|---|---|
| `--independent-timing` | Primary-activity start from a kernel density estimate and duration from a log-normal regression per timing group, redrawn until it fits. |
| `--consistency rescale` | No time windows: overlaps are fixed afterwards by shifting and proportional shrinking (counted). |
| `--time-resolution 15 / 30 / 60` | Period length. |

## Outputs

The [shared outputs](source_aligned.md#outputs): `models/daysim_models.pkl`,
`validation.json` (with every tour-time coefficient under
`info.tod_coefficients`, the stop-generation and exact-tours coefficients under
`info.spec_coefficients` and parameter counts under `info.n_params`) and `generation_report.json`.
