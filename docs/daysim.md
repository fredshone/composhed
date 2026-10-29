# DaySim-aligned variant

**Code:** `composhed/daysim/`, which builds on `composhed/common/` (see [what the source-aligned variants share](source_aligned.md))

DaySim (Bowman & Ben-Akiva, 2001; Bradley, Bowman & Griesenbeck, 2010) plans a
day around tours: trips that leave home and come back. This variant follows the
person-day part of DaySim's model chain. Each component's form was taken from
the DaySim source code (RSGInc/DaySim on GitHub, `DaySim/ChoiceModels/Default`;
checked October 2026), and every coefficient is re-estimated on NTS. The chain
runs in DaySim's order (`ChoiceModelRunner`):

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
`TimeWindow.GetAvailableMinute` does.

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

## Inputs

The person type (DaySim's eight types, here seven: see the
[ActivitySim page](activitysim.md#components)) and all eight person
attributes, one-hot encoded. Tour and trip time models use person type, tour
purpose, the lowest and highest income bands, and Saturday / Sunday.

## Components

"Fidelity" says how closely each component follows DaySim:

* **full:** same structure and model form;
* **simplified:** same idea, reduced form;
* **absent:** not modelled.

| DaySim model | What it decides | Model here | Code | Fidelity | Differences |
|---|---|---|---|---|---|
| `IndividualPersonDayPatternModel` | Whether the day has a tour of each of the 7 purposes and a stop of each of the 5 stop purposes | Logit over all 1,324 feasible patterns (≤ 3 tour purposes, ≤ 4 stop purposes, ≤ 5 in all, stops only with a tour, as DaySim). Constants per tour and stop purpose and for tour–stop, tour–tour, stop–stop and tour-count × stop-count combinations; person features act on each purpose's presence, on 2 / 3 tours, 2 / 3 / 4 stops, any non-mandatory tour and any stop. | `models.DaysimModels.pattern` | simplified | DaySim has 2,080 alternatives: work and school stops do not exist here (work inside a tour is merged into the work block, [tour_rules.md](tour_rules.md)). No accessibility logsums, densities or household composition. DaySim's rule that only workers can have work tours and only students school tours is not imposed: NTS contradicts it. 47 days that break DaySim's limits are mapped to the nearest feasible pattern for this model only. |
| `PersonExactNumberOfToursModel` | 1, 2 or 3+ tours of each purpose in the pattern | Multinomial logit per purpose on the person features and the pattern | `.exact` | full | No logsums. Tours beyond 3 of one purpose are dropped from the training data (counted). |
| Tour priority | Order in which tours are scheduled | Fixed: work → education → escort → other (personal business) → shop → visit → leisure (social / recreation; DaySim's meal purpose has no NTS equivalent) | `data.priority_order` | full | |
| `WorkTourTimeModel`, `SchoolTourTimeModel`, `OtherHomeBasedTourTimeModel` | Arrival and departure period at the primary destination (1,176 pairs at 30 minutes) | Conditional logit with DaySim's grouped arrival, departure and duration constants; arrival and duration shift terms by person type, tour purpose, income, the four priority flags, stop purposes in the pattern and escort stops; time-band interactions; and the time-window terms (log free minutes in the arrival and departure periods, adjacent free time for later tours, stops and tours still to place per free hour). Separate models for work, education and other tours. | `common.tod.TourTOD` with `daysim.tod_spec` | simplified | No mode impedances, transit or subtour terms. Saturday and Sunday shift terms added (DaySim models a weekday). |
| Time windows | Which pairs and periods are still possible | Time taken by earlier tours, plus a 10-minute buffer, is removed from the choice set | `common.timewindow.TimeWindow` | full | |
| `IntermediateStopGenerationModel` | The next stop's purpose, or no more stops | Multinomial logit over none + escort, other, shop, visit, leisure; a purpose is available only if the pattern has it. Covariates: tour purpose, direction, trip number in the half-tour, before a mandatory destination, tours still to schedule, free hours on this side of the tour, time of day of the tour anchor, stops so far, person type and sex. | `.stopgen` | simplified | Alternative-specific coefficients on every covariate (DaySim restricts most terms to some purposes). At most 4 stops per half-tour. A stop cannot have the purpose of the activity next to it (see below). |
| `TripTimeModel` | The half-hour of each stop's outer boundary | Conditional logit over the free periods between the anchor and the window edge: direction-specific period constants, a period shift and duration-from-anchor constants and shifts by person type, direction × work tour and stop purpose, the log free share of the period, and the trip-remaining and tours-remaining window terms | `models.TripTimeModel` | simplified | Zero travel time, so a stop lasts from its own boundary to the next one. No mode or impedance terms. |
| Home | Time between tours | Gaps are home | `common.schedule.from_episodes` | full | |
| Infeasible tours and stops | | A tour with no free pair is dropped; a stop whose boundary has no free minute is dropped. Both counted. | `daysim.chain` | simplified | DaySim marks the person-day invalid and re-simulates it. |
| Location, mode, travel time, work-based subtours, household models | | — | — | absent | Not in the data. |

**Stop purposes next to each other:** the output has no locations, so two
neighbouring activities of one type would merge into one and silently remove a
stop. A stop's purpose therefore excludes the purpose of the activity next to
it (the primary activity or the previous stop), as in the ActivitySim variant.

**Estimation:** each observed tour is estimated against the time its day's
higher-priority tours left free, buffer included. An observed choice that lands
inside a buffer is kept available and counted (`forced_available` in
`validation.json`).

## Calibration

[Shared method](source_aligned.md#calibration). Off in the reported
experiment (`--no-calibrate`).

**What is adjusted:**
* the day-pattern constants (to match pattern shares);
* the exact-number-of-tours constants per purpose;
* the stop-generation constants;
* for each tour-time model, the arrival-period, departure-period and
  duration-band constants (steps scaled by 1/3, because they act on the same
  choice).

## Deviations from DaySim

* **No locations, modes or travel times**, and a 10-minute buffer at home
  between tours.
* **Coefficients are re-estimated on NTS**, with the variables NTS has.
* **One person-day**: no household or joint tours.
* **No work or school stops, no tours from work.** Work interrupted by other
  activities is merged into one block ([tour_rules.md](tour_rules.md)).
* **Day starts at midnight**, not 03:00.
* **Failures drop a tour or stop (counted)** instead of re-simulating the day.

## Ablations

| Switch | Effect |
|---|---|
| `--no-calibrate` | Estimation only. |
| `--independent-timing` | Primary-activity start from a kernel density estimate and duration from a log-normal regression per timing group, redrawn until it fits. |
| `--consistency rescale` | No time windows: overlaps are fixed afterwards by shifting and proportional shrinking (counted). |
| `--time-resolution 15 / 30 / 60` | Period length. |

## Outputs

The [shared outputs](source_aligned.md#outputs): `models/daysim_models.pkl`,
`validation.json` (with every tour-time coefficient under
`info.tod_coefficients`) and `generation_report.json`.
