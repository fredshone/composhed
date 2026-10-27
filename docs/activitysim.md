# ActivitySim-aligned variant

**Code:** `composhed/activitysim/`, which builds on `composhed/common/` (see [what the source-aligned variants share](source_aligned.md))

ActivitySim (from the CT-RAMP family) runs a long chain of small models, with
separate coefficients for each **person type**, such as full-time worker,
retiree or university student. This variant runs the person-level part of that
chain:

1. **Person type.** Assigned from age and employment.
2. **Day pattern (CDAP).** Mandatory (work / school), non-mandatory only, or
   at home.
3. **Mandatory tours.** How many (one work, two work, one school, work and
   school, ...), then when each one leaves and returns, on an hourly grid.
4. **Non-mandatory tours.** How many of each purpose, then when each one leaves
   and returns, in a fixed order, from the hours still free.
5. **Stops.** How many on each half of each tour, then each stop's purpose and
   departure hour from lookup tables.

A timetable records which hours are taken, so later tours only see the time
that is still free.

## Running it

```bash
uv run composhed-train-activitysim --attributes A.csv --schedules S.csv --output-dir models/ --seed 0
uv run composhed-generate-activitysim --attributes A.csv --models models/activitysim_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

Options beyond the [shared ones](source_aligned.md#running-them):

| Option | Default | Effect |
|---|---|---|
| `--time-resolution {15,30,60}` | 60 | Length of a time period after the 00:00–05:00 early period |
| `--segmentation / --no-segmentation` | on | Separate coefficients per person type, or one pooled model |
| `--n-min` | 200 | Person types with fewer observations than this use the pooled model |
| `--min-support` | 30 | Non-mandatory tour combinations seen fewer times than this are merged into the nearest common one |

## Inputs

All eight person attributes, one-hot encoded with the first category dropped,
plus the derived person type.

## Components

"Fidelity" says how closely each component follows ActivitySim:

* **full:** same structure and model form;
* **simplified:** same idea, reduced form;
* **absent:** not modelled.

| ActivitySim model | What it decides | Model | Code | Fidelity | Notes |
|---|---|---|---|---|---|
| Person type | Full-time worker, part-time worker, university student, non-worker, retiree, school child, ... | Rules on age × employment (table in the module docstring) | `persontype.person_type` | simplified | NTS's age bands can't separate pre-school, non-driving and driving-age children. |
| CDAP | Mandatory, non-mandatory or home day | Multinomial logit per person type | `models.AsimModels.cdap` | simplified | No household interaction terms. NTS has no home-all-day diaries, so the home option is only available where it is observed. |
| Mandatory tour frequency | work1, work2, school1, school2 or work_and_school | Multinomial logit per person type over the options seen in the data | `.mtf` | full | |
| Mandatory tour scheduling | Departure and return hour of each work / school tour | Conditional logit over all (start, end) pairs: an early 00:00–05:00 period plus hourly periods to midnight, giving 20 periods and 210 pairs. Start, end and duration constants, plus timing terms interacted with attributes, person type and a second-tour flag. | `common.tod.TourTOD` | simplified | ActivitySim uses piecewise departure, arrival and duration shift terms; here period constants plus interactions. |
| Non-mandatory tour frequency | How many escort, shopping, maintenance, social and discretionary tours (≤ 4 in total) | Multinomial logit per person type over the combinations seen at least 30 times. People with a non-mandatory day can't choose zero tours. | `.nmtf` | simplified | No extension model; the cap covers 99.84% of days. NTS purposes are mapped: medical → maintenance, visit → social, other → discretionary. |
| Non-mandatory tour scheduling | Departure and return hour of each tour | As for mandatory tours, with purpose terms; tours scheduled in the order escort → shopping → maintenance → social → discretionary | `TourTOD` | full | |
| Timetable | Which hours are still free | Scheduled tours plus a 10-minute buffer are blocked | `common.timewindow.TimeWindow` | full | |
| Stop frequency | Outbound and return stops (0–3 each) | Multinomial logit per tour purpose over the combinations seen | `.stopfreq` | full | |
| Trip purpose | Purpose of each stop | Probability table by tour purpose, direction and stop number; falls back to coarser keys below 20 observations | `.trip_purpose` | full | |
| Trip scheduling | Departure hour of each trip | Probability table by tour type, direction and stop number, limited to the tour's hours. The exact minute comes from the observed within-hour distribution; stop durations are the gaps between departures. | `.trip_departure`, `trip_intra` | simplified | ActivitySim's table is relative to the tour's hours; here it uses clock hours restricted to the tour. |
| Location, mode, travel time, joint tours, at-work sub-tours | | — | — | absent | Not in the data, or out of scope for a person-day model. |

**Segmentation:** each person type with at least 200 observations
(`--n-min`) gets its own coefficients. Smaller types use the pooled model; they
are logged at training time and listed in `validation.json` under
`pooled_segments`.

## Calibration

[Shared method](source_aligned.md#calibration). As in ActivitySim practice, the
constants are adjusted per segment.

**What is matched:**
* day-pattern, mandatory-frequency and non-mandatory-frequency shares, per
  person type;
* stop-frequency shares, per tour purpose;
* start-hour, end-hour and duration-band shares, per scheduling group. These
  steps are scaled by 1/3, as for DaySim.

**Tolerance:** it widens for small segments. For example, university students
have about 120 simulated people per pass.

## Deviations from ActivitySim

* **No locations, modes or travel times**, and a 10-minute buffer at home
  between tours.
* **Coefficients are re-estimated on NTS**, not taken from the ActivitySim
  example models.
* **One person-day**: no household CDAP interactions or joint tours.
* **Capped choice sets:** only options seen in the data; rare non-mandatory
  combinations are merged into the nearest one; at most 3 × 3 stops; no
  frequency extension model.
* **Approximate person types**, from binned age and employment. Combinations
  that don't map become non-workers, counted under `info.unmapped` in
  `validation.json`.
* **No at-work sub-tours.**
* **Unplaceable tours are dropped (counted).**

## Ablations

| Switch | Effect |
|---|---|
| `--no-segmentation` | One pooled model everywhere, with no person-type terms in scheduling. |
| `--time-resolution 15 / 30 / 60` | Period length after the early 00:00–05:00 period. |
| `--independent-timing` | Compositional-style timing: start from a kernel density estimate, independent log-normal duration. |
| `--consistency rescale` | No timetable: overlaps are fixed afterwards by shifting and shrinking. |
| `--no-calibrate` | Estimation only. |

## Outputs

The [shared outputs](source_aligned.md#outputs): `models/activitysim_models.pkl`,
`validation.json` and `generation_report.json`.
