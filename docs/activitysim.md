# ActivitySim-aligned variant

**Code:** `composhed/activitysim/`, which builds on `composhed/common/` (see [what the source-aligned variants share](source_aligned.md))

ActivitySim (from the CT-RAMP family) runs a long chain of small models,
segmented by **person type** (full-time worker, retiree, university student,
...). This variant runs the person-level part of that chain, in ActivitySim's
order (`prototype_mtc` `settings.yaml`, v1.6.0):

1. **Person type.** Assigned from age and employment.
2. **Day pattern (CDAP).** Mandatory (work / school), non-mandatory only, or
   at home.
3. **Mandatory tour frequency**, then **mandatory tour scheduling**: when each
   work / school tour leaves and returns, on an hourly grid.
4. **Non-mandatory tour frequency**: how many tours of each purpose (escort up
   to 2, the others up to 1), then the **extension** step that can add up to 2
   more of a chosen purpose. This sees the mandatory tours already placed,
   through the longest stretch of the day left free.
5. **Non-mandatory tour scheduling**, in a fixed purpose order, from the hours
   still free.
6. **Stops.** How many on each half of each tour, then each stop's purpose and
   departure period from lookup tables anchored on the tour's hours.

A timetable records which times are taken, so later tours only see the time
that is still free. ActivitySim stops at time periods; the exact minute is
drawn uniformly from the free minutes of the chosen period.

The choice models read ActivitySim's own `prototype_mtc` v1.6.0 spec files,
vendored in `composhed/activitysim/specs/` (BSD-3-Clause, licence included):
CDAP, mandatory and non-mandatory tour frequency (one coefficient file per
person type) and stop frequency. Spec rows whose variables NTS lacks are
dropped; every remaining coefficient is re-estimated on NTS; nothing is
calibrated. Scheduling follows the `tour_scheduling_*.csv` terms
(`tod_spec.py`). Day of week is the only variable ActivitySim does not have.
[How the variants differ from the sources](source_differences.md) lists every
remaining difference.

## Running it

```bash
uv run composhed-train-activitysim --attributes A.csv --schedules S.csv --output-dir models/ --seed 0
uv run composhed-generate-activitysim --attributes A.csv --models models/activitysim_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

Training takes a few minutes on NTS 2023 when run alone; `scripts/time_models.py` measures it.

Options beyond the [shared ones](source_aligned.md#running-them):

| Option | Default | Effect |
|---|---|---|
| `--time-resolution {15,30,60}` | 60 | Period length. 60: an early 00:00–05:00 period, then hourly (as `prototype_mtc`). 15 / 30: an early 00:00–03:00 period, then 15- or 30-minute periods (the half-hourly SEMCOG, ARC and MWCOG examples start at 03:00). |
| `--segmentation / --no-segmentation` | on | One non-mandatory frequency coefficient file per person type, as `prototype_mtc`, or the full-time worker file for everyone (see Ablations) |

## Inputs

The spec files' own variables, built from the NTS labels where NTS has a
counterpart (`sourcevars.py`):

| ActivitySim variable | From NTS |
|---|---|
| `ptype` | person type (below) |
| `age` | midpoint of the age band |
| `sex`, `female` | `sex` |
| `income_in_thousands` and the income dummies | `hh_income` band rank → 10, 30, 45, 75, 150 (MTC's 20 / 50 / 100 k$ thresholds placed on the NTS bands) |
| `auto_ownership` against workers / drivers | `veh_per_adult`: 0, 0.5 or 1.5 cars per adult against one worker / driver |
| `home_is_urban` | `hh_zone == urban` |
| `auPkTotal`, `auOpRetail` (auto accessibility) | `wt_travel_time_to_cities`: 3 − band rank |
| joint-tour counts, at-work subtours | 0 (not modelled) |
| `saturday`, `sunday` | `day` (added; ActivitySim models a weekday) |

Rows using anything else (household composition, distances and travel times
to work or school, tour mode, walk / transit accessibility, household tour
counts) are dropped and listed in `validation.json` (`dropped_rows`).
Coefficients with no variation on NTS (for example the age sub-bands within
the single 0–11 band) are fixed at 0 and listed (`unidentified`).
Scheduling uses its own term list (person type, income, rural home, Saturday,
Sunday). The labels themselves are not covariates.

## Components

"Fidelity" says how closely each component follows ActivitySim v1.6.0
(`prototype_mtc` unless noted):

* **full:** same structure, conditioning and model form;
* **simplified:** same idea, reduced form;
* **absent:** not modelled.

| ActivitySim model | Here | Code | Fidelity | Differences from ActivitySim |
|---|---|---|---|---|
| Person type | As ActivitySim's `annotate_persons`, children by age and adults by employment (table in the module docstring); 7 types: FTW, UNIV, NONWORK, RETIRED, DRIVING, SCHOOL, PRESCHOOL | `persontype.person_type` | simplified | ActivitySim has 8 types. NTS `employment` no longer separates full- and part-time work, so PTW is merged into FTW (40,651 person-days). PRESCHOOL is the 0–11 band with `employment = void` (not in school, 5,176), SCHOOL the rest of 0–11 (6,031, which includes some 5-year-olds); the 11–18 band is DRIVING (7,242), although it mixes non-driving 11–15-year-olds with 16–17-year-olds. RETIRED is self-reported retirement at any age, not ActivitySim's "non-worker aged 65+" (the 62–71 band straddles 65). |
| CDAP | One logit on `cdap_indiv_and_hhsize1.csv` with `cdap_coefficients.csv`: person-type constants and person-type interactions with age, sex, cars against workers, income and accessibility; Saturday and Sunday terms added. 40 estimated coefficients (source: 54 individual-level). | `models.AsimModels.cdap` | simplified | Person-level: no household interaction terms, no fixed proportions for large households. As ActivitySim, non-working adults and retirees cannot choose M: their 321 NTS days with work or education are left out of estimation and such persons never get M days. |
| Mandatory tour frequency | One logit on `mandatory_tour_frequency.csv` with its coefficients and availability rows (e.g. no work tours for driving-age children, only school1 for pre-schoolers). 31 coefficients (source: 51). | `.mtf` | simplified | No distance, auto-time, household or student-employment terms. 280 NTS days whose choice the spec makes unavailable are left out of estimation. On work_and_school days non-workers schedule school first, as ActivitySim does. |
| Mandatory tour scheduling | Conditional logit over (start period, end period) pairs: an early 00:00–05:00 period plus hourly periods, 20 periods and 210 pairs. Utility term by term from `tour_scheduling_work.csv` / `tour_scheduling_school.csv` (`activitysim.tod_spec`): MTC's 10 departure, 10 arrival and 9 duration constant groups; departure and duration shift effects by person type, income (band rank), first of 2+ and subsequent tours; early-departure / late-arrival interactions for high income and rural homes; full-time worker under 9 hours and 10:00–12:00 departure; short-duration terms for 2+ tours and work-and-school days; and the timetable terms (departs in the hour a previous tour ended, arrives in the hour a later tour starts, gap to the adjacent tours, tours still to schedule ÷ free hours). | `common.tod.TourTOD` | simplified | No mode-choice logsum, auto times, CBD or household terms (no data). Saturday and Sunday shift effects added: NTS covers every day, ActivitySim a weekday. The adjacent-window terms are gaps in hours, not the four first / second+ tour interactions. ActivitySim's 190 pairs map times before 05:00 into the 05:00 period instead of a separate period. |
| Non-mandatory tour frequency | `non_mandatory_tour_frequency.csv` (`interaction_simulate`) over the 48 alternatives of `non_mandatory_tour_frequency_alternatives.csv` without eat-out, with one coefficient file per person type as `prototype_mtc` (the full-time worker file for workers). Terms include tour-count constants, mandatory tours and the log of the longest free window left by them, income, sex, cars, accessibility and urban home, each on the purposes and counts the file gives it; Saturday and Sunday × each purpose added. 24–69 coefficients per person type (source: about 100). The zero-tour alternative is unavailable without mandatory tours (the spec's own −999 row). Then the extension step: a probability table by purpose, has-mandatory-tour and person type gives 0, 1 or 2 extra tours (`non_mandatory_tour_frequency_extension_probs.csv`, re-tabulated on NTS). | `.nmtf`, `.nm_extension` | simplified | No joint-tour or accessibility terms. Purposes: escort, shopping, othmaint ← other, othdiscr ← leisure, social ← visit. NTS has no eat-out purpose (eating out falls under leisure). |
| Non-mandatory tour scheduling | Term by term from `tour_scheduling_nonmandatory.csv`: separate escort and non-escort sets of departure, arrival and duration constants; departure and duration shifts by purpose (shopping, othmaint, social) and person type; number of mandatory, escort and other non-mandatory tours × departure; first / subsequent tour of the same purpose; MTC's early-departure, late-arrival and short-duration interactions; the timetable terms. Tours are scheduled after the mandatory ones in ActivitySim's order escort → shopping → othmaint → othdiscr → social. | `TourTOD` | simplified | No joint-tour, logsum, CBD or household-composition terms. Saturday and Sunday shift effects added. |
| Timetable | Scheduled tours are blocked at minute level, with at least 1 minute at home between tours | `common.timewindow.TimeWindow` | simplified | ActivitySim blocks whole periods, and a tour may start in the period another ends. Here too, from the minute after it ends. |
| Stop frequency | `stop_frequency_<purpose>.csv` with its coefficients, over the 16 outbound × inbound alternatives: the file's out / in / total constants and its income, sex, cars, tour-count and tour-time terms; Saturday and Sunday added. Segments work, school, univ (school tours of university students), escort, shopping, othmaint, othdiscr, social. 13–19 coefficients per purpose (source: about 27). | `.stopfreq` | simplified | No eatout or atwork segments; no household, mode, distance or accessibility terms. The work and othmaint income rows compare `income_in_thousands` with 19999 in the source file, so they are always 0 there and here (left unidentified). |
| Trip purpose | Probability table by tour purpose, direction, tour-hour band (start hour outbound, end hour inbound: before 09:00, 09:00–16:00, after) and person type; falls back to coarser keys below 20 observations | `.trip_purpose` | simplified | Same form and conditioning as `trip_purpose_probs.csv`, with coarser hour bands, tabulated on NTS. Stops are always discretionary (work and education inside a tour are part of the work block, see [tour rules](tour_rules.md)). |
| Trip scheduling | Probability table over departure periods by direction, mandatory / non-mandatory, tour hour (start period outbound, end period inbound), tour purpose and trip number from that anchor. Outbound trips are scheduled forwards from the tour start and inbound trips backwards from the tour end. | `.trip_departure` | simplified | ActivitySim's default "departure" mode keys on the same things. Infeasible periods are removed and the rest renormalised, which equals ActivitySim's fail-and-retry with unlimited retries. If no feasible period has probability, the one nearest the anchor is used (as `choose_most_initial`) and counted as `trip_departure_failfix` in `generation_report.json`. ActivitySim stops at the period; the minute is drawn uniformly, see below. |
| Joint tours, school escorting, at-work sub-tours | — | — | absent | One person-day: no household (`hid` is dropped at load). Escort-to-school is an ordinary escort tour, as in `prototype_mtc` (`prototype_mtc_extended` has a school-escorting model). Activities inside a work tour become inbound stops after a merged work block ([tour rules](tour_rules.md)), which moves midday at-work activities to after work. |
| Accessibility, auto ownership, free parking, work from home / telecommute, location, mode, travel time | — | — | absent | Not in the data. NTS `veh_per_adult` is used as an ordinary input label instead of auto ownership being modelled. |

**Segmentation:** as `prototype_mtc`, only non-mandatory tour frequency has
one coefficient set per person type. CDAP and mandatory tour frequency are one
model each, with person-type terms. Scheduling uses person-type terms.

### From periods to minutes

ActivitySim stops at time periods. To produce a schedule with minutes, this
variant:

* draws a tour's start and end minute, and each trip's departure minute,
  uniformly from the free minutes of the chosen period (the usual way
  ActivitySim periods are turned into times, and what DaySim does);
* uses zero travel time. A stop lasts until the next departure, the primary
  activity fills the rest of the tour, and home fills the gaps between tours.

An earlier version drew these minutes from NTS's own within-period minute
distribution and same-period trip gaps. That reproduced NTS's rounding of
reported times (46% of starts on the hour or half-hour) and made about 13% of
generated days exact copies of observed ones. Neither ActivitySim nor DaySim has
such a step, so it was removed. A stop that leaves in the same hour as its tour
starts now lasts about half that hour on average.

### Neighbouring stops of one type

Two neighbouring activities of the same type (e.g. two shops, or a shop stop
on a shopping tour) are written as separate episodes, as in the observed data
(4,682 such pairs in NTS 2023). Nothing is merged. An earlier version prevented
them by excluding the neighbouring purpose from each stop's purpose draw, and
a later one merged them in the writer; ActivitySim does neither, so both were
removed.

## Checks on NTS 2023

*These checks use the older NTS extract (56,734 days, no stay-at-home days) with calibration on (since removed), and an earlier version of this variant (observed within-period minutes, saturated scheduling constants). They are kept for the record; results on the current data and version are in the README.*

Seed 0, generated for all 56,734 people and compared with the observed days.
"Before" is the version reviewed in the fidelity report.

| | Observed | Before | Now |
|---|---|---|---|
| Stops per tour | 0.207 | 0.150 | 0.205 |
| Same-type merges (stops lost) | — | 4,797 | 0 |
| Work tour, outbound stop, median minutes | 20 | 70 | 27 |
| Work tour, inbound stop, median minutes | 45 | 70 | 60 |
| Work tour 06:00–09:00 start, one outbound stop, median | 15 | 60 | 20 |
| Work tour 16:00–19:00 end, one inbound stop, median | 45 | 90 | 60 |
| Non-mandatory tour stop, median minutes | 38 | 15 | 25 |

Stops on non-mandatory tours remain too short. The trip departure tables, like
ActivitySim's, are not conditioned on how long the tour is.

The non-mandatory frequency model then had 630 parameters per person type,
against 1,755 for the earlier alternative-specific logit (now 24–69, from the spec file). On a 20% holdout it fits
better for 5 of 7 person types (e.g. full-time workers 0.241 against 0.234
pseudo-R², part-time workers 0.169 against 0.151). University students are the
exception: 0.186 against 0.288, on about 70 held-out people.

The fitted timetable terms have ActivitySim's signs (NTS
2023, seed 0, before the spec-file rewrite). A non-mandatory tour is less likely to leave in the hour another
tour ended (−1.17; `prototype_mtc`'s work spec has −0.89) or to arrive in the
hour the next one starts (−1.22; `prototype_mtc`: −1.33).

## Deviations from ActivitySim, in short

**Forced by the data:**
* no locations, modes, travel times or logsums;
* one person-day: no household CDAP, joint tours or school escorting;
* approximate person types from binned age and employment;
* no eat-out purpose.

**Choices made here** (see [how the variants differ from the
sources](source_differences.md)):
* coefficients re-estimated on NTS, not taken from the ActivitySim examples;
  no calibration;
* NTS counterparts for the spec variables, approximated where NTS is coarser
  (age bands, income bands, cars per adult, one accessibility index);
* Saturday and Sunday terms added to CDAP, non-mandatory frequency, stop
  frequency and scheduling (NTS covers every day);
* trip purpose, trip departure and extension tables tabulated from NTS;
* uniform minutes inside the chosen periods;
* no at-work sub-tours;
* stops that do not fit a tour by whole minutes, and unplaceable tours, are
  dropped and counted.

## Ablations

| Switch | Effect |
|---|---|
| `--no-segmentation` | The full-time worker non-mandatory frequency coefficients for every person type, and no person-type terms in scheduling. CDAP and mandatory frequency are single models either way. |
| `--time-resolution 15 / 30 / 60` | Period length (see Running it). |
| `--independent-timing` | Compositional-style timing: start from a kernel density estimate, independent log-normal duration. |
| `--consistency rescale` | No timetable: overlaps are fixed afterwards by shifting and shrinking. |

## Outputs

The [shared outputs](source_aligned.md#outputs): `models/activitysim_models.pkl`,
`validation.json` and `generation_report.json`. `validation.json` also records
every spec coefficient (`info.spec_coefficients`), the parameter counts
(`info.n_params`), the dropped and unidentified spec rows of each model, the
fitted timetable terms (`timetable_terms` under each scheduling model), the
trip purpose table and the number of tours left out of the trip departure
table because their departures are out of order (43 on NTS 2023; see
[tour rules](tour_rules.md)).
