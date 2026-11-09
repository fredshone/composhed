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
   work / school tour leaves and returns, in `prototype_mtc`'s 19 hourly
   periods (05:00 to 23:00).
4. **Non-mandatory tour frequency**: how many tours of each purpose (escort up
   to 2, the others up to 1), then the **extension** step that can add up to 2
   more tours of a purpose already at its maximum, for people with 1 to 4
   tours. This sees the mandatory tours already placed, through the longest
   run of free periods.
5. **Non-mandatory tour scheduling**, in a fixed purpose order, from the hours
   still free.
6. **Stops.** How many on each half of each tour, then each stop's purpose and
   departure period from probability tables with the keys of ActivitySim's
   own tables.

A timetable records which periods are taken, so later tours only see the time
that is still free. As in ActivitySim, every time is a period (an hour, as
`prototype_mtc`), and nothing finer; each period is written at hh:30.

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
| `--time-resolution {15,30,60}` | 60 | Period length. 60: `prototype_mtc`'s 19 hourly periods 5–23, the first holding every earlier time. 15 / 30: an early 00:00–03:00 period, then 15- or 30-minute periods (the half-hourly SEMCOG, ARC and MWCOG examples start at 03:00). |
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
| Mandatory tour scheduling | Conditional logit over `prototype_mtc`'s 190 (start period, end period) pairs: hourly periods 5–23, the first holding every earlier time. Utility row by row from `tour_scheduling_work.csv` / `tour_scheduling_school.csv` (`activitysim.tod_spec`), with the constants their coefficient files fix at 0 left out: MTC's 10 departure, 10 arrival and 9 duration constant groups; the person-type shifts the files have (work: non-working adult duration, university student departure; school: full-time worker, non-worker, university student and `ptype == 7` terms); income × departure; first of 2+ and later tours; high-income and rural early / late interactions; full-time worker under 9 hours and 10:00–12:00 departure; the under-8 / under-6-hour terms; and the timetable rows as in ActivitySim's `timetable.py`: a previous tour ends in the departure period / begins in the arrival period, whether a free period is adjacent before departure / after arrival (for the first of 2+ and for later tours; the specs use `adjacent_window_before(...) > 0`, a 0/1 flag), and 1 / periods left for the first of 2+. As the specs' "subsequent tour must start after previous tour ends" row (coefficient −100), a second mandatory tour cannot start before the first one's end period. | `common.tod.TourTOD` | simplified | No mode-choice logsum, auto times, CBD or household terms (no data); no part-time worker terms (no part-time workers). Saturday and Sunday shift effects added: NTS covers every day, ActivitySim a weekday. Where ActivitySim divides by zero periods left, 1 is used. |
| Non-mandatory tour frequency | `non_mandatory_tour_frequency.csv` (`interaction_simulate`) over the 48 alternatives of `non_mandatory_tour_frequency_alternatives.csv` without eat-out, with one coefficient file per person type as `prototype_mtc` (the full-time worker file for workers). Terms include tour-count constants, mandatory tours and the log of the longest free window left by them, income, sex, cars, accessibility and urban home, each on the purposes and counts the file gives it; Saturday and Sunday × each purpose added. 24–69 coefficients per person type (source: about 100). The zero-tour alternative is unavailable without mandatory tours (the spec's own −999 row). The residual window is `log1p` of the longest run of periods not in the middle of a mandatory tour, as `max_time_block_available`. Then the extension step, as `extend_tour_counts`: for a person with 1 to 4 tours, each purpose at its maximum (escort 2, others 1) gets 0, 1 or 2 extra tours from a probability table by purpose, has-mandatory-tour and person type (`non_mandatory_tour_frequency_extension_probs.csv`'s keys, tabulated on NTS from the days the rule covers). | `.nmtf`, `.nm_extension` | simplified | No joint-tour or accessibility terms. Purposes: escort, shopping, othmaint ← other, othdiscr ← leisure, social ← visit. NTS has no eat-out purpose (eating out falls under leisure). |
| Non-mandatory tour scheduling | Row by row from `tour_scheduling_nonmandatory.csv`: separate escort and non-escort constants, with the ones its coefficient file fixes at 0 left out (escort arrivals other than evening and late among them); shopping, maintenance and visit departure and duration shifts; `ptype == 6` (driving-age) and `ptype == 7` (school child) shifts; numbers of mandatory, escort and other tours × departure; first / later tour of the same purpose; MTC's early-departure, late-arrival and short-duration interactions; the timetable rows (adjacent-window flags by tours of the same purpose; tours still to schedule, this one included, / periods left). As the spec's −999 row, a second tour of a purpose cannot start before the previous tour ends. Tours are scheduled after the mandatory ones in ActivitySim's order escort → shopping → othmaint → othdiscr → social. | `TourTOD` | simplified | No joint-tour, logsum, CBD, eat-out or household-composition terms. Saturday and Sunday shift effects added. |
| Timetable | Scheduled tours block the time between their start and end periods; as in ActivitySim, a tour may start in the period another ends, and a tour may start and end in one period | `common.tod.point_mask` | full | The home stay between two such tours lasts zero minutes. |
| Stop frequency | `stop_frequency_<purpose>.csv` with its coefficients, over the 16 outbound × inbound alternatives: the file's out / in / total constants and its income, sex, cars, tour-count and tour-time terms; Saturday and Sunday added. Segments work, school, univ (school tours of university students), escort, shopping, othmaint, othdiscr, social. 13–19 coefficients per purpose (source: about 27). | `.stopfreq` | simplified | No eatout or atwork segments; no household, mode, distance or accessibility terms. The work and othmaint income rows compare `income_in_thousands` with 19999 in the source file, so they are always 0 there and here (left unidentified). |
| Trip purpose | Probability table with `trip_purpose_probs.csv`'s keys: tour purpose (school tours of university students as `univ`), direction, tour start hour (split only for work tours: 5–8 / 9–23 outbound, 5–14 / 15–23 inbound) and person type; falls back to coarser keys below 20 observations | `.trip_purpose` | simplified | Tabulated on NTS. As in ActivitySim, stop purposes include work and education (an interrupted work or education episode is a stop, see [tour rules](tour_rules.md)). |
| Trip scheduling | Probability table over the 19 departure periods with `trip_scheduling_probs.csv`'s keys: tour purpose, direction, tour hour (start period outbound, end period inbound) and trip number in the leg; falls back to coarser keys below 20 observations. As `trip_scheduling.py` with `prototype_mtc`'s settings: outbound trips scheduled in order from the tour start, inbound trips in reverse from the tour end, each constraining the next; probability outside a trip's feasible periods is a failure, not renormalised; a leg with a failure is drawn again, up to 100 times; then `choose_most_initial` (earliest period outbound, latest inbound). `logic_version: 2`: inbound trips may not leave before the last outbound departure of the same try, or the tour start when only the inbound leg is retried. (`prototype_mtc` sets no version, so ActivitySim falls back to version 1, which leaves such a retried inbound leg without an earliest period: it fails every time and all its trips leave at the tour end. That is a known ActivitySim bug, fixed by version 2.) | `AsimChain._schedule_trips` | simplified | Tabulated on NTS. Failures are counted in `generation_report.json` (`trip_leg_retries`, `trip_departure_failfix`). An outbound departure that ends up after an inbound one (possible when only one leg is retried) is clamped and counted (`trip_departure_out_of_order`). |
| Joint tours, school escorting, at-work sub-tours | — | — | absent | One person-day: no household (`hid` is dropped at load). Escort-to-school is an ordinary escort tour, as in `prototype_mtc` (`prototype_mtc_extended` has a school-escorting model). Activities between two work episodes of one tour are stops of the work tour, with the other work episode as a work stop ([tour rules](tour_rules.md)); ActivitySim would make most of them at-work sub-tours. |
| Accessibility, auto ownership, free parking, work from home / telecommute, location, mode, travel time | — | — | absent | Not in the data. NTS `veh_per_adult` is used as an ordinary input label instead of auto ownership being modelled. |

**Segmentation:** as `prototype_mtc`, only non-mandatory tour frequency has
one coefficient set per person type. CDAP and mandatory tour frequency are one
model each, with person-type terms. Scheduling uses person-type terms.

### From periods to minutes

ActivitySim schedules in periods and stops there. This variant does the same:

* every time is a period, written at hh:30. The first period, 05:00, also
  holds every earlier time, as in `prototype_mtc`, so nothing is written
  before 05:30 and observed tours leaving earlier are estimated as leaving in
  it. There is no finer draw;
* travel takes no time. A stop lasts from one trip departure to the next, the
  primary activity fills the rest of the tour, and home fills the gaps between
  tours;
* so two events in one period are zero minutes apart. A stop whose arrival and
  departure fall in one period, a tour that starts and ends in one period, or
  the home stay between a tour and one starting in the period it ended, all
  last zero minutes and are written as zero-length episodes.

Earlier versions drew a minute inside each period, first from NTS's own
within-period minutes and then uniformly. ActivitySim has neither step, so both
were removed, and with them the dropping of stops that did not fit a tour at
whole minutes (about 300 a run).

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
* trip purpose, trip departure and extension tables tabulated from NTS, with
  the source tables' keys and a fall-back to coarser keys for sparse cells;
* each period written at hh:30;
* no at-work sub-tours (interrupted work episodes are work stops instead);
* unplaceable tours are dropped and counted.

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
