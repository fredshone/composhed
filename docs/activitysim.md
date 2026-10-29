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

The variant was checked component by component against the ActivitySim v1.6.0
source and example configurations (October 2026). The review is in
`reports/ActivitySim variant fidelity.md`. The scheduling, non-mandatory
frequency and stop-frequency utilities were then rewritten term by term from
the `prototype_mtc` spec files, so that the variant is fitted to NTS only
through ActivitySim's own model forms.

## Running it

```bash
uv run composhed-train-activitysim --attributes A.csv --schedules S.csv --output-dir models/ --seed 0
uv run composhed-generate-activitysim --attributes A.csv --models models/activitysim_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

Training takes about 7 minutes on NTS 2023, of which about half is calibration.

Options beyond the [shared ones](source_aligned.md#running-them):

| Option | Default | Effect |
|---|---|---|
| `--time-resolution {15,30,60}` | 60 | Period length. 60: an early 00:00–05:00 period, then hourly (as `prototype_mtc`). 15 / 30: an early 00:00–03:00 period, then 15- or 30-minute periods (the half-hourly SEMCOG, ARC and MWCOG examples start at 03:00). |
| `--segmentation / --no-segmentation` | on | Separate models per person type, or one pooled model (see Ablations) |
| `--n-min` | 200 | Person types with fewer observations than this use the pooled model |

## Inputs

All eight person attributes, one-hot encoded with the first category dropped,
plus the derived person type.

## Components

"Fidelity" says how closely each component follows ActivitySim v1.6.0
(`prototype_mtc` unless noted):

* **full:** same structure, conditioning and model form;
* **simplified:** same idea, reduced form;
* **absent:** not modelled.

| ActivitySim model | Here | Code | Fidelity | Differences from ActivitySim |
|---|---|---|---|---|
| Person type | As ActivitySim's `annotate_persons`, children by age and adults by employment (table in the module docstring); 7 types: FTW, UNIV, NONWORK, RETIRED, DRIVING, SCHOOL, PRESCHOOL | `persontype.person_type` | simplified | ActivitySim has 8 types. NTS `employment` no longer separates full- and part-time work, so PTW is merged into FTW (40,651 person-days). PRESCHOOL is the 0–11 band with `employment = void` (not in school, 5,176), SCHOOL the rest of 0–11 (6,031, which includes some 5-year-olds); the 11–18 band is DRIVING (7,242), although it mixes non-driving 11–15-year-olds with 16–17-year-olds. RETIRED is self-reported retirement at any age, not ActivitySim's "non-worker aged 65+" (the 62–71 band straddles 65). |
| CDAP | Multinomial logit per person type on the 8 labels | `models.AsimModels.cdap` | simplified | Person-level: no household member interactions, no fixed proportions for large households. H (stay-at-home) is estimated and generated: 41% of NTS days are home all day, from about 33% (FTW) to 50% (NONWORK). ActivitySim's rule that non-workers and retirees cannot have an M day is not imposed, because NTS-based typing contradicts it (9.5% of NONWORK days are M). |
| Mandatory tour frequency | Multinomial logit per person type over work1, work2, school1, school2, work_and_school | `.mtf` | simplified | Same alternatives. No person-type availability rules (NTS contradicts them too), no travel-time, car or household terms. RETIRED and UNIV use the pooled model. On work_and_school days non-workers schedule school first, as ActivitySim does. |
| Mandatory tour scheduling | Conditional logit over (start period, end period) pairs: an early 00:00–05:00 period plus hourly periods, 20 periods and 210 pairs. Utility term by term from `tour_scheduling_work.csv` / `tour_scheduling_school.csv` (`activitysim.tod_spec`): MTC's 10 departure, 10 arrival and 9 duration constant groups; departure and duration shift effects by person type, income (band rank), first of 2+ and subsequent tours; early-departure / late-arrival interactions for high income and rural homes; full-time worker under 9 hours and 10:00–12:00 departure; short-duration terms for 2+ tours and work-and-school days; and the timetable terms (departs in the hour a previous tour ended, arrives in the hour a later tour starts, gap to the adjacent tours, tours still to schedule ÷ free hours). | `common.tod.TourTOD` | simplified | No mode-choice logsum, auto times, CBD or household terms (no data). Saturday and Sunday shift effects added: NTS covers every day, ActivitySim a weekday. The adjacent-window terms are gaps in hours, not the four first / second+ tour interactions. ActivitySim's 190 pairs map times before 05:00 into the 05:00 period instead of a separate period. |
| Non-mandatory tour frequency | Logit over all 48 combinations of escort 0–2 and shopping, othmaint, othdiscr, social 0–1 (`non_mandatory_tour_frequency_alternatives.csv` without eatout), per person type, then the extension step: a probability table by purpose, has-mandatory-tour and person type gives 0, 1 or 2 extra tours of each chosen purpose (`non_mandatory_tour_frequency_extension_probs.csv`). As in ActivitySim's `interaction_simulate` spec, utilities are built from alternative attributes (tours per purpose, two escort tours, total-tour dummies) interacted with person features. Covariates include the log of the longest free window left by the mandatory tours ("Logged Maximum Residual Window"). People with a non-mandatory day can't choose zero tours. | `.nmtf`, `.nm_extension` | simplified | No joint-tour or accessibility terms. Purposes: escort, shopping, othmaint ← other, othdiscr ← leisure, social ← visit. NTS has no eat-out purpose (eating out falls under leisure). |
| Non-mandatory tour scheduling | Term by term from `tour_scheduling_nonmandatory.csv`: separate escort and non-escort sets of departure, arrival and duration constants; departure and duration shifts by purpose (shopping, othmaint, social) and person type; number of mandatory, escort and other non-mandatory tours × departure; first / subsequent tour of the same purpose; MTC's early-departure, late-arrival and short-duration interactions; the timetable terms. Tours are scheduled after the mandatory ones in ActivitySim's order escort → shopping → othmaint → othdiscr → social. | `TourTOD` | simplified | No joint-tour, logsum, CBD or household-composition terms. Saturday and Sunday shift effects added. |
| Timetable | Scheduled tours plus a 10-minute home buffer are blocked, at minute level | `common.timewindow.TimeWindow` | simplified | ActivitySim blocks whole periods, and a tour may start in the period another ends. Here a tour may also start in that period, but at least 10 minutes after the other ends. |
| Stop frequency | Logit over all 16 outbound × inbound alternatives (0–3 each), per tour purpose. As `stop_frequency_*.csv`: outbound-count, inbound-count and total-stop constants; covariates (labels, departure 06:00–07:59, arrival from 19:00, tour of 3+, 9+ and 11+ hours, the person's work, school and escort tours) act on "any stop", "any outbound stop" and "any inbound stop". | `.stopfreq` | simplified | Segments: work, school, escort, shopping, othmaint, othdiscr, social. ActivitySim also has univ, eatout and atwork segments, and household, mode, distance and accessibility terms. |
| Trip purpose | Probability table by tour purpose, direction, tour-hour band (start hour outbound, end hour inbound: before 09:00, 09:00–16:00, after) and person type; falls back to coarser keys below 20 observations | `.trip_purpose` | simplified | Same form and conditioning as `trip_purpose_probs.csv`, with coarser hour bands. Stops are always discretionary (work and education inside a tour are part of the work block, see [tour rules](tour_rules.md)). The adjacent activity's purpose is excluded (see below). |
| Trip scheduling | Probability table over departure periods by direction, mandatory / non-mandatory, tour hour (start period outbound, end period inbound), tour purpose and trip number from that anchor. Outbound trips are scheduled forwards from the tour start and inbound trips backwards from the tour end. | `.trip_departure` | simplified | ActivitySim's default "departure" mode keys on the same things. Infeasible periods are removed and the rest renormalised, which equals ActivitySim's fail-and-retry with unlimited retries. If no feasible period has probability, the one nearest the anchor is used (as `choose_most_initial`) and counted as `trip_departure_failfix` in `generation_report.json`. ActivitySim stops at the period; the minute is drawn uniformly, see below. |
| Joint tours, school escorting, at-work sub-tours | — | — | absent | One person-day: no household (`hid` is dropped at load). Escort-to-school is an ordinary escort tour, as in `prototype_mtc` (`prototype_mtc_extended` has a school-escorting model). Activities inside a work tour become inbound stops after a merged work block ([tour rules](tour_rules.md)), which moves midday at-work activities to after work. |
| Accessibility, auto ownership, free parking, work from home / telecommute, location, mode, travel time | — | — | absent | Not in the data. NTS `veh_per_adult` is used as an ordinary input label instead of auto ownership being modelled. |

**Segmentation:** each person type with at least 200 observations
(`--n-min`) gets its own CDAP, mandatory-frequency and non-mandatory-frequency
model. This goes further than ActivitySim, which has per-person-type
coefficient files only for non-mandatory tour frequency, and shares one spec
with person-type terms elsewhere. Smaller types use the pooled model; they are
logged at training time and listed in `validation.json` under
`pooled_segments`. Scheduling uses person-type terms, as ActivitySim does.

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

### Stop purposes and merging

The schedule output has no locations, so two neighbouring activities of the
same type (e.g. two shops) would be merged into one, silently removing the
stop. A stop's purpose therefore excludes the purpose of the activity next to
it. NTS has 4,768 such same-type neighbours at different locations, which this
cannot reproduce. Before this rule, 7% of generated people lost a stop this way
(4,797 merges); now there are none.

## Calibration

[Shared method](source_aligned.md#calibration). ActivitySim v1.6.0 has no
built-in calibration (its documentation's Calibration section is a
placeholder). Agencies adjust constants iteratively against targets with their
own scripts. The loop here does the same, but **its targets are the training
data**, so the calibrated marginals are partly fitted by construction.

**What is matched:**
* day-pattern, mandatory-frequency and non-mandatory-frequency shares, per
  person type;
* stop-frequency shares, per tour purpose;
* start-hour, end-hour and duration-band shares, per scheduling group. These
  steps are scaled by 1/3, as for DaySim.

The targets interact. Non-mandatory frequency depends on the free window left
by the calibrated mandatory schedule, so a pass can move one target out of
tolerance while fixing another. **Calibration does not always converge in the
8 passes.** It converges for some seeds and leaves one or two
non-mandatory-frequency segments slightly out of tolerance for others (for
example 0.015 against a 0.011 tolerance). Each run's report is in
`validation.json` under `info.calibration`.

**Tolerance:** it widens for small segments. For example, university students
have about 120 simulated people per pass.

## Checks on NTS 2023

*These checks use the older NTS extract (56,734 days, no stay-at-home days) with calibration on, and the earlier version of this variant (observed within-period minutes, saturated scheduling constants). They are kept for the record; results on the current data and version are in the README.*

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

The non-mandatory frequency model has 630 parameters per person type, against
1,755 for the earlier alternative-specific logit. On a 20% holdout it fits
better for 5 of 7 person types (e.g. full-time workers 0.241 against 0.234
pseudo-R², part-time workers 0.169 against 0.151). University students are the
exception: 0.186 against 0.288, on about 70 held-out people.

The fitted timetable terms have ActivitySim's signs. A non-mandatory tour is
less likely to leave in the hour another tour ended (−1.23; `prototype_mtc`'s
work spec has −0.89) or to arrive in the hour the next one starts (−1.25;
`prototype_mtc`: −1.33).

## Deviations from ActivitySim, in short

**Forced by the data:**
* no locations, modes, travel times or logsums, and a 10-minute home buffer
  between tours;
* one person-day: no household CDAP, joint tours or school escorting;
* approximate person types from binned age and employment;
* no eat-out purpose.

**Choices made here:**
* coefficients re-estimated on NTS, not taken from the ActivitySim examples;
* Saturday and Sunday shift effects in scheduling (NTS covers every day);
* separate models per person type for CDAP and both frequency models;
* uniform minutes inside the chosen periods, and stop purposes that avoid the
  neighbouring activity's purpose;
* no at-work sub-tours;
* unplaceable tours are dropped and counted (none on NTS 2023);
* calibration on the training data.

## Ablations

| Switch | Effect |
|---|---|
| `--no-segmentation` | One pooled model for CDAP and both frequency models, and no person-type terms in scheduling. Stop frequency stays per tour purpose, as in ActivitySim. |
| `--time-resolution 15 / 30 / 60` | Period length (see Running it). |
| `--independent-timing` | Compositional-style timing: start from a kernel density estimate, independent log-normal duration. |
| `--consistency rescale` | No timetable: overlaps are fixed afterwards by shifting and shrinking. |
| `--no-calibrate` | Estimation only. |

## Outputs

The [shared outputs](source_aligned.md#outputs): `models/activitysim_models.pkl`,
`validation.json` and `generation_report.json`. `validation.json` also records
the fitted timetable terms (`timetable_terms` under each scheduling model), the
trip purpose table and the number of tours left out of the trip departure
table because their departures are out of order (47; see
[tour rules](tour_rules.md)).
