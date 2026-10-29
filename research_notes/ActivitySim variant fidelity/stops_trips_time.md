# ActivitySim variant fidelity: stop frequency, trip purpose, trip scheduling, schedule assembly and time periods

Scope note: ActivitySim sources were read from a shallow clone of tag `v1.6.0` (GitHub `ActivitySim/activitysim`, commit 8ca6741). Upstream links below point at that tag (`GH` = `https://github.com/ActivitySim/activitysim/blob/v1.6.0/`). Local citations are `path:line` in `/home/fred/Projects/composhed`. Empirical checks were run on the NTS 2023 inputs and on the trained bundle `stability_runs/activitysim/seed_0/models/activitysim_models.pkl` (throwaway scripts, not committed).

## Q1. Stop frequency: what does the local code do compared with ActivitySim's 16-alternative, purpose-segmented model?

### Takeaway
The structure matches: one choice per tour over `{0..3}out × {0..3}in`, segmented by tour purpose, run after tour scheduling and conditioned on tour timing. The utility form does not match. ActivitySim uses a compact, structured utility: shared "any-stop" covariates plus decomposed outbound, inbound and total-stop constants. Local fits an unrestricted MNL with alternative-specific coefficients for every attribute dummy. It also uses continuous tour start and duration where ActivitySim uses hour dummies. "Simplified" describes it better than the "full" fidelity claimed in docs/activitysim.md.

### Cited Findings
- ActivitySim alternatives are the 16 combinations `0out_0in … 3out_3in` (columns `alt,out,in`) — [GH prototype_mtc/configs/stop_frequency_alternatives.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/stop_frequency_alternatives.csv)
- ActivitySim runs an MNL segmented by `SEGMENT_COL: primary_purpose`, with separate specs for work, school, univ, social, shopping, eatout, escort, othmaint, othdiscr and atwork. `primary_purpose` is `univ` for university school tours and `atwork` for subtours — [GH stop_frequency.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/stop_frequency.yaml), [GH stop_frequency_annotate_tours_preprocessor.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/stop_frequency_annotate_tours_preprocessor.csv)
- In the MTC work spec, most covariates (income, household size, female, number of work tours, tour duration ≥ 11 h `(duration > 10)`, AM-peak departure `(start>5)&(start<8)`, accessibility, distance, transit or walk mode) carry one shared coefficient on every non-zero alternative. Constants are split into "outbound stops" constants by out-count, "return stops" constants by in-count, and a "total number of stops" constant. Example values: 1out −0.833, 2out −2.613, 3out −3.934, 1in −0.445, 2in −1.775, 3in −2.139 — [GH stop_frequency_work.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/stop_frequency_work.csv), [GH stop_frequency_coefficients_work.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/stop_frequency_coefficients_work.csv)
- The stop frequency model also creates the trips table: a tour with `2out_0in` has three outbound trips and one inbound trip — [GH abm/models/stop_frequency.py L82-L100](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/stop_frequency.py#L82-L100)
- Local: observed stop counts are capped at 3 per half-tour (`stop_alt`, `MAX_STOPS = 3`). The choice set is the alternatives observed across all purposes; on NTS all 16 are observed. `SegmentedMNL` is fitted with segment = tour purpose and is always segmented, even under `--no-segmentation` (`seg(self.n_min, True)`) — composhed/activitysim/models.py:61-62, 183-207; composhed/activitysim/data.py:28; trained bundle `stop_classes` lists all 16 alternatives
- Local features: the 8 one-hot person labels plus tour duration/1440 and tour start/1440 (`x_stopfreq`) — composhed/activitysim/models.py:57-58
- Local `MNL` has "alternative-specific coefficients" (one coefficient vector per alternative over all features). Purposes with fewer than 200 tours fall back to a pooled model — composhed/common/choice.py:48-49; composhed/activitysim/segmented.py:1-11 (models.py lines as above)
- Local generation samples the alternative after the tour is placed, then drops stops one at a time when `n_out + n_in + 1 > tour minutes`. This is counted in `stops_dropped`: 38 events in the seed-0 run — composhed/activitysim/chain.py:69-82; stability_runs/activitysim/seed_0/generation_report.json
- NTS check (local data): 75,209 tours and only 15,577 stops, about 0.21 stops per tour. 112 tours have more than 3 stops on one half-tour and are truncated to 3 in the stop-frequency label, but every stop is kept in the trip-purpose and departure tables (`min(k, 2)` keys) — composhed/activitysim/models.py:186-204; own computation on `/home/fred/Data/foundata/out/nts/2023/activities.csv`
- Calibration adjusts stop-frequency shares per tour purpose, as ActivitySim calibration practice does with stop constants — composhed/activitysim/chain.py:153-158; docs/activitysim.md:77-82

### Inferences
- Matches: the 4×4 alternative set, segmentation by tour purpose, placement in the chain (after tour time-of-day), and dependence on tour duration and departure time.
- Deviations:
  - Local has no `univ`, `eatout` or `atwork` segments.
  - It has no household, mode, accessibility or distance terms (unavoidable with this data).
  - It has no "number of tours of each type by person" terms, which ActivitySim uses and which the local chain already knows.
  - Its parametrisation is much richer and less structured (≈15 × (≈40 + 2) coefficients per purpose against ~30 shared ones in ActivitySim). With purposes such as othmaint (2,000 tours) and roughly 0.2 stops per tour, this risks overfitting the rare 3out/3in cells. The L2 penalty (1e-4) is light.
- docs/activitysim.md:62 rates stop frequency as "full". Given the utility form, "simplified (same alternatives and segmentation; unstructured MNL)" is more defensible.

### Gaps
- I did not check whether local MNL coefficient estimates on rare alternatives are stable across seeds; `validation.json` would be the place to look.

## Q2. Trip purpose: ActivitySim's lookup by tour purpose, direction, person type and departure period. What is the local equivalent?

### Takeaway
Local uses a probability lookup (`Lookup`), which matches ActivitySim's model form. The conditioning keys differ: local uses (tour purpose, direction, stop index ≤ 2) with back-off, while ActivitySim uses (primary purpose, outbound, person type, tour-start-hour range). Person type and time-of-day conditioning are both missing locally, and stop index is added. Last-trip purposes are handled equivalently.

### Cited Findings
- ActivitySim default `probs_join_cols = ["primary_purpose", "outbound", "person_type"]`, `use_depart_time: True`. When the probs table has `depart_range_start` and `depart_range_end`, the row is selected by `start` within the range — [GH abm/models/trip_purpose.py L30, L52-L100, L172-L178](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_purpose.py#L30)
- In prototype_mtc, `start` for trip purpose is the tour start (`reindex_i(tours.start, df.tour_id)`), because trip departure times are not yet known when trip purpose runs — [GH trip_purpose_annotate_trips_preprocessor.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/trip_purpose_annotate_trips_preprocessor.csv)
- MTC `trip_purpose_probs.csv` has columns `primary_purpose,outbound,depart_range_start,depart_range_end,person_type,work,univ,school,escort,shopping,othmaint,eatout,social,othdiscr`. Work tours are split into depart ranges 5–8 and 9–23. For example, outbound stops on full-time workers' work tours are 46.6% escort in 5–8 and 17.2% escort in 9–23. Most other purposes use one range, 5–23 — [GH trip_purpose_probs.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/trip_purpose_probs.csv)
- ActivitySim sets the last outbound trip's purpose to the primary purpose and the last inbound trip's purpose to home (work for at-work subtours). Only intermediate stops are drawn from the table — [GH abm/models/trip_purpose.py L236-L254](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_purpose.py#L236-L254)
- ActivitySim's stop purpose alternatives include mandatory purposes (work, univ, school), so a work stop on an escort tour is possible — [GH trip_purpose_probs.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/trip_purpose_probs.csv)
- Local: `Lookup(STOP_ACTS)` is keyed by `(tour purpose, direction, min(k,2))`. Outcomes are the 5 discretionary NTS types only (`STOP_ACTS = sorted(DISC_ACTS)`). It backs off to shorter key prefixes when a cell has fewer than 20 observations (`min_count=20`) — composhed/activitysim/models.py:36, 69-96, 193, 208; composhed/activitysim/chain.py:106-108
- Local stops can never be work or education: tour extraction merges all mandatory episodes in a tour into the primary block, and only `DISC_ACTS` become stops — composhed/common/tours.py:128-149
- Trained bundle examples: shopping tour, inbound stop 0 → shop 0.489; work tour, outbound stop 0 → escort 0.756 — own probe of `stability_runs/activitysim/seed_0/models/activitysim_models.pkl`
- NTS contains 4,768 consecutive identical activity rows out of 223,006 (other 1,943; escort 1,323; shop 1,237), for example shop→shop at different zones. So the lookup learns same-type adjacency. Local `finalise` merges consecutive identical activities, and the seed-0 generation reports 4,797 merges across 3,986 persons (7.0%) — own query on activities.csv; composhed/common/schedule.py:38-68; stability_runs/activitysim/seed_0/generation_report.json

### Inferences
- The form matches (empirical lookup, last-trip purpose implied).
- Deviations:
  - No person-type segmentation, although the variant already computes `ptype`. Adding it to the key would be a trivial fidelity gain.
  - No time-of-day range.
  - It adds a stop-index key, which ActivitySim does not use.
  - It has a back-off hierarchy; ActivitySim has none, and an unmatched row logs an error.
  - Mandatory stop purposes are excluded.
- The merge interaction is a local artefact. ActivitySim keeps same-purpose consecutive trips as separate trips because they have different destinations. Local, which has no locations, collapses them. A sampled stop is therefore silently lost about 4,800 times per 56,734 persons, and the realised stop count falls below the sampled stop-frequency alternative. Calibration targets the sampled alternatives (`trace["stop_alts"]`), not the realised ones, so it cannot see this.

### Gaps
- I did not measure how the merge loss splits between "stop equals primary" and "stop equals adjacent stop".

## Q3. Trip scheduling (and trip_scheduling_choice): how do lookup offsets, retries and failure fallback compare, and how are within-tour activity durations derived?

### Takeaway
Local uses a probabilistic departure-period lookup clipped to the tour window, like ActivitySim's default "departure" mode, but it differs materially:
- Its key is (mandatory/non-mandatory class, direction, stop index). It has no tour purpose and no `tour_hour`, so trip departures are not anchored to the tour's start or end hour.
- It schedules inbound trips forward, not backward from tour end.
- It renormalises clipped probabilities instead of failing and retrying.
- It adds minute-level resolution, from which stop durations are derived.

Probing the trained model shows this distorts stop durations badly. In NTS, a single outbound stop on a work tour starting 06–08h has a median duration of 15 minutes. The local model generates a median of 90 minutes for a 07:00–17:00 work tour.

### Cited Findings
- ActivitySim default (probabilistic) trip scheduling assigns "a departure hour based on an input lookup table of percents by tour purpose, direction (inbound/outbound), tour hour, and trip index". The tour hour is the tour start hour for outbound trips and the tour end hour for inbound trips, with up to four trips per half-tour — [ActivitySim v1.6.0 docs: Trip Scheduling (Probabilistic)](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/trip_scheduling_probabilistic.html); [GH abm/models/trip_scheduling.py L72-L93](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_scheduling.py#L72-L93)
- In MTC `trip_scheduling_probs.csv` (1,367 rows), columns are `primary_purpose,outbound,tour_hour,trip_num,HR5…HR23`, which are absolute clock hours with `DEPART_ALT_BASE: 5`. The table is keyed per tour purpose (work, school, univ, escort, shopping, eatout, othmaint, social, othdiscr, atwork). For example, work/outbound/tour_hour 5/trip 1 puts probability 1 on HR5 (the first trip departs at tour start). Settings: `MAX_ITERATIONS: 100`, `FAILFIX: choose_most_initial` — [GH trip_scheduling.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/trip_scheduling.yaml), [GH trip_scheduling_probs.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/trip_scheduling_probs.csv)
- Order of scheduling in ActivitySim: "The first outbound trip starts at the tour start time, and subsequent outbound trips are processed in trip_num order… Inbound trips are handled similarly, except in reverse order, starting with the last trip, and working backwards". The last inbound trip is anchored at the tour end hour. After the outbound leg, inbound `earliest` is raised to the maximum outbound departure — [GH trip_scheduling.py L471-L482, L251-L307, L133-L191](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_scheduling.py#L463-L530)
- Clipping in ActivitySim: probabilities outside `[earliest, latest]` are zeroed (`_clip_probs`). Only the first trip in a leg is renormalised. For later trips the lost mass becomes an explicit `fail` alternative (`1 - sum(probs)`). Failed trips and their leg-mates are rescheduled for up to `MAX_ITERATIONS`. On the last iteration, `choose_most_initial` assigns the previous trip's bound (`ADJUST_NEXT_DEPART_COL`); the alternative is `drop_and_cleanup` — [GH abm/models/util/probabilistic_scheduling.py L16-L62, L120-L128, L196](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/probabilistic_scheduling.py#L16-L62); [GH trip_scheduling.py L322-L330, L570-L625](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_scheduling.py#L570-L625)
- `earliest`/`latest` bounds are inclusive in ActivitySim (`ix_map >= earliest` and `<= latest`), so several trips can depart in the same hour — [GH probabilistic_scheduling.py L52-L58](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/probabilistic_scheduling.py#L52-L58)
- Alternative modes in v1.6.0: `scheduling_mode ∈ {departure, stop_duration, relative}`. In `relative` mode (prototype_mwcog), trips are scheduled relative to the previous trip's departure, keyed on `periods_left`, `outbound`, `tour_purpose_grouped` (mandatory vs non-mandatory) and `half_tour_stops_remaining_grouped` (0 / 1+). This mode cannot fail and requires `logic_version: 2` — [docs Trip Scheduling (Probabilistic)](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/trip_scheduling_probabilistic.html); [GH trip_scheduling.py L45-L53, L441-L457](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_scheduling.py#L441-L457)
- `trip_scheduling_choice` (used in prototype_arc together with `trip_departure_choice`) is a logit over how the tour duration splits into outbound leg, main leg (last outbound stop → first inbound stop) and inbound leg. `trip_departure_choice` then picks per-trip stop-time durations within each leg by interaction_simulate — [GH docs trip_scheduling_choice.md](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/trip_scheduling_choice.md); [GH abm/models/trip_scheduling_choice.py L44-L102](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_scheduling_choice.py#L44-L102); [GH abm/models/trip_departure_choice.py L78-L163](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_departure_choice.py#L78-L163); [GH prototype_arc/configs/settings.yaml L129-L134](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_arc/configs/settings.yaml#L129-L134)
- Local `trip_departure = Lookup(range(grid.n))` is keyed by `(tour_class, direction, min(k,2))`, where `tour_class` is only mandatory or non-mandatory. Outcomes are absolute clock periods (`grid.end_period_of(dep)`). Training departures are taken from each outbound stop, and for inbound from `primary_end` and each inbound stop except the last — composhed/activitysim/models.py:65-66, 186-209
- Local generation (`_stops`, `_departure`):
  - Outbound departures are sampled forward from tour start S.
  - Inbound departures are also sampled forward, starting from the last outbound departure; this is not ActivitySim's backward order.
  - The last inbound stop ends at tour end E.
  - Each draw zeroes periods outside `[lo, hi]` and renormalises (`p / p.sum()`), with a uniform fallback if all mass is zero. There is no fail alternative and no leg-level retry.
  - The minute inside the period comes from `IntraPeriodSampler` keyed only by direction.
  - Bounds force at least 1 minute per remaining stop and for the primary.
  — composhed/activitysim/chain.py:69-123; composhed/common/periods.py:59-112
- Local activity durations: outbound stop k lasts `d_k − d_{k−1}`, starting at S. The primary lasts from the last outbound departure to the first inbound departure. Inbound stops run between successive inbound departures, with the last ending at E. `tours_to_rows` lays these out and fills gaps with home — composhed/activitysim/chain.py:84-104; composhed/common/tourgen.py:100-114
- Probe of the trained seed-0 model (2,000 draws each, stop count forced):
  - Work tour 07:00–17:00 with `1out_0in`: generated outbound stop median 90 min (p10 43, p90 180).
  - Work tour 07:00–17:00 with `0out_1in`: generated inbound stop median 90 (p90 250).
  - Work tour 14:00–22:00 with `0out_1in`: generated inbound stop median 360 min (p10 242), leaving about 2 h of work.
  - Observed NTS, one outbound stop on work tours starting 06–08h: median 15 min (p10 5, p90 75, n=191).
  - Observed NTS, one inbound stop on work tours ending 16–18h: median 40 (n=424).
  - Observed NTS, tours ending 21–23h: median 240 (n=95), probably dominated by merged fragmented-work patterns.
  — own scripts against composhed/activitysim/chain.py `_stops` and `build_records`
- Local training records for fragmented mandatory tours (work → stop → work) put the in-block stop as an inbound stop, and `primary_end` is computed as `tour end − Σ inbound durations`. The recorded inbound departure sequence can therefore be non-monotone (for example 16:00 from the primary, then 13:00 from the in-block stop). NTS counts: 275 fragmented mandatory tours, 265 tours with an inbound stop that starts before `primary_end`, and 47 tours with non-monotone recorded inbound departures — composhed/activitysim/models.py:196-199; composhed/activitysim/data.py:84-92 (`_asim_tour`); composhed/common/tours.py:128-149; own computation
- Calibration does not touch trip purpose or trip departure; only stop-frequency shares and tour TOD marginals are targeted — composhed/activitysim/chain.py:136-172

### Inferences
- What matches:
  - Empirical lookup of departure periods, clipped to the tour window.
  - First outbound trip at tour start and last inbound trip at tour end.
  - Outbound trips monotone in trip order.
  - No mode-choice logsums.
  - Grouping into mandatory/non-mandatory classes resembles prototype_mwcog's `tour_purpose_grouped` in relative mode.
- Deviation 1 (most important): **no `tour_hour` conditioning and no relative indexing.** ActivitySim's departure mode conditions on the tour start hour (outbound) or end hour (inbound). Its relative mode conditions on periods left. Local conditions on neither, so a stop's departure is drawn from the population-wide clock-time distribution for that class, truncated to the tour. This is the mechanism behind the too-long stops (90 min vs 15 min observed) and the "inbound stop eats the work day" result. Adding `tour_hour` to the key, or switching to relative offsets, would bring it in line with ActivitySim and fix much of the bias.
- Deviation 2: **inbound trips are scheduled forward rather than backward from tour end.** ActivitySim anchors inbound trips to the tour end and works backwards.
- Deviation 3: **renormalise instead of fail/retry.** For a single trip, a renormalised truncated draw has the same distribution as rejection. But ActivitySim retries the whole leg when a later trip fails, which changes the joint distribution towards jointly feasible legs. Local never fails, so `FAILFIX` has no analogue. This is reasonable, but it should be documented as such.
- Deviation 4: **no purpose-specific tables.** ActivitySim keys on primary purpose; local pools all non-mandatory purposes, and work with school.
- Deviation 5: **minute-level durations.** These are a local construction (`IntraPeriodSampler`). ActivitySim only outputs departure periods.
- Bug/defect: fragmented mandatory tours produce non-monotone and synthetic inbound departure observations in training (47 tours, so small in magnitude). The at-work-subtour collapse puts in-block stops on the inbound leg, which ActivitySim would model as an at-work subtour.
- Not used locally: `trip_scheduling_choice` / `trip_departure_choice` (the leg-duration logit variant). Its explicit main-leg duration choice would address the primary-duration problem directly.

### Gaps
- I did not compare the synthetic-population distribution of stop durations against NTS at scale; the probe used forced stop counts and zero label vectors. The 90-minute vs 15-minute contrast is indicative, not a full validation.
- The exact meaning of `trip_num` in MTC probs when there are more than 4 trips per leg was not checked.

## Q4. Time periods: hourly vs 30-minute, and how local maps to them

### Takeaway
Local defaults to an hourly grid with an extra 00:00–05:00 early period: 20 periods and 210 (start ≤ end) pairs. prototype_mtc uses hours 5…23 (19 periods, 190 tdd alternatives), where 5 absorbs everything before 06:00 and 23 everything from 23:00. Newer examples (SEMCOG, ARC, MWCOG) use 48 half-hour periods (1,176 tdd alternatives) with period 1 = 03:00. Local `--time-resolution 30` gives 1 + 38 periods, not ActivitySim's 48.

### Cited Findings
- prototype_mtc `network_los.yaml`: `skim_time_periods: time_window: 1440, period_minutes: 60, periods: [0, 3, 5, 9, 14, 18, 24], labels: ['EA','EA','AM','MD','PM','EV']` — [GH prototype_mtc/configs/network_los.yaml L30-L34](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/network_los.yaml#L30-L34)
- prototype_mtc `tour_departure_and_duration_alternatives.csv` runs from `5,5` to `23,23`, giving 190 alternatives. The work scheduling spec labels `start < 6` as "Departure Constants -- Early (up to 5)" — [GH tour_departure_and_duration_alternatives.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_departure_and_duration_alternatives.csv); [GH tour_scheduling_work.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_scheduling_work.csv)
- production_semcog: `period_minutes: 30`, `periods: [0, 7, 12, 23, 31, 48]`, tdd alternatives `1,1 … 48,48` (1,176), trip scheduling `DEPART_ALT_BASE: 1`, `MAX_ITERATIONS: 50`. Period 24 is labelled "02:30 PM to 03:00 PM", which implies period 1 = 03:00–03:30. prototype_arc and prototype_mwcog also use `period_minutes: 30` — [GH production_semcog/configs/network_los.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/production_semcog/configs/network_los.yaml); [GH production_semcog tour_scheduling_school.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/production_semcog/configs/tour_scheduling_school.csv); [GH prototype_arc network_los.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_arc/configs/network_los.yaml); [GH prototype_mwcog network_los.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mwcog/configs/network_los.yaml)
- In ActivitySim, skim time periods (EA/AM/MD/PM/EV) are a separate, coarser layer used for LOS lookups. The tour/trip scheduling periods are the tdd hours or half-hours — [GH prototype_mtc network_los.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/network_los.yaml)
- Local: `PeriodGrid.with_early_bin(resolution=60, early_end=300)` gives edges `[0, 300, 360, …, 1440]`, which the trained bundle confirms (`grid.n = 20`). `TourTOD` uses all pairs with start ≤ end (210). End minutes on a period edge belong to the earlier period (`end_period_of`) — composhed/common/periods.py:30-31, 46-48; composhed/activitysim/models.py:170; docs/activitysim.md:35, 58
- Local intra-period minutes come from an empirical within-period histogram (`IntraPeriodSampler`, min 20 observations per cell, else a pooled relative-offset histogram, else uniform) — composhed/common/periods.py:59-112
- ActivitySim's timetable allows one tour to end and the next to start in the same period (`I_END` with `I_START` is not in `COLLISIONS`) — [GH core/timetable.py L21-L43](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/core/timetable.py#L21-L43). Local reserves minutes with a 10-minute home buffer between tours and at least 1 minute of home at each end of the day — composhed/common/timewindow.py:46-58

### Inferences
- Local hourly mode is MTC-like but not identical:
  - The early bin is split into 00:00–05:00 and 05:00–06:00 rather than one "≤ 5" period.
  - The last period is 23:00–24:00 rather than "23 or later". The 24-hour diary window cannot represent tours that cross midnight; ActivitySim's day also ends at period 23 or 48.
- For 30-minute fidelity, a `uniform(30)` grid starting at 03:00 with 48 periods would match SEMCOG/MWCOG. The current `with_early_bin(30)` keeps a 5-hour early bin.
- The `trip_departure` lookup uses the same grid, so at the default hourly resolution departures inside a tour are quantised to hours before minute jitter. That is ActivitySim-faithful, but it drives the duration inflation in Q3.

### Gaps
- I did not find documentation of exactly how MTC TM1 period 5 treats departures before 05:00 (whether 03:00–05:59 is all in period 5). The "Early (up to 5)" label implies it.

## Q5. Home stays and out-of-home activity durations: how ActivitySim represents them, and how faithfully local derives them

### Takeaway
ActivitySim never outputs activity episodes or durations. It outputs tours (start/end/duration in periods) and trips (depart period). Stay durations are implicit: gaps between trip departures within a tour, and timetable gaps between tours for home. Local reconstructs a minute-level sequence from these same ingredients (zero travel time, stop duration = gap between departures, primary = residual, home = gaps between tours plus a 10-minute buffer). The construction is logically consistent with ActivitySim's semantics but is necessarily an invention. Its realism depends on the departure model, which (Q3) is not anchored to tour timing.

### Cited Findings
- ActivitySim trip scheduling "assigns depart times for trips within the start, end limits of the tour", in periods. No arrival or activity-duration field is produced by the probabilistic model — [GH trip_scheduling.py L471-L473](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_scheduling.py#L463-L530)
- Tours carry `start`, `end` and `duration` from the tdd alternatives, and the per-person timetable encodes the periods in use (`tour_map` writes start/middle/end codes) — [GH core/timetable.py L207, L360-L373](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/core/timetable.py#L207)
- The ARC-style `trip_scheduling_choice` does explicitly choose outbound, main and inbound leg durations. This is the closest ActivitySim analogue to a primary-activity duration — [GH docs trip_scheduling_choice.md](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/trip_scheduling_choice.md)
- Local assembly: `tours_to_rows` builds outbound stops → primary (`t.end − Σ inbound`) → inbound stops for each tour, and `from_episodes` fills every gap with `home`. `finalise` rounds cumulative bounds, drops zero-length episodes and merges consecutive identical activities. `check_rows` then enforces sum = 1440, home at both ends, no consecutive repeats and integer contiguity — composhed/common/tourgen.py:100-114; composhed/common/schedule.py:1-91
- Local home stays between tours are at least `home_inter = 10` minutes (TimeWindow buffer), and the day starts and ends with at least 1 minute of home — composhed/activitysim/models.py:100-101; composhed/common/timewindow.py:46-58; docs/activitysim.md:89-90
- Local tours are generated in ActivitySim priority order (mandatory, then escort → shopping → othmaint → social → othdiscr) and then sorted by time. Activity names are mapped back to NTS types (`NTS_ACT`) before layout — composhed/activitysim/chain.py:33-67; composhed/activitysim/data.py:21-25

### Inferences
- Faithful: home time is purely residual (timetable gaps) and stop durations equal the departure gaps, both consistent with ActivitySim under zero travel time.
- Not faithful, or invented:
  - Minute-level resolution.
  - The 10-minute home buffer. ActivitySim allows same-period end and start, so a 0–59-minute home gap is possible at hourly resolution.
  - Forward inbound scheduling.
- Primary activity duration is never modelled directly; it is a residual. Combined with the clock-time departure lookup, this produces implausibly short primaries in late-starting tours (Q3 probe). The ARC `trip_scheduling_choice` leg-duration logit would be the faithful way to model it explicitly.

### Gaps
- None beyond those in Q3.

## Q6. Summary list: matches, deviations, simplifications, errors and bugs

### Takeaway
The trip-level chain has the right skeleton (stop frequency → trip purpose → trip scheduling, all lookup or MNL as in ActivitySim). Its fidelity is overstated in docs/activitysim.md (stop frequency and trip purpose are marked "full"). The main substantive defect is the missing `tour_hour` / relative conditioning in trip scheduling, which visibly inflates stop durations. The others are an unmodelled stop loss through activity merging and a small training-data inconsistency for fragmented work tours.

### Cited Findings
- **Matches:**
  - The 16 stop alternatives — models.py:61-62, [GH alternatives](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/stop_frequency_alternatives.csv)
  - Segmentation by tour purpose — models.py:206, [GH stop_frequency.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/stop_frequency.yaml)
  - Trip purpose by probability lookup with implied last-trip purposes — models.py:69-96, [GH trip_purpose.py](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_purpose.py#L236-L254)
  - Departure-period lookup clipped to the tour window, first outbound trip at tour start, last inbound trip at tour end — chain.py:69-123, [GH trip_scheduling.py](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/trip_scheduling.py#L463-L530)
  - Hourly tdd-style grid — periods.py:30-31
- **Deviations and simplifications:**
  - Unstructured alternative-specific MNL for stop frequency instead of decomposed out/in/total constants with shared covariates.
  - No univ, eatout or atwork segments.
  - No person type or tour-hour range in trip purpose; stop index added; back-off.
  - Trip departure keyed by tour class rather than purpose, with no `tour_hour`.
  - Inbound scheduled forward.
  - Renormalisation instead of fail/retry with `choose_most_initial`.
  - Minute-level intra-period sampling.
  - Extra 00:00–05:00 period.
  - 10-minute home buffer.
  - Discretionary-only stop purposes.
  - At-work subtours collapsed into inbound stops.
  - Sources: see Q1–Q5 citations.
- **Errors/bugs (local):**
  1. Trip departure lookup without `tour_hour`, which yields stop durations of about 90 minutes median against 15 observed, and inbound stops consuming most of late work tours (chain.py:110-123; probe in Q3).
  2. Stop loss from merging same-type adjacent episodes (4,797 merges, 7.0% of persons), invisible to calibration (schedule.py:38-68; generation_report.json).
  3. Non-monotone or synthetic inbound departure observations for fragmented mandatory tours (models.py:196-199; 47 tours).
  4. The `--no-segmentation` ablation does not affect stop frequency, which is always purpose-segmented (models.py:206). This may be intended, but it is undocumented in the ablation table (docs/activitysim.md:107).
  5. Documentation: docs/activitysim.md:62-64 rates stop frequency and trip purpose "full", which overstates fidelity. Line 64 says ActivitySim's trip scheduling table is "relative to the tour's hours". In fact MTC's default table uses absolute clock-hour columns conditioned on `tour_hour`; only the optional `relative` mode is relative ([GH trip_scheduling_probs.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/trip_scheduling_probs.csv)).

### Inferences
- The cheapest fidelity fixes:
  - Add `tour_hour` (tour start period for outbound, tour end period for inbound) and tour purpose to the `trip_departure` key, or adopt the MWCOG relative mode keyed on periods left.
  - Add `ptype` to the `trip_purpose` key.
  - Schedule inbound trips backwards from E.
  - Forbid a stop purpose equal to the adjacent activity (or count realised stops in calibration).
  - Correct the docs/activitysim.md fidelity ratings and the line-64 description.

### Gaps
- I did not re-run the full generation with any fix, so the improvement in evaluation metrics from fixing bug 1 is unquantified.
