# ActivitySim variant fidelity: cross-cutting machinery, calibration, RNG, pipeline scope

Scope: the local ActivitySim-aligned variant (`composhed/activitysim/`, shared `composhed/common/`) compared with ActivitySim v1.6.0. v1.6.0 was released on 2026-09-17 ([GitHub releases API](https://api.github.com/repos/ActivitySim/activitysim/releases)). Most source and config files were read from the `main` branch shortly after that release, so they may differ slightly from the v1.6.0 tag. Docs come from https://activitysim.github.io/activitysim/v1.6.0/. Local paths are relative to `/home/fred/Projects/composhed`.

Upstream file URLs used below:
- `random.py`: https://github.com/ActivitySim/activitysim/blob/main/activitysim/core/random.py
- `tour_frequency.py`: https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/util/tour_frequency.py
- `vectorize_tour_scheduling.py`: https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/util/vectorize_tour_scheduling.py
- `non_mandatory_tour_frequency.py`: https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/non_mandatory_tour_frequency.py
- `trip_scheduling.py`: https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/trip_scheduling.py
- prototype_mtc configs: https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs

---

## Q1. Choice model forms: which local forms match ActivitySim and which are substituted?

### Takeaway
The ActivitySim variant uses only three kinds of model: an MNL (like ActivitySim's `simple_simulate`), a conditional logit over a start × end table with availability masks (like `interaction_sample_simulate` in tour scheduling), and empirical probability lookups (like ActivitySim's trip_purpose and trip_scheduling tables). In that sense the forms match ActivitySim more closely than those of the CEMDAP or DaySim variants. Ordered logit and AFT/log-normal duration models are **not** used in this variant. A log-normal duration appears only in the `--independent-timing` ablation, which is labelled as such. The one real substitution is in non-mandatory tour frequency: ActivitySim's `interaction_simulate`, with generic coefficients on alternative attributes, becomes a plain MNL with a full coefficient vector for every combination alternative. The docs do not mention this substitution. Neither side uses nested logit in these components: ActivitySim uses NL only in mode choice, which is out of scope here.

### Cited Findings
- **ActivitySim's model types.** The docs list four kinds of expression configuration:
  - "Simple Simulate", a choice from a fixed set of alternatives;
  - "Simulate with Interaction", where expressions are combined with an alternatives file;
  - "Combinatorial", used by CDAP;
  - nested logit, whose "nesting structure (including nesting coefficients) is specified in the YAML settings file".
  — [ActivitySim v1.6.0 Core docs](https://activitysim.github.io/activitysim/v1.6.0/core.html)
- **`interaction_sample_simulate`** is documented as running "a simulation in the situation in which alternatives must be merged with choosers because there are interaction terms or because alternatives are being sampled". — [Core docs, API](https://activitysim.github.io/activitysim/v1.6.0/core.html)
- **Tour scheduling** imports and calls `interaction_sample_simulate` (`vectorize_tour_scheduling.py` line 23). It also computes mode-choice logsums via `simulate.simple_simulate_logsums` (line 251). — [vectorize_tour_scheduling.py](https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/util/vectorize_tour_scheduling.py)
- **Non-mandatory tour frequency** uses `interaction_simulate` (`non_mandatory_tour_frequency.py` lines 32 and 325). It is not `simple_simulate`. — [non_mandatory_tour_frequency.py](https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/non_mandatory_tour_frequency.py)
- **The NMTF alternatives table** has 96 alternatives, with columns `escort,shopping,othmaint,othdiscr,eatout,social,tot_tours`. Its spec uses alternative-attribute expressions, e.g. `(tot_tours == 0) & (num_mand == 0) & (num_hh_joint_tours == 0)`, and a per-person-type coefficient file such as `coef_escorting_tour`. — [prototype_mtc non_mandatory_tour_frequency_alternatives.csv / .csv spec](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Logit types in prototype_mtc.** `stop_frequency.yaml` and `non_mandatory_tour_scheduling.yaml` both declare `LOGIT_TYPE: MNL`. — [prototype_mtc configs](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **ActivitySim trip purpose and trip scheduling are probability lookups, not logit models.**
  - `trip_purpose.yaml` has `PROBS_SPEC: trip_purpose_probs.csv`, with columns `primary_purpose,outbound,depart_range_start,depart_range_end,person_type,<purposes>`.
  - `trip_scheduling.yaml` has `PROBS_SPEC: trip_scheduling_probs.csv`, with columns `primary_purpose,outbound,tour_hour,trip_num,HR5..HR23`, plus `MAX_ITERATIONS: 100` and `FAILFIX: choose_most_initial`.
  — [prototype_mtc configs](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Local MNL** (`common/choice.py:48-109`):
  - alternative-specific coefficients on every covariate plus a constant (`X1 = [1, X]`, `W` is K×J with the first column normalised to 0);
  - availability masks;
  - L-BFGS-B with an L2 ridge of 1e-4 applied to **all** parameters, constants included (`choice.py:92`).
  — [local choice.py](composhed/common/choice.py)
- **Local ConditionalLogit** (`choice.py:293-397`) has utility `V = F·γ + z·B·G + offset`:
  - sparse start-period, end-period and duration-bin dummies;
  - dense start/end/duration/duration² shape terms, interacted with person features;
  - availability masks.
  `TourTOD` (`common/tod.py:652-735`) wraps it over all (start ≤ end) period pairs. — [local tod.py](composhed/common/tod.py)
- **ActivitySim variant composition.** It uses `SegmentedMNL` for CDAP, mandatory tour frequency, non-mandatory tour frequency and stop frequency (`activitysim/models.py:462-527`), `TourTOD` for scheduling (`models.py:491-502`) and `Lookup` tables for trip purpose and departure (`models.py:390-417, 529-531`). `grep` finds no use of `OrderedLogit`, hazard or LogNormal in `composhed/activitysim/`. A log-normal duration enters only through `IndependentTiming` (`tod.py:738-764`) when `--independent-timing` is set. — [local models.py](composhed/activitysim/models.py)
- **Local non-mandatory frequency** is an MNL over observed purpose-count vectors with support ≥ 30 (`models.py:473-487`). Each alternative gets its own coefficients on all one-hot labels, plus `cdap=="M"` and `n_mand/2` (`x_nmtf`, `models.py:374-375`). — [local models.py](composhed/activitysim/models.py)
- **Local docs on NMTF** describe it as "Multinomial logit per person type over the combinations seen at least 30 times" and label it "simplified". They do not say that ActivitySim uses `interaction_simulate` with generic, alternative-attribute coefficients. — [docs/activitysim.md:59](docs/activitysim.md)

### Inferences
- Treating MNL as the analogue of `simple_simulate` and the tdd conditional logit as the analogue of `interaction_sample_simulate` is a fair claim. The utility parameterisations differ, though:
  - ActivitySim specs are hand-written and sparse, with constrained coefficients.
  - The local MNL is a fully saturated "label one-hots × alternative" model. That is closer to statsmodels `MNLogit` than to an ActivitySim spec.
- The local NMTF has (K+1)·(J−1) parameters per person type, where J is the number of kept combinations. That is far more than ActivitySim's generic NMTF spec. The local docs should say so.
- Ridge on the ASCs (`choice.py:92`) slightly shrinks the estimated constants towards the share-initialised start values. Calibration then corrects them, so the effect is small, but it is not maximum likelihood in the way larch estimation is.

### Gaps
- The v1.6.0 docs say nothing on whether tour scheduling in prototype_mtc samples alternatives or uses the full tdd set. The code passes the tdd alternatives through `interaction_sample_simulate`, but sample sizes were not checked.

---

## Q2. Utility specification conventions, ASCs and person-type segmentation

### Takeaway
ActivitySim specifies utilities in expression CSVs plus coefficient CSVs, with optional coefficient templates. Person-type segmentation takes two forms:
- a separate coefficient file per person type, through `SPEC_SEGMENTS`, as in NMTF;
- person-type indicator expressions and ASC rows inside a single spec, as in CDAP, mandatory tour frequency and tour scheduling.

The local variant instead fits a fully separate MNL per person type for CDAP, MTF and NMTF, pooling segments with fewer than 200 observations. Its person-type segmentation is therefore stronger than ActivitySim's for CDAP and MTF. Its person-type definitions are an approximation: 7 types rather than 8. Stop frequency is segmented by tour purpose, as in ActivitySim. Local ASCs exist as the MNL intercept plus a calibration `offset` vector.

### Cited Findings
- **Spec CSV conventions.** Each spec row has a Label, a Description and an Expression, plus "a column for each alternative and its relevant coefficient from the submodel coefficient file". The Label "is the unique expression name (used for model estimation integration)". — [Core docs](https://activitysim.github.io/activitysim/v1.6.0/core.html)
- **NMTF segments.** `non_mandatory_tour_frequency.yaml` sets `SEGMENT_COL: ptype` and `SPEC_SEGMENTS` for all 8 person types (PTYPE_FULL, PART, UNIVERSITY, NONWORK, RETIRED, DRIVING, SCHOOL, PRESCHOOL). Each has its own `non_mandatory_tour_frequency_coefficients_PTYPE_*.csv`. — [prototype_mtc configs](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **MTF uses one spec and one coefficient file.** Person-type ASC rows include `util_ft_worker,...,ptype == 1,0,coef_ft_worker_work2_asc` and interactions such as `(ptype == 1) & female`. The coefficient file includes `coef_unavailable,-999,T` for unavailable alternatives. — [prototype_mtc mandatory_tour_frequency.csv / _coefficients.csv](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Tour scheduling (work) person-type terms** are shift terms inside one spec, e.g. `(ptype == 2) * start` and `(ptype == 1) & (duration < 9)`. The spec also has grouped departure constants (`start < 6`, `start == 6`, …, `start > 21`), arrival constants and duration constants. — [prototype_mtc tour_scheduling_work.csv](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Stop frequency** is segmented by `primary_purpose` across 10 purposes (work, school, univ, social, shopping, eatout, escort, othmaint, othdiscr, atwork), each with its own SPEC and COEFFICIENTS. — [prototype_mtc stop_frequency.yaml](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **CDAP person-type constants.** `cdap.yaml` defines 8 person-type constants: FULL=1, PART=2, UNIVERSITY=3, NONWORK=4, RETIRED=5, DRIVING=6, SCHOOL=7, PRESCHOOL=8. It also points to `cdap_indiv_and_hhsize1.csv`, `cdap_coefficients.csv` and `cdap_interaction_coefficients.csv`. — [prototype_mtc cdap.yaml](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Coefficient templates.** The estimation docs refer to "the spec, coefficients, and coefficients template file for the tour mode choice model" (notebook 17). — [Estimation mode / Larch docs](https://activitysim.github.io/activitysim/v1.6.0/users-guide/estimation-mode/larch.html)
- **Local segmentation:**
  - `SegmentedMNL` fits a pooled MNL plus one MNL for each segment with n ≥ `n_min` (default 200) (`activitysim/segmented.py:30-45`).
  - CDAP, MTF and NMTF are segmented by person type (`models.py:465, 469-471, 485-487`).
  - Stop frequency is segmented by tour purpose (`models.py:527`).
  - Scheduling has person-type one-hot terms in `z` (`models.py:369-371`).
  — [local segmented.py](composhed/activitysim/segmented.py)
- **Local person types** (`persontype.py:1-42`): 7 types, with CHILD covering pre-school and non-driving ages. A student aged 10-21 becomes `DRIVING_STUDENT`; only students aged 21 or over become `UNIV`. The 2026-10-27 stability runs (seed 0) report person-type counts of FTW 21,878, RETIRED 14,997, PTW 6,685, CHILD 5,770, DRIVING_STUDENT 4,375, NONWORK 2,686 and UNIV 343, and 142 unmapped records. — [local persontype.py](composhed/activitysim/persontype.py); [stability_runs/activitysim/seed_0/validation.json](stability_runs/activitysim/seed_0/validation.json)
- **Local offsets.** Every local model carries an `offset` vector over alternatives that calibration adjusts (`choice.py:15-16, 71, 124`). For TOD models, the offsets are start-period, end-period and duration-bin constants (`tod.py:677-692`). — [local choice.py](composhed/common/choice.py)

### Inferences
- **The docs overstate how ActivitySim segments.** `docs/activitysim.md:5-6` says ActivitySim uses "separate coefficients for each person type". That is literally true only for NMTF, which has per-type coefficient files. CDAP, MTF and scheduling use shared specs with person-type-specific terms. Fitting fully separate MNLs per type for CDAP and MTF therefore departs from ActivitySim, and the departure is not recorded.
- **University students are probably misassigned.** ActivitySim classes university students aged 18 or over as UNIV. The local rule sends NTS students aged 18-20 to DRIVING_STUDENT, because of the 10-21 age bin. This may help explain why UNIV is so small (343).
- **No hard availability rules.** ActivitySim uses `-999` / `coef_unavailable` terms, e.g. so that non-workers cannot choose work tours. Locally, availability is purely data-driven: an alternative unseen in a segment just gets a very low estimated utility.

### Gaps
- The person-type rules in prototype_mtc's `annotate_persons.csv` (age and employment thresholds) were not read in detail. The 18+ UNIV rule above comes from general knowledge of CT-RAMP and was not verified in the downloaded file.

---

## Q3. Logsums, accessibility, pipeline order and omitted components

### Takeaway
The prototype_mtc pipeline has about 30 steps. Many of them (accessibility, location choice, auto ownership, free parking, joint tours, at-work subtours, destination and mode choice) are out of scope locally. The local docs list only "Location, mode, travel time, joint tours, at-work sub-tours" as absent. They omit:
- accessibility terms;
- auto ownership: the NTS `vehicles` attribute is used as an exogenous input instead;
- free parking, school escorting, telecommute/work-from-home and other optional components;
- mode-choice logsums in scheduling;
- residual-time-window terms in NMTF.

There is also an ordering difference. Locally, NMTF is sampled **before** any tour is scheduled. In ActivitySim, mandatory tour scheduling runs before NMTF, and NMTF uses the residual time window.

### Cited Findings
- **prototype_mtc `settings.yaml` models list**, in order:
  1. initialize_landuse, initialize_households, compute_accessibility
  2. school_location, workplace_location
  3. auto_ownership_simulate, free_parking
  4. cdap_simulate
  5. mandatory_tour_frequency, mandatory_tour_scheduling
  6. joint_tour_frequency, joint_tour_composition, joint_tour_participation, joint_tour_destination, joint_tour_scheduling
  7. non_mandatory_tour_frequency, non_mandatory_tour_destination, non_mandatory_tour_scheduling
  8. tour_mode_choice_simulate
  9. atwork_subtour_frequency, atwork_subtour_destination, atwork_subtour_scheduling, atwork_subtour_mode_choice
  10. stop_frequency, trip_purpose, trip_destination, trip_purpose_and_destination, trip_scheduling, trip_mode_choice
  11. write_data_dictionary, track_skim_usage, write_trip_matrices, write_tables, summarize

  — [prototype_mtc settings.yaml](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/settings.yaml)
- **Other components in v1.6.0.** The dev-guide component list adds disaggregate accessibility, vehicle type choice, telecommute frequency, work from home, school escorting, transit pass ownership and subsidy, trip scheduling choice, trip departure choice and parking location choice. — [ActivitySim v1.6.0 dev-guide index](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/index.html)
- **Logsums in mandatory tour scheduling.** `mandatory_tour_scheduling.yaml` sets `LOGSUM_SETTINGS: tour_mode_choice.yaml`, and the work scheduling spec has `util_mode_choice_logsum,...,mode_choice_logsum`. — [prototype_mtc configs](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Timetable terms in the work scheduling spec:**
  - `(tour_num > 1) & (start < end_previous)`;
  - `tt.adjacent_window_before/after`;
  - `tt.previous_tour_ends`;
  - `tt.remaining_periods_available`.
  — [prototype_mtc tour_scheduling_work.csv](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **NMTF spec terms** include "Logged Maximum Residual Window, tour frequency = 0…5+" (time left after mandatory scheduling) and accessibility terms such as `nmRetail * (tot_tours == 1)` and `trOpRetail * (tot_tours == 1)`. It also distinguishes "Total Number of Tours = 0 (No Prior Tours)" from "(1 or more Prior Tours)". Zero tours is therefore an available alternative for everyone. — [prototype_mtc non_mandatory_tour_frequency.csv](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Local generation order** (`activitysim/chain.py:36-60`):
  1. CDAP (line 38);
  2. MTF (line 44);
  3. NMTF (line 49), using only labels, the CDAP=M flag and the mandatory count;
  4. placement of all tours (`place_tours`, line 60).

  The docs list the steps as "Mandatory tours. How many…, then when each one leaves and returns" followed by "Non-mandatory tours", which implies that mandatory scheduling comes before NMTF. — [local chain.py](composhed/activitysim/chain.py); [docs/activitysim.md:13-16](docs/activitysim.md)
- **Local NMTF availability.** People with an N day cannot choose zero tours (`models.py:438-440`, `docs/activitysim.md:59`). — [local models.py](composhed/activitysim/models.py)
- **Local docs' list of absent components:** "Location, mode, travel time, joint tours, at-work sub-tours | absent". — [docs/activitysim.md:65](docs/activitysim.md)
- **Non-mandatory tour order in ActivitySim.**
  - `create_tours` stacks tour counts in the column order of the alternatives file and numbers them with `tour_num` (`tour_frequency.py:59-89`).
  - Scheduling loops `for tour_num, nth_tours in tours.groupby("tour_num", sort=True)` (`vectorize_tour_scheduling.py:1083`).
  - The alternatives' column order is escort, shopping, othmaint, othdiscr, eatout, social.

  — [tour_frequency.py](https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/util/tour_frequency.py); [vectorize_tour_scheduling.py](https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/util/vectorize_tour_scheduling.py)
- **Local non-mandatory tour order.** `NM_ORDER = ["escort", "shopping", "othmaint", "social", "othdiscr"]` (`activitysim/data.py:21`), which puts social before othdiscr. — [local data.py](composhed/activitysim/data.py)
- **Mandatory tour order for students.** For work_and_school students, ActivitySim swaps `tour_num` so that school is tour 1 ("swap tour_nums for non-workers so school tour is 1 and work is 2", `tour_frequency.py:242-248`). Locally, work always comes first (`TOUR_ORDER`, `data.py:22`). — [tour_frequency.py](https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/util/tour_frequency.py)
- **The tdd table.** prototype_mtc's `tour_departure_and_duration_alternatives.csv` covers start and end hours 5..23 with start ≤ end: 189 rows plus a header, i.e. 190 lines. Locally, an early 00:00–05:00 bin is added, giving 20 periods and 210 pairs (`periods.py:30-31`), which is consistent with `docs/activitysim.md:58`. — [prototype_mtc configs](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)

### Inferences
- **NMTF cannot see the time left in the day.** Because local NMTF is sampled before mandatory scheduling, its utility cannot respond to remaining time, which is a key ActivitySim mechanism. Infeasibility is handled afterwards by dropping tours. The stability runs report `tours_dropped: 0` for seeds 0-4, so in practice this has been harmless. It is still a structural deviation and should be documented.
- **Order fidelity is overstated.** `docs/activitysim.md:60` rates NM scheduling "full" with order "escort → shopping → maintenance → social → discretionary". ActivitySim's prototype order puts othdiscr before social (with eatout in between). This is minor, but it is a factual mismatch under a "full" label.
- **The `vehicles` input stands in for an ActivitySim model output.** Using NTS `vehicles` as a covariate replaces what is an endogenous model (auto_ownership_simulate) in ActivitySim. This should be listed among the deviations.

### Gaps
- No check was made of whether prototype_arc, prototype_semcog or prototype_mwcog reorder the NMTF alternative columns. Only prototype_mtc was inspected.

---

## Q4. Estimation: ActivitySim estimation mode with larch vs local estimation

### Takeaway
ActivitySim runs survey households through the model in "estimation mode" to write estimation data bundles (EDBs), then estimates the same specs in larch (larch ≥ 6.0.46 in the current dev dependencies). By default larch uses BHHH, or SLSQP when constraints or bounds are present. The local code instead estimates its own numpy/scipy models by ridge-penalised L-BFGS directly on NTS-derived records. There are no EDBs, no spec files and no constrained or fixed coefficients. The docs acknowledge that the models are their own implementations (CLAUDE.md, `choice.py` docstring), but not that this departs from the larch/EDB workflow.

### Cited Findings
- **Estimation workflow.** "ActivitySim is then run in estimation mode to read the ActivitySim-format travel survey files, and apply the ActivitySim submodels to write estimation data bundles (EDBs) that contains the model utility specifications, coefficients, chooser data, and alternatives data for each submodel." Survey choices "must match the model design (i.e. a person cannot have more work tours than the model allows)". — [Estimation Mode index, v1.6.0](https://activitysim.github.io/activitysim/v1.6.0/users-guide/estimation-mode/index.html)
- **EDBs come from the same specs as simulation.** "There is no duplication of model specifications. ActivitySim owns the specification and larch pivots off of it." "Estimation mode runs single-processed and without destination sampling." — [Model Development docs](https://activitysim.github.io/activitysim/v1.6.0/users-guide/model_dev.html)
- **larch defaults.** "By default, when no constraints or bounds are present, Larch uses an implementation of the BHHH algorithm … When constraints or bounds are present, by default Larch uses … SLSQP." The `component_model` function loads EDBs for any standard component. — [Larch estimation docs](https://activitysim.github.io/activitysim/v1.6.0/users-guide/estimation-mode/larch.html)
- **Estimation submodules.** The `activitysim.estimation.larch` submodules include `simple_simulate`, `cdap`, `location_choice`, `mode_choice`, `nonmand_tour_freq`, `scheduling` and `stop_frequency`. — [Model Development docs](https://activitysim.github.io/activitysim/v1.6.0/users-guide/model_dev.html)
- **larch version.** `pyproject.toml` on main lists `"larch>=6.0.46"` in its development dependency list. — [ActivitySim pyproject.toml](https://github.com/ActivitySim/activitysim/blob/main/pyproject.toml)
- **Coefficient constraint flags.** Coefficient files carry a `constrain` flag, e.g. `coef_unavailable,-999,T` and `coef_ft_worker_work2_asc,-3.3781,F`. — [prototype_mtc mandatory_tour_frequency_coefficients.csv](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Local estimation.**
  - Ridge-penalised L-BFGS-B for the MNL (l2=1e-4, `choice.py:51-109`) and for ConditionalLogit (l2=1e-3, `choice.py:301, 346`).
  - Chosen alternatives are forced to be available in training (`choice.py:72-74, 324-327`).
  - The docstring justifies this by statsmodels' Hessian cost.

  — [local choice.py](composhed/common/choice.py)

### Inferences
- **No standard errors.** The local estimates are penalised MLEs without standard errors or t-statistics. ActivitySim/larch estimation produces coefficient tables with standard errors. If the paper reports coefficients, it should say that they are ridge-penalised.
- **The ConditionalLogit penalty may bias timing estimates.** A ridge of 1e-3 on all ConditionalLogit parameters, including period dummies, is fairly strong relative to per-observation log-likelihood scaled by 1/N. Calibration then compensates via the start, end and duration offsets.

### Gaps
- The exact v1.6.0 tag `pyproject.toml` was not checked. The larch version comes from `main`.

---

## Q5. Calibration: ActivitySim practice vs local calibration

### Takeaway
ActivitySim v1.6.0 has **no built-in calibration procedure**. The "Calibration" section of its docs is a "<Todo Placeholder>". In practice, agencies and consultants calibrate ASCs iteratively against targets with external scripts or notebooks; for example, MWCOG Gen3 iterated to a 4×10⁻³ share tolerance in 6-7 iterations. An "Auto-Calibrate" design effort exists, but its first phase targets auto ownership, work location and work tour mode choice.

The local loop has the same overall shape: simulate, compare shares, adjust ASCs by a damped log ratio, repeat. It differs in four ways:
- it is built in and automatic;
- its targets are the **training data itself**, not independent survey or count targets;
- it also calibrates time-of-day marginal constants;
- it keeps the best iterate.

The local docs cite "ActivitySim documentation" for "calibration by segment constants", which the documentation does not actually support.

### Cited Findings
- **The ActivitySim v1.6.0 Model Development page** reads "Calibration\n-----------\n\n<Todo Placeholder>". — [Model Development docs source](https://activitysim.github.io/activitysim/v1.6.0/_sources/users-guide/model_dev.rst.txt)
- **MWCOG Gen3 (ActivitySim-based), transit pass subsidy model:**
  - "constants … were updated in the script, model outputs were updated through partial runs, and this process was iterated until modeled shares matched the targets within a 4 × 10−3 tolerance, which was achieved after six iterations";
  - "staff later developed a Python script that automates this entire process".

  For free parking, the script "iteratively updates model coefficients and alternative-specific constants", converging "after seven iterations". The memo also tabulates calibration constants for tour mode choice, work tour mode × auto sufficiency, trip mode choice and stop arrangement. — [MWCOG Gen3 Model v1.1.0 Calibration Efforts](https://www.mwcog.org/assets/1/6/Gen3_Model_v1.1.0_Calibration_Efforts_v4.pdf)
- **The ActivitySim "Auto-Calibrate Task Scope and Design Proposal"** describes current calibration as ad hoc and partly manual, varying across agencies and consultants. Phase 2 targets auto ownership, work location choice and work tour mode choice, tested on SANDAG. This comes from the search-result summary only; the deck was not fetched. — [Google Slides deck](https://docs.google.com/presentation/d/1QdPNEtft1bormXB9P7fjor_NSm1XJCM2/htmlpresent)
- **The ARC ABM calibration report** describes calibration as adjusting alternative-specific parameters to match observed data. This also comes from the search snippet only, not a full read. — [ARC ABM calibration report](https://documents.atlantaregional.com/The-Atlanta-Region-s-Plan/RTP/abm-calibration-report.pdf)
- **Local calibration algorithm** (`common/calibration.py:398-580`):
  - step `offset_k += clip(lr·(log target_k − log sim_k), ±max_step)` with `lr=0.8`, `max_step=1.0`;
  - share tolerance `max(0.01, 2·SE)`;
  - mean tolerance 2%;
  - `max_iter=8`;
  - the best iterate is kept.

  Targets come from `trace_stats` on the training records (`activitysim/train.py:776-783`). TOD start/end/duration offsets get `lr_scale=1/3` (`train.py:711-713`). — [local calibration.py](composhed/common/calibration.py); [local train.py](composhed/activitysim/train.py)
- **Local docs claim** "all three variants calibrate the way practitioners calibrate the real frameworks" (`docs/source_aligned.md:47-49`) and "As in ActivitySim practice, the constants are adjusted per segment" (`docs/activitysim.md:74-75`). `docs/citations.md:144` attributes "calibration by segment constants" to the "ActivitySim documentation and example model". — [docs/source_aligned.md](docs/source_aligned.md); [docs/citations.md](docs/citations.md)
- **Convergence claim contradicted.** `docs/source_aligned.md:67` says "On NTS 2023 every target converges for all three variants". The ActivitySim stability runs dated 2026-10-27, which are newer than the code files dated 2026-10-26, report `converged: False` for seeds 1, 2 and 3:
  - seed 1: out of tolerance `mtf:__pooled__`, `nmtf:FTW`;
  - seed 2: `mtf:DRIVING_STUDENT`, `nmtf:CHILD`;
  - seed 3: `cdap:CHILD`, `mtf:__pooled__`, `stopfreq:shopping`, `tod_end_nonmandatory`.

  Only seeds 0 and 4 converged. — [stability_runs/activitysim/seed_*/validation.json](stability_runs/activitysim/)

### Inferences
- **The update rule matches practice; the targets do not.** A damped log(target/model) ASC update matches common practice, as the MWCOG scripts show. But practice calibrates to expanded survey or observed targets, often after the coefficients were transferred or estimated elsewhere. Local calibration to the same training sample mainly corrects for chain-composition effects and simulation error. That should be stated, along with its implication: calibrated in-sample marginals are partly "fit by construction" in the evaluation.
- **No ActivitySim citation supports the calibration method.** Calibrating time-of-day start, end and duration marginal constants is plausible practice but has no cited ActivitySim source. The citation row should instead point to an agency calibration report (e.g. MWCOG/ARC/SEMCOG) or be reworded as "common practice".

### Gaps
- No primary source was found that documents CDAP or tour-frequency ASC calibration specifically for an ActivitySim deployment. The SEMCOG calibration PDF appeared in search results but was not read.

---

## Q6. Random number handling and reproducibility

### Takeaway
ActivitySim reseeds a numpy Mersenne Twister on the fly for every row. Each row's seed is derived from the global seed, a channel hash (households, persons, tours, trips), a step offset and the row index. A household's results are therefore repeatable across sample sizes, multiprocessing set-ups and scenarios. v1.6.0 also offers optional Explicit Error Terms (EET), which draw Gumbel errors per chooser-alternative pair.

The local code uses one sequential `np.random.Generator` for the whole population. Runs with the same seed and inputs are byte-identical, but there is no per-person stream stability: changing one person's draws, or the number of resamples, shifts every later person. "Common random numbers" during calibration are only approximate.

### Cited Findings
- **ActivitySim's RNG goals:** reproducible regression tests; identical results under different multithreading configurations; "Repeatable household-level choices … with different sample sizes"; and "Repeatable household level results with different scenarios". The docs continue: "ActivitySim seeds on-the-fly and uses a stream of random numbers seeded by the household id, person id, tour id, trip id, the model step offset, and the global seed." "ActivitySim generates a separate, distinct, and stable random number stream for each tour type and tour number … This is done for trips as well, by direction." — [Core docs, Random](https://activitysim.github.io/activitysim/v1.6.0/core.html)
- **Seed formula.** `SimpleChannel` uses `(global_seed + channel_seed + step_seed + row_index) % (1 << 32)`, with `channel_seed = hash32(channel_name)` (md5). It tracks per-row `offset` so that successive draws within a step continue the row's stream (`random.py` lines 43-112, 206-250). — [random.py](https://github.com/ActivitySim/activitysim/blob/main/activitysim/core/random.py)
- **EET.** The default path draws "one uniform random number per chooser" and selects by cumulative probability. With `use_explicit_error_terms: True`, error terms are drawn "for each chooser-alternative pair", which reduces scenario-comparison noise. — [Explicit Error Terms, v1.6.0](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/explicit-error-terms.html)
- **Local RNG.**
  - `make_rng(seed)` returns a single `np.random.default_rng(seed)` (`common/cli.py:246-251`).
  - `generate_population` passes it through persons in order (`common/runner.py:33-55`).
  - Sampling is inverse-CDF with one uniform per row (`choice.py:34-39`).
  - Calibration re-creates `default_rng(sim_seed)` at the start of each pass (`runner.py:58-68, 80`).

  — [local runner.py](composhed/common/runner.py)
- **Local docs** say "The same random numbers are reused in every pass, so differences between passes come from the adjustments, not from noise" (`docs/source_aligned.md:51-53`). — [docs/source_aligned.md](docs/source_aligned.md)

### Inferences
- **The docs' common-random-numbers claim is approximate.** The stream restarts identically each pass, but the number of draws each person consumes varies. Examples: an extra tour from a changed NMTF choice, extra stops, TOD redraws (`tourgen.py:186-190`), the `IntraPeriodSampler` draws. So the streams of later persons desynchronise after the first changed choice, and the claim holds only roughly. ActivitySim's per-row channels avoid this. This plausibly contributes to the non-convergence in 3 of 5 seeds.
- **What can and cannot be claimed.** For the paper, "byte-identical reruns" is accurate. "ActivitySim-style random number handling" would not be. If scenario stability matters, per-person seeding, e.g. `np.random.default_rng([seed, pid_index, step])`, would be the closer analogue.

### Gaps
- None material.

---

## Q7. Factual errors in local docs, citation checks, and code issues

### Takeaway
The citations for the ActivitySim sources are bibliographically correct. Several fidelity labels and claims in `docs/activitysim.md` and `docs/source_aligned.md` are inaccurate or overstated:
- the trip-scheduling explanation is wrong;
- trip purpose is rated "full" but uses different conditioning;
- the NM scheduling order is wrong under a "full" label;
- the "ActivitySim documentation" calibration citation is unsupported;
- the "every target converges" statement is contradicted by the variant's own runs;
- the `--no-segmentation` description is wrong;
- the omitted-components list is incomplete.

No crash-level bugs were found. There are several small consistency issues between training and generation.

### Cited Findings
- **Citations check out.** Crossref confirms all three ActivitySim-related references in `docs/citations.md:156, 162, 163`:
  - Bradley & Vovsha (2005), *Transportation* 32(5), 545–571, doi:10.1007/s11116-005-5761-0;
  - Vovsha & Bradley (2004), TRR 1894, 46–56, doi:10.3141/1894-06;
  - Vovsha, Petersen & Donnelly (2002), TRR 1805, 68–77, doi:10.3141/1805-09.

  — [Crossref API](https://api.crossref.org/works?query.bibliographic=Hybrid+discrete+choice+departure-time+and+duration+model)
- **Wrong reason for the trip-scheduling deviation.** `docs/activitysim.md:64` says "ActivitySim's table is relative to the tour's hours; here it uses clock hours restricted to the tour." But ActivitySim's default is absolute clock departure hours:
  - `scheduling_mode: Literal["departure", "stop_duration", "relative"] = "departure"` (`trip_scheduling.py:451`);
  - prototype_mtc's table has absolute `HR5..HR23` columns, conditioned on `tour_hour` (the tour start hour for outbound trips, the end hour for inbound: `trip_scheduling.py:90-92`) and on `trip_num`;
  - "relative" is an optional mode.

  The real differences are that the local lookup is keyed on (tour class mandatory/non-mandatory, direction, stop index) and does **not** condition on tour_hour or the specific primary purpose (`activitysim/models.py:521-523`). — [trip_scheduling.py](https://github.com/ActivitySim/activitysim/blob/main/activitysim/abm/models/trip_scheduling.py); [prototype_mtc trip_scheduling_probs.csv](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs)
- **Trip purpose rated "full" despite different conditioning.**
  - Local (`docs/activitysim.md:63`): rated "full", using tour purpose × direction × stop number with back-off.
  - ActivitySim: conditions on primary_purpose × outbound × departure-hour range × person_type, and does **not** condition on stop number.

  — [prototype_mtc trip_purpose_probs.csv](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/prototype_mtc/configs); [local models.py:514](composhed/activitysim/models.py)
- **`--no-segmentation` does not pool everything.** `docs/activitysim.md:107` says `--no-segmentation` gives "One pooled model everywhere". But stop frequency is always fitted with `seg(self.n_min, True)`, i.e. segmented by tour purpose regardless of the flag (`activitysim/models.py:527`). — [local models.py](composhed/activitysim/models.py)
- **The spec and the code disagree on trip purpose.** SPECIFICATION ASIM-8 says "MNL per stop using purpose, direction, and tour purpose", but the code uses a probability lookup. The lookup is in fact closer to ActivitySim. PLAN.md Phase 4 records the lookup. — [SPECIFICATION.md:245](SPECIFICATION.md); [PLAN.md:156](PLAN.md)
- **Training and generation features differ:**
  - In training, `x_nmtf` uses the actual count of mandatory tours, which can be 3 or more for 14 persons (`models.py:482`). At generation it uses `len(mand) ≤ 2` (`chain.py:49`).
  - Training TOD designs include every observed non-mandatory tour, even beyond the 4-tour NMTF cap (`data.py:113-115`, `models.py:448-457`).
  - The nth-tour flag is per purpose (`chain.py:58`), whereas ActivitySim uses `tour_num > 1` across all tours (work spec rows).

  — [local chain.py](composhed/activitysim/chain.py)
- **Generation reports (stability runs, seeds 0-4).**
  - 0 fallbacks, 0 tours dropped, 35-42 stops dropped.
  - About 7% of persons (`merge_share` 0.069-0.071) have consecutive same-type episodes merged.

  The latter is a post-hoc repair, recorded in `generation_report.json` but not discussed in `docs/activitysim.md`. — [stability_runs/activitysim/seed_0/generation_report.json](stability_runs/activitysim/seed_0/generation_report.json)
- **The citations file flags its own status:** "Not yet checked in detail … That check has not been done." — [docs/citations.md:124-127](docs/citations.md)

### Inferences
- **Suggested doc corrections:**
  1. Trip scheduling: replace the "relative" explanation with "ActivitySim conditions on tour hour, purpose and trip number; here, on tour class, direction and stop index only".
  2. Downgrade trip purpose and non-mandatory scheduling from "full" to "simplified".
  3. Add the omitted components: accessibility, auto ownership (replaced by exogenous `vehicles`), free parking, telecommute/WFH, school escorting, logsums.
  4. Note that NMTF precedes scheduling and has no residual-window term.
  5. Fix the convergence claim.
  6. Fix the `--no-segmentation` description.
  7. Replace or qualify the "ActivitySim documentation" calibration citation.
  8. Add a citation for ActivitySim itself, with version, and for CT-RAMP / MTC Travel Model One.
- **Mandatory tour frequency is "full" only structurally.** Its "full" label (`docs/activitysim.md:57`) holds for the alternative set and the MNL form. The utility omits distance, accessibility and logsum terms and the `-999` availability rules.

### Gaps
- Whether the about-7% merge rate mostly comes from stop purpose = primary purpose (e.g. a shop stop on a shopping tour) was not checked. In ActivitySim, such a stop would still be a separate trip.
