# ActivitySim variant fidelity: joint tours, school escorting, INMTF, non-mandatory scheduling, at-work subtours, pipeline ordering

Scope: compares the local ActivitySim-aligned variant (`/home/fred/Projects/composhed/composhed/activitysim/`, plus `composhed/common/`) against ActivitySim **v1.6.0** (source at tag `v1.6.0`, `prototype_mtc` / `prototype_mtc_extended` example configs, docs at activitysim.github.io/activitysim/v1.6.0). ActivitySim source was downloaded from raw.githubusercontent.com at tag v1.6.0 and read directly. Local citations are `path:line`, relative to `/home/fred/Projects/composhed/`. Run evidence comes from `stability_runs/activitysim/seed_0/` (validation.json, generation_report.json, the saved model bundle).

Abbreviations: INMTF = individual non-mandatory tour frequency; tdd = tour departure-and-duration alternatives table; NM = non-mandatory.

---

## Q1. Does the local INMTF follow ActivitySim's structure (per-person-type models, alternatives as combinations of tour counts per purpose via interaction_simulate, extension probabilities for more than one tour)?

### Takeaway
It keeps two of the three parts: per-person-type segmentation, and alternatives defined as vectors of tour counts per purpose. It differs in the other three ways asked about. (1) It is a plain alternative-specific-constant MNL rather than an `interaction_simulate` utility built from alternative attributes. (2) It enumerates the observed combinations (40 of them, up to 4 tours in total) directly instead of ActivitySim's 96-alternative "0/1/2 escort × 0/1 other purposes" table plus a stochastic extension step. (3) It is estimated without the residual-time-window and joint-tour terms that drive ActivitySim's INMTF. The docs describe it as "simplified", which is accurate.

### Cited Findings

**ActivitySim v1.6.0 (reference)**
- The INMTF is run on persons with `cdap_activity` in M or N only — [non_mandatory_tour_frequency.py L227](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/non_mandatory_tour_frequency.py#L227).
- It is segmented by `SEGMENT_COL: ptype`, with 8 spec segments, each with its own coefficient file: PTYPE_FULL, PART, UNIVERSITY, NONWORK, RETIRED, DRIVING, SCHOOL, PRESCHOOL — [prototype_mtc non_mandatory_tour_frequency.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency.yaml); [non_mandatory_tour_frequency.py L168, L261-266](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/non_mandatory_tour_frequency.py#L261).
- The choice is made with `interaction_simulate` over an alternatives table — [non_mandatory_tour_frequency.py L325](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/non_mandatory_tour_frequency.py#L325).
- The alternatives file has 96 rows: escort ∈ {0,1,2} and each of shopping, othmaint, othdiscr, eatout, social ∈ {0,1}, so `tot_tours` runs from 0 to 7 — [non_mandatory_tour_frequency_alternatives.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency_alternatives.csv) (verified by parsing: 96 data rows; column maxima 2,1,1,1,1,1).
- The docs describe a two-stage model: "A choice is made using a random utility model between combinations of tours containing zero, one, and two or more escort tours, and between zero and one or more tours of each other purpose. Up to two additional tours of each purpose are added according to fixed extension probabilities." — [ActivitySim v1.6.0 docs, Non-Mandatory Tour Frequency](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/non_mandatory_tour_frequency.html).
- The extension step (`extend_tour_counts`) has three rules:
  - only persons with 1–4 modelled tours are eligible (`tour_counts.sum(axis=1).between(1, 4)`);
  - a purpose is extended only if its count equals that purpose's maximum in the alternatives table;
  - the number of extra tours (0, 1 or 2) is drawn from fixed probabilities —
  [non_mandatory_tour_frequency.py L98-L141](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/non_mandatory_tour_frequency.py#L98).
- The extension probabilities are keyed by `ptype, has_mandatory_tour, has_joint_tour, nonmandatory_tour_type`, with columns `0_tours, 1_tours, 2_tours` — [non_mandatory_tour_frequency_extension_probs.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency_extension_probs.csv).
- The INMTF spec conditions on what was already scheduled and on joint tours:
  - "Total Number of Tours = 0 (No Prior Tours)": `(tot_tours == 0) & (num_mand == 0) & (num_hh_joint_tours == 0)`;
  - `num_mand*(tot_tours==k)`;
  - `num_hh_joint_tours*(tot_tours==k)`;
  - per-purpose joint-tour counts;
  - "Logged Maximum Residual Window × tour frequency" terms —
  [non_mandatory_tour_frequency.csv L8, L23, L33-34](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency.csv). The model's `locals_dict` exposes `person_max_window` / `person_available_periods` from the timetable — [non_mandatory_tour_frequency.py L229-L237](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/non_mandatory_tour_frequency.py#L229).
- The zero-tour alternative stays in the choice set for N persons (there is no availability restriction). The spec simply puts utility terms on it — [non_mandatory_tour_frequency.csv L8](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency.csv).

**Local implementation**
- INMTF is fitted on M and N persons (`mn_idx`) — `composhed/activitysim/models.py:153`.
- The alternatives are the observed purpose-count vectors with support of at least `min_support=30`, with the zero vector forced in — `models.py:154-159`.
- Persons with rare vectors are mapped to the nearest kept vector by L1 distance, with ties broken by frequency — `models.py:110-115`.
- Seed-0 run: 40 alternatives; 464 persons mapped to the nearest alternative; 114 person-days capped — `stability_runs/activitysim/seed_0/validation.json` (`info.nm_alternatives`, `info.nm_persons_mapped`, `info.nm_caps`).
- Purpose mapping and order:
  - mapping: `ASIM_PURPOSE = {... shop→shopping, medical→othmaint, visit→social, other→othdiscr}` — `composhed/activitysim/data.py:21-22`;
  - order: `NM_ORDER = ["escort","shopping","othmaint","social","othdiscr"]` — `data.py:24`;
  - there is no `eatout` purpose.
- At most `MAX_NM_TOURS = 4` per day. Any excess is removed from the purpose with the largest count and logged — `data.py:27, 46-52`. The docstring says "No extension step" (`data.py:9-10`), and the docs say the same (`docs/activitysim.md:59, 94-96`).
- The fitted alternative set includes multi-tour counts that ActivitySim would only reach through extension: escort up to 4, shopping up to 2, social up to 2, othdiscr up to 4, total up to 4. For example `es0sh2…` (n=422), `…ot4` (n=34), `es4…` (n=39) — taken from the saved bundle `stability_runs/activitysim/seed_0/models/activitysim_models.pkl` (`nm_vectors`, `nm_counts`).
- Model form:
  - `SegmentedMNL` gives each person type with n ≥ `n_min=200` its own `MNL` and pools smaller types — `composhed/activitysim/segmented.py:15-36`;
  - the `MNL` has alternative-specific coefficients on every feature — `composhed/common/choice.py:48-49`;
  - fitted weight matrix shape for FTW: (44, 40), i.e. 44 coefficients × 40 alternatives per segment;
  - 8 models were fitted (7 person types + pooled), and no segment was pooled.
- Covariates are the 8 one-hot labels plus `[cdap=="M", n_mandatory/2]` — `models.py:53-54`. There is no residual-window term and no joint-tour term.
- Availability: for CDAP=N persons the zero-tour alternative is unavailable — `models.py:117-119`.
- In generation, INMTF is sampled immediately after mandatory tour frequency and before any tour is scheduled — `composhed/activitysim/chain.py:42-51`. Timing is placed only afterwards, by `place_tours` — `chain.py:60-61`.
- Person types: 7 types. `CHILD` merges ActivitySim's preschool and non-driving school child because NTS has a single `≤10` age bin. `DRIVING_STUDENT` covers the 10–21 band — `composhed/activitysim/persontype.py:1-42` (docstring table).

### Inferences
- **Matches:**
  - segmentation by person type (7 local types against ActivitySim's 8);
  - alternatives are purpose-count vectors;
  - choosers are M and N persons;
  - escort, shopping, othmaint (via medical), social (via visit) and othdiscr (via other) are present.
- **Deviation (model form):** ActivitySim builds the utility of each alternative from its attributes: per-purpose count terms × person attributes, `tot_tours` dummies, and interactions with residual window, mandatory and joint tours. The local model instead gives each of the 40 alternatives its own coefficient vector. That is far less parsimonious: about 1,760 parameters per segment. For example, UNIV has n = 343 persons (`validation.json` `info.person_types`) to estimate about 1,760 parameters, held in check only by L2 = 1e-4 (`segmented.py:16`). Overfitting or poorly identified coefficients in small segments are likely. Holdout validation would show this, but `n_holdout` was 0 in the stability runs.
- **Deviation (no extension step):** ActivitySim caps most purposes at 1 in the logit and adds extra tours by fixed probability. The local model puts the high-count combinations directly into the logit. This is documented and defensible: the cap of 4 tours covers 99.84% of NTS days (`docs/tour_rules.md:186`). It does mean combinations that ActivitySim could produce through extension (any total of 5–6 tours) cannot be generated, and combinations with fewer than 30 observations collapse onto their neighbours.
- **Deviation (conditioning order):** in ActivitySim, mandatory tours are already scheduled before INMTF runs, so INMTF can respond to the residual time window (`person_max_window`). The local chain draws INMTF before any scheduling and conditions only on `n_mand`. INMTF therefore cannot react to how long or late the sampled work tour is. This is an undocumented structural deviation.
- **Deliberate fix (zero tours unavailable for N):** in ActivitySim an N person may have zero individual NM tours because joint tours can satisfy the N pattern. With no joint tours locally, making zero unavailable for N keeps CDAP and INMTF consistent. This is a reasonable adaptation and is documented (`docs/activitysim.md:59`).
- **Purpose set:** ActivitySim has 6 NM purposes (escort, shopping, othmaint, othdiscr, eatout, social); the local model has 5. Eat-out has no NTS category of its own and presumably sits inside `other`, which maps to othdiscr. The omission of eatout is not stated explicitly in the docs; the docs list only the mapping. Mapping medical → othmaint is a narrowing, because othmaint in MTC also covers banking, personal business and similar.
- **Cosmetic issue:** `nm_label` uses two-letter prefixes (`data.py:55-56`), so othmaint and othdiscr are both rendered as "ot" (labels such as `es0sh0ot0so0ot1`). Labels stay unique because they are positional, so this is not a functional bug, but reports are ambiguous.

### Gaps
- Whether NTS 2023 "other" contains identifiable eat-out activity was not checked; the trip-purpose codes were not inspected.
- No holdout INMTF fit statistics were available (`n_holdout: 0`), so overfitting of the 40-alternative MNL in small segments could not be quantified.

---

## Q2. How are joint tours and school escorting handled or omitted locally, and is the omission documented?

### Takeaway
Both are absent. Joint tours (frequency, composition, participation, scheduling) are omitted and the omission is documented in several places. School escorting is omitted with **no mention anywhere** in the local docs. NTS escort-education activity is instead absorbed into the individual `escort` purpose. That matches `prototype_mtc`, which has no school_escorting step, but not `prototype_mtc_extended`, which has one.

### Cited Findings

**ActivitySim v1.6.0 (reference)**
- `prototype_mtc` runs `joint_tour_frequency → joint_tour_composition → joint_tour_participation → joint_tour_destination → joint_tour_scheduling` after mandatory tour scheduling and before `non_mandatory_tour_frequency`. `school_escorting` is **not** in its model list — [prototype_mtc settings.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/settings.yaml).
- `prototype_mtc_extended` inserts `school_escorting` directly after `mandatory_tour_scheduling` and before `joint_tour_frequency` — [prototype_mtc_extended settings.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc_extended/configs/settings.yaml).
- Joint tour frequency is a household-level `simple_simulate`. It is applied only to households flagged `participates_in_jtf_model`, i.e. `(num_travel_active > 1) & (num_travel_active_non_preschoolers > 0)` — [joint_tour_frequency.py L64, L106](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/joint_tour_frequency.py#L64); [joint_tour_participation.py L94 comment](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/joint_tour_participation.py#L94).
- The joint frequency alternatives are 21 combinations of 0–2 joint tours over shopping, othmaint, eatout, social and othdiscr (`0_tours`, `1_Shop` … `2_DD`). Escort is not a joint purpose — [joint_tour_frequency_alternatives.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/joint_tour_frequency_alternatives.csv).
- Composition is a choice among adults, children and mixed. Adult-only parties are unavailable if fewer than two adults have an M or N pattern — [joint_tour_composition.csv L20-21](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/joint_tour_composition.csv).
- In participation, "each eligible person sequentially makes a choice to participate or not participate in each joint tour … the person participation model cycles through the household members multiple times until the required types of people have joined" — [ActivitySim v1.6.0 docs, Joint Tour Participation](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/joint_tour_participation.html). The satisfaction check is in [joint_tour_participation.py L81-L126](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/joint_tour_participation.py#L81), and the `max_participation_choice_iterations` loop in [L183-L296](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/joint_tour_participation.py#L183).
- Joint tour scheduling uses person time windows. Because each joint tour may have different participants, it builds timetables per joint tour from the free time of all participants — [joint_tour_scheduling.py L68-L71, L103](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/joint_tour_scheduling.py#L68); [docs](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/joint_tour_scheduling.html).
- INMTF utilities depend on joint tours (`num_hh_joint_tours`, joint counts per purpose), and the extension probabilities are keyed on `has_joint_tour` (see Q1). Non-mandatory scheduling has a "Number of joint tours – departure shift effects" term — [tour_scheduling_nonmandatory.csv L22](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_scheduling_nonmandatory.csv).
- School escorting:
  - "determines whether children are dropped-off at or picked-up from school, simultaneously with the chaperone responsible for chauffeuring the children, which children are bundled together on half-tours, and the type of tour (pure escort versus rideshare)";
  - it runs "after work and school tours have been generated and scheduled";
  - it is a household-level choice over bundle alternatives —
  [ActivitySim v1.6.0 docs, School Escorting](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/school_escorting.html).
  - It runs in three stages: outbound, inbound, and outbound conditional on inbound. It adds pure escort tours and updates the timetable "to avoid joint tours scheduled over school escort tours" — [school_escorting.py docstring](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/school_escorting.py).
  - Defaults are `NUM_ESCORTEES = 3` and `NUM_CHAPERONES = 2` — [school_escorting.py L30-31](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/school_escorting.py#L30).

**Local implementation**
- Joint tours are documented as omitted:
  - "One person-day: no household CDAP interactions or joint tours" — `docs/activitysim.md:93`;
  - component table row "joint tours … absent" — `docs/activitysim.md:65`;
  - "model one person's day, with no household interactions or joint trips" — `docs/source_aligned.md:12`;
  - SPECIFICATION §7.3 marks household CDAP and joint tours as "optional/future" because "`hid` [is] currently dropped by `load_schedules`" — `SPECIFICATION.md:253-255`, plus `SPECIFICATION.md:32, 295`;
  - PLAN decision "Household extension | Out of v1" — `PLAN.md:58`.
- `load_schedules` drops `hid` — `composhed/data.py:23-25`. The ActivitySim records docstring says "household interactions out of scope, §7.3" — `composhed/activitysim/data.py:1`.
- A grep of the repo's `.md` and `.py` files for "school escort", "school_escort" and "eatout" returned no hits. **School escorting is not mentioned anywhere locally.**
- NTS `escort` activities (including any escort-to-education) are treated as individual `escort` NM tours or stops — `data.py:21` (`"escort": "escort"`).

### Inferences
- Omitting joint tours is justified by data scope: the person-day interface, with `hid` dropped. The project lists this as optional future work, so it is a documented simplification, not an error. NTS does carry household IDs (the input has an `hid` column that the loader drops), so a household-level extension is a matter of scope rather than strictly impossible. Whether NTS identifies *shared* or joint trips well enough to estimate composition and participation is untested.
- Consequences:
  - Joint discretionary activity that NTS respondents report (for example family shopping) is modelled as independent individual tours. This drops cross-person synchronisation, which matters little for person-level schedule metrics.
  - The INMTF and scheduling terms keyed on joint tours are simply absent.
  - ActivitySim's sequencing that reserves joint-tour windows before individual NM tours does not exist locally.
- School escorting: treating escort tours as individual INMTF escort tours is faithful to `prototype_mtc`, which has no school escort model; there, escort tours come from INMTF with escort allowed up to 2 plus extension. It is not faithful to ActivitySim configurations that enable `school_escorting` (`prototype_mtc_extended`, and several agency models). This should be stated in `docs/activitysim.md`, because escort is a large NTS purpose: 2,757 + 1,150 + 156 + … persons with 1–4 escort tours in the fitted alternatives.

### Gaps
- No check was made on whether NTS 2023 escort trips can be split into school-escort and other escort; the trip purpose codes were not inspected.
- It is not established whether ActivitySim v1.6.0 docs list `school_escorting` as part of the "canonical" pipeline. Its presence varies by example model, and only the two MTC examples were checked.

---

## Q3. Non-mandatory tour scheduling: ordering (mandatory, then joint, then non-mandatory, with time-window availability) and alternatives; what is the local equivalent?

### Takeaway
The local model reproduces the core mechanism: a logit over start/end period pairs with a timetable that removes alternatives overlapping previously scheduled tours, mandatory tours scheduled first, and NM tours scheduled one at a time in a fixed purpose order. It deviates in five ways:
- a different purpose order (social before othdiscr);
- no joint tours to block time;
- no look-ahead, adjacent-window or previous-tour utility terms;
- generic smooth shape terms × purpose rather than ActivitySim's piecewise departure, arrival and duration constants and shifts;
- a minute-level timetable with a 10-minute home buffer, rather than ActivitySim's period-level footprints.

The docs rate this component "full" fidelity (`docs/activitysim.md:60`). That overstates it; "simplified" would be accurate.

### Cited Findings

**ActivitySim v1.6.0 (reference)**
- `non_mandatory_tour_scheduling` selects tours with `tour_category == "non_mandatory"` and calls `run_tour_scheduling` with no tour segment column — [non_mandatory_scheduling.py](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/non_mandatory_scheduling.py).
- Spec and coefficient files are `tour_scheduling_nonmandatory.csv` and `tour_scheduling_nonmandatory_coefficients.csv`, MNL — [non_mandatory_tour_scheduling.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_scheduling.yaml).
- `vectorize_tour_scheduling` iterates `for tour_num, nth_tours in tours.groupby("tour_num", sort=True)`: all persons' first tours, then all second tours, and so on. After each choice it updates `previous_tour` and calls `timetable.assign(...)` — [util/vectorize_tour_scheduling.py L1083, L899-L905](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/vectorize_tour_scheduling.py#L1083).
- The interaction dataset removes alternatives that are not available via `timetable.tour_available(window_row_ids, alts_ids)` — [L632](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/vectorize_tour_scheduling.py#L632).
- `tour_num` order comes from `tour_counts.stack()`, i.e. the column order of the alternatives file — [util/tour_frequency.py L60, L88](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/tour_frequency.py#L60). In `prototype_mtc` that order is `escort, shopping, othmaint, othdiscr, eatout, social` — [non_mandatory_tour_frequency_alternatives.csv header](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency_alternatives.csv).
- The tdd alternatives are 190 hourly (start, end) pairs for hours 5–23 with start ≤ end (rows `5,5` … `23,23`) — [tour_departure_and_duration_alternatives.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_departure_and_duration_alternatives.csv).
- The NM scheduling spec (100 rows) includes ([tour_scheduling_nonmandatory.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_scheduling_nonmandatory.csv)):
  - piecewise departure, arrival and duration constants;
  - separate escort-tour departure, arrival and duration constants;
  - purpose shift effects (shopping, maintenance, eat out, visit, discretionary);
  - "Subsequent tour must start after previous tour for this purpose ends" (L7);
  - "Some previously-scheduled tour ends in this departure hour" (L40);
  - "Adjacent window exists before/after …" × first or second+ tour (L42-45);
  - "Remaining individual non-mandatory tours to be scheduled / number of unscheduled hours" (L46);
  - "Number of mandatory / joint / escort tours – departure shift effects" (L21-23);
  - "First of 2+ tours for same purpose" and "subsequent of 2+ tours for same purpose";
  - person-type shifts (school child under 16 / 16+, university student);
  - a mode choice logsum.
- Pipeline order: `mandatory_tour_scheduling` → (`school_escorting`) → `joint_tour_*` → `joint_tour_scheduling` → `non_mandatory_tour_frequency` → `non_mandatory_tour_destination` → `non_mandatory_tour_scheduling` — [prototype_mtc settings.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/settings.yaml); [prototype_mtc_extended settings.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc_extended/configs/settings.yaml).

**Local implementation**
- One `TourTOD` per group (`work`, `school`, `nonmandatory`). It is a `ConditionalLogit` over all (start period, end period) pairs with end ≥ start — `composhed/common/tod.py:1-14, 32-54`.
- The grid is `PeriodGrid.with_early_bin(60)`: a 00:00–05:00 early period plus hourly periods to 24:00, giving 20 periods and 210 pairs — `composhed/common/periods.py:310-311`; `models.py:170-171`; `docs/activitysim.md:58`. (`PLAN.md:52` says "≈230 alternatives", which is inconsistent with the implemented 210.)
- Utility terms:
  - sparse start-period, end-period and duration-bin constants (`DUR_BIN_EDGES` 0, 60, 120, 180, 240, 360, 480, 600, 720, 1440) — `tod.py:29, 45-49`;
  - dense shape terms [start, end, dur, dur²] interacted with z — `tod.py:50`;
  - z = [1, labels, person-type one-hot, NM purpose one-hot (escort baseline), nth-tour-of-same-purpose>0] — `models.py:48-50`.
- Timetable: a minute-level `TimeWindow`. Usable windows shrink by `buffer=10` minutes next to reserved tours and by 1 minute at the day edges — `composhed/common/timewindow.py:229-241`. The period-pair mask is `period_availability_mask` (`timewindow.py:263-280`, `tod.py:66-69`).
- Ordering:
  - mandatory tours first, then NM tours sorted by `NM_ORDER` = escort → shopping → othmaint → social → othdiscr — `chain.py:52` (`purposes = mand + sorted(nm, key=NM_ORDER.index)`); `data.py:24-25`;
  - each tour is placed sequentially with the timetable updated — `composhed/common/tourgen.py:54-78`;
  - placement is per person (all of one person's tours), not ActivitySim's cross-person `tour_num` waves. This is equivalent for a person-day model.
- Estimation uses the same order. Each observed tour's availability mask is computed from the timetable of the person's earlier tours in `TOUR_ORDER`, then start time — `models.py:122-137`; `data.py:104`. Seed-0 run: 55,867 NM tours; 3 chosen alternatives had to be forced available — `validation.json` "ASIM-4/6 tdd nonmandatory" `forced_available: 3`.
- Unplaceable tours are dropped and counted — `tourgen.py:72-74`. The seed-0 generation report shows `tours_dropped: 0`.
- Ablations: `--consistency rescale` (no timetable) and `--independent-timing` (KDE plus log-normal) — `docs/activitysim.md:103-111`.

### Inferences
- **Matches:**
  - joint (start, end) period choice;
  - availability from a timetable built from previously scheduled tours;
  - mandatory scheduled before NM;
  - fixed purpose order within NM;
  - a "second+ tour of same purpose" indicator, analogous to ActivitySim's "subsequent of 2+ tours for same purpose";
  - person-type terms;
  - estimation mode that builds the timetable from earlier observed tours.
- **Deviation (order):** the local order puts `social` before `othdiscr`. `prototype_mtc` order is escort, shopping, othmaint, othdiscr, eatout, social, so social is scheduled last. `docs/activitysim.md:60` states the local order and calls it "full" fidelity. A one-line reorder of `NM_ORDER` (`data.py:24`) would align it; `nm_label` and the alternatives would also re-order. With the timetable this order affects which purpose gets first choice of the free time.
- **Deviation (no joint layer):** in ActivitySim, joint tours (and school-escort tours) occupy windows before individual NM tours are scheduled. Locally only mandatory tours do. This follows from Q2.
- **Deviation (utility specification):** ActivitySim's look-ahead term ("remaining tours to be scheduled / unscheduled hours") and adjacent-window terms make early tours leave room for later ones. Locally there is no look-ahead. In this data infeasibility is rare (0 tours dropped; 0 resamples in seed 0), but early-ordered purposes (escort, shopping) may take time windows that later purposes would have used, without any penalty.
- **Deviation (functional form):** ActivitySim also has escort-specific period constants and purpose shift terms. Locally, purpose enters only through four smooth interactions (start, end, duration, duration²). Period, duration and constant shapes are shared across NM purposes. This is a simplification, and the docs do not list it under the scheduling row.
- **Deviation (timetable granularity):** ActivitySim footprints are hourly, and a new tour may begin in the hour a previous tour ended (the spec has a term for it). Locally the timetable is in minutes with a 10-minute home buffer. This is finer than ActivitySim and justified by the absence of travel times (documented in `docs/tour_rules.md:190-200`).
- **Not modelled:** the "Subsequent tour must start after previous tour for this purpose ends" constraint has no local analogue. Two same-purpose tours can therefore be placed in either clock order. Since the outputs carry no tour index this is harmless.
- Mandatory `work_and_school` ordering: ActivitySim swaps to school-first for non-workers ([util/tour_frequency.py L242-L248](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/tour_frequency.py#L242)). Locally the order is always work then school (`data.py:42-43`, `data.py:104`). This is minor, since `work_and_school` is a small class, but it is a deviation for UNIV and DRIVING_STUDENT.

### Gaps
- The effect of the purpose order on fit was not quantified. A run with `NM_ORDER` matching MTC would show it.
- `tour_scheduling_nonmandatory_coefficients.csv` was not inspected to see which ActivitySim terms carry most weight.

---

## Q4. At-work subtours: present or collapsed?

### Takeaway
Absent. Activities inside a work tour that ActivitySim would generate as at-work subtours (eat, business, maint) are collapsed into the work tour as **inbound stops after a merged work block**. This is documented in `docs/activitysim.md:100` and `docs/tour_rules.md:148-162`, but it moves those activities in time, and it creates a small inconsistency in the training data for trip departure times.

### Cited Findings

**ActivitySim v1.6.0 (reference)**
- `atwork_subtour_frequency` runs on work tours (`tours[tours.tour_type == "work"]`) with alternatives `no_subtours`, `eat`, `business1`, `maint`, `business2`, `eat_business` — [atwork_subtour_frequency.py L60](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/atwork_subtour_frequency.py#L60); [atwork_subtour_frequency_alternatives.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/atwork_subtour_frequency_alternatives.csv).
- The docs say subtours are "taken during the workday with their origin at the work location, rather than from home" — [ActivitySim v1.6.0 docs, At-work Subtours Frequency](https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/atwork_subtour_frequency.html).
- It runs after `tour_mode_choice_simulate`, i.e. after all home-based tours are scheduled, and is followed by `atwork_subtour_destination`, `atwork_subtour_scheduling` and `atwork_subtour_mode_choice` — [prototype_mtc settings.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/settings.yaml).
- In subtour scheduling, the "Timetable has to be initialized to set all timeperiods outside parent tour footprint as unavailable. So atwork subtour timewindows are limited to the footprint of the parent work tour" — [util/vectorize_tour_scheduling.py L1178-L1204](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/vectorize_tour_scheduling.py#L1178).

**Local implementation**
- Fragmented mandatory activity: "all mandatory episodes in a tour are merged into one block that starts at the first mandatory start, lasts the *sum* of the mandatory episode durations … all later ones (including those inside the block) are inbound stops" — `composhed/common/tours.py:10-15`; implemented at `tours.py:128-150`.
- Docs: "everything after the block's start is an inbound stop. That includes activities squeezed between work episodes … so there are no tours from work" — `docs/tour_rules.md:157-160`. "No at-work sub-tours." — `docs/activitysim.md:100`. The "absent" row is at `docs/activitysim.md:65`.
- Scale: 275 fragmented tours (275 persons) and 11 mixed work/education tours in NTS 2023 — `docs/tour_rules.md:162`; `validation.json` `info.extraction.fragmented_tours: 275`.
- Inbound stops keep their **observed** start time — `tours.py:139-140`. `primary_end` is recomputed as `t.end - sum(inbound durations)` — `composhed/activitysim/data.py:91`.
- The trip-departure training data then uses `deps = [t.primary_end] + [s + dur for stops[:-1]]` — `models.py:199`.

### Inferences
- The collapse is a documented simplification, justified by the person-day scope (it inherits the compositional baseline rule and ActVAE scope; see the CLAUDE.md note on work-based subtours). ActivitySim would produce home→work→[work→eat→work]→home with the eat inside the work footprint. The local variant generates home→work(merged duration)→eat→home, which moves midday at-work activities to after work.
- **Training-data inconsistency in the local code (minor, 275 tours):**
  - for a fragmented work tour, an "inbound" stop that actually occurred between work episodes has `start + duration` earlier than the synthetic `primary_end`;
  - so in the ASIM-9 trip departure lookup the departure from the primary activity is later than the observed departure from the first inbound stop: the departures are non-monotone;
  - the departure periods recorded for these stops also reflect midday clock times while they are generated after the work block.
  - The volume is tiny (275 of about 19k mandatory tours), so the practical impact is negligible, but it is formally inconsistent and undocumented.
- Without subtours, the local stop-frequency model for work tours absorbs these activities as extra inbound stops. That slightly inflates work-tour inbound stop shares relative to an ActivitySim-structured extraction.

### Gaps
- It was not checked how NTS codes in-course-of-work ("business") trips. NTS has an "in course of work" purpose that may be mapped to `work` or `other` upstream in foundata, which would affect how many at-work subtours are hidden.

---

## Q5. General tour generation ordering in the ActivitySim pipeline compared with the local chain

### Takeaway
ActivitySim interleaves frequency and scheduling by tour category: mandatory frequency, then mandatory scheduling, then [school escorting], then joint frequency, composition, participation and scheduling, then individual NM frequency, then NM scheduling, then at-work subtours, then stops and trips. The local chain draws **all frequencies first** (CDAP, MTF, INMTF) and only then schedules all tours (mandatory first, then NM in purpose order). The scheduling order matches; the frequency/scheduling interleaving does not. That breaks the dependence of INMTF on the residual time window, and it is not documented as a departure.

### Cited Findings
- prototype_mtc order: `cdap_simulate, mandatory_tour_frequency, mandatory_tour_scheduling, joint_tour_frequency, joint_tour_composition, joint_tour_participation, joint_tour_destination, joint_tour_scheduling, non_mandatory_tour_frequency, non_mandatory_tour_destination, non_mandatory_tour_scheduling, tour_mode_choice_simulate, atwork_subtour_frequency, atwork_subtour_destination, atwork_subtour_scheduling, atwork_subtour_mode_choice, stop_frequency, trip_purpose, trip_destination, trip_purpose_and_destination, trip_scheduling, trip_mode_choice …` — [prototype_mtc settings.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/settings.yaml).
- prototype_mtc_extended adds `school_escorting` after `mandatory_tour_scheduling`, and `vehicle_type_choice` / `vehicle_allocation` — [prototype_mtc_extended settings.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc_extended/configs/settings.yaml).
- Tour categories are typed `["mandatory", "joint", "non_mandatory", "atwork"]` — [util/tour_frequency.py L95-L100](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/tour_frequency.py#L95).
- Local chain (`composhed/activitysim/chain.py`, `AsimChain.generate`), in order:
  1. person type → CDAP;
  2. H returns a home day;
  3. MTF if M;
  4. INMTF (conditioned on `len(mand)` only);
  5. build the list of planned tours in the order mandatory, then `NM_ORDER`;
  6. `place_tours` (timetable placement);
  7. per tour, stop frequency (`_stops`), trip purpose lookup and trip departure lookup;
  8. `tours_to_rows`.
- Module docstring table, ASIM-1 … ASIM-9 — `composhed/activitysim/models.py:1-14`; `docs/activitysim.md:10-21`.
- Local mandatory tour frequency alternatives are `work1, work2, school1, school2, work_and_school`, fitted over the observed options — `data.py:26`; `models.py:146-150`.

### Inferences
- **Matches:**
  - CDAP before frequency;
  - mandatory before non-mandatory;
  - timetable-constrained sequential scheduling;
  - stop frequency, trip purpose and trip scheduling after tours are placed;
  - person-type segmentation throughout.
- **Structural deviation (undocumented):** INMTF is drawn before mandatory scheduling. ActivitySim's INMTF sees the scheduled mandatory footprint (Logged Maximum Residual Window). Locally a person can draw 4 NM tours and then a 10-hour work tour, with no feedback. The timetable then has to fit them. In this data `tours_dropped` = 0 and `stops_dropped` = 38 (seed 0, `generation_report.json`), so the deviation hurts conditional realism more than it causes feasibility failures. A faithful fix is to schedule mandatory tours first and add a `log(max free window)` covariate to `x_nmtf` (`models.py:53-54`), computed from the observed mandatory tours at estimation time. That mirrors ActivitySim's `person_max_window`.
- **Absent stages (all documented except school escorting):** joint tours, school escorting (undocumented), at-work subtours, and destination, mode and logsum stages. The absence of locations removes the mode choice logsum and accessibility terms that appear in the ActivitySim INMTF and scheduling specs. That is justified by the data (no zones or skims).
- **Bookkeeping observation (outside the requested scope):** `generation_report.json` (seed 0) shows `merges: 4797` across 3,986 persons (7.0%). These are merges of consecutive identical activities in `from_episodes`, for example an inbound stop whose purpose equals the tour primary activity. This is a post-hoc repair with no ActivitySim counterpart; it is worth checking in the stop and trip-purpose notes.

### Gaps
- The question of whether the INMTF-before-scheduling order materially changes acteval metrics was not tested; it would need an experiment.
- Only the `prototype_mtc` and `prototype_mtc_extended` pipelines were checked; agency implementations (SANDAG, ARC, PSRC and others) vary in where `school_escorting` and the joint models sit.

---

## Q6. Summary list: matches, deviations, data-limited simplifications, and errors or bugs

### Takeaway
The variant is a credible person-day reduction of the ActivitySim chain. Its fidelity claims are mostly honest. Three should be corrected:
- non-mandatory scheduling is labelled "full";
- school escorting is not mentioned;
- the order of frequency and scheduling differs from ActivitySim without being listed as a departure.

There are no serious correctness bugs. There is one minor training-data inconsistency (fragmented work tours) and one cosmetic labelling issue.

### Cited Findings
(All items are drawn from the cited findings above. Sources are repeated briefly.)
- **Matches:**
  - INMTF segmented by person type, with alternatives as purpose-count vectors, choosers M and N (`models.py:152-166`; [NMTF yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency.yaml));
  - joint (start, end) choice with timetable availability; mandatory then NM; fixed NM purpose order (`tourgen.py:41-86`; [vectorize_tour_scheduling.py L1083](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/vectorize_tour_scheduling.py#L1083));
  - estimation-mode timetable built from earlier observed tours (`models.py:122-137`).
- **Deviations (design choices):**
  - alternative-specific MNL rather than interaction_simulate (`choice.py:48`);
  - no extension step; observed combinations up to 4 tours instead (`data.py:9-10, 46-52`; [extend_tour_counts](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/non_mandatory_tour_frequency.py#L98));
  - zero NM tours unavailable for N persons (`models.py:117-119`);
  - NM order social before othdiscr ([MTC alternatives header](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/non_mandatory_tour_frequency_alternatives.csv) against `data.py:24`);
  - work-then-school always ([tour_frequency.py L242-248](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/tour_frequency.py#L242) against `data.py:42-43`);
  - INMTF drawn before mandatory scheduling ([settings.yaml order](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/settings.yaml) against `chain.py`);
  - scheduling utility without look-ahead, adjacent-window or previous-tour terms ([tour_scheduling_nonmandatory.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_scheduling_nonmandatory.csv) against `models.py:48-50`, `tod.py:45-50`);
  - 210-pair grid with an early bin, against 190 hourly 5–23 pairs ([tdd alts](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_departure_and_duration_alternatives.csv) against `periods.py:310`).
- **Data-limited simplifications (documented):**
  - no joint tours, because `hid` is dropped (`SPECIFICATION.md:253-255`; `docs/activitysim.md:93`);
  - no at-work subtours (`docs/activitysim.md:100`);
  - 7 person types from binned age (`persontype.py`);
  - purpose mapping medical→othmaint, visit→social, other→othdiscr (`docs/activitysim.md:59`);
  - no locations, modes or logsums (`docs/activitysim.md:89-90`);
  - 10-minute home buffer (`docs/tour_rules.md:190-200`).
- **Undocumented omissions and overclaims:**
  - school escorting is not mentioned (repo grep);
  - eatout purpose not mentioned;
  - "full" fidelity claimed for NM scheduling (`docs/activitysim.md:60`);
  - `PLAN.md:52` says "≈230" tdd alternatives while the implementation has 210 (`docs/activitysim.md:58`).
- **Errors and bugs (minor):**
  - fragmented-work-tour inbound stops keep observed midday times while `primary_end` is synthetic, giving non-monotone departures in the ASIM-9 training data (`tours.py:139-140`, `data.py:91`, `models.py:199`), 275 tours;
  - `nm_label` prefixes collide for othmaint and othdiscr ("ot") (`data.py:55-56`), cosmetic only;
  - INMTF has about 1,760 parameters per segment, with a UNIV segment of n = 343 regularised only by L2 = 1e-4 (`segmented.py:16`; bundle weights (44, 40); `validation.json` person_types). This is an overfitting risk, not a bug.

### Inferences
- Suggested documentation fixes:
  1. Change the NM scheduling fidelity to "simplified" and list the missing look-ahead, adjacent-window and purpose-specific period constants.
  2. Add "school escorting: absent (escort-to-school is an individual escort tour, as in prototype_mtc without school_escorting)".
  3. Add "eatout has no NTS counterpart (inside othdiscr)".
  4. Add "INMTF is drawn before mandatory scheduling; ActivitySim conditions INMTF on the residual window".
  5. Reorder `NM_ORDER` to MTC's escort, shopping, othmaint, othdiscr, (eatout), social, or document the difference.
- Suggested code improvements, in order of fidelity gain:
  - (a) schedule mandatory tours before INMTF and add a log-max-residual-window covariate;
  - (b) give the INMTF an interaction_simulate-style parameterisation: per-purpose count terms × covariates plus `tot_tours` dummies, to cut parameters by an order of magnitude;
  - (c) add a "remaining NM tours / free hours" covariate to z_tod.

### Gaps
- None of the deviations' quantitative impact on acteval metrics was measured in this research.
