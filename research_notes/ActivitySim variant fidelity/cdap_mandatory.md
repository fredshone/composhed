# ActivitySim variant fidelity: person types, CDAP, mandatory tour frequency, mandatory tour scheduling

Scope: the local `composhed/activitysim/` variant (with `composhed/common/`) compared against ActivitySim **v1.6.0**: the `ActivitySim/activitysim` repo at tag `v1.6.0` (raw files downloaded and read directly), its `prototype_mtc` configs (plus `prototype_arc` / `prototype_mwcog` for how person types are derived), and the v1.6.0 component docs. Local paths are relative to `/home/fred/Projects/composhed`. Local empirical figures come from `stability_runs/activitysim/seed_0/validation.json`: a training run on all 56,734 NTS person-days.

Short URL keys used below:
- `[MTC-cfg]` = https://github.com/ActivitySim/activitysim/tree/v1.6.0/activitysim/examples/prototype_mtc/configs
- `[abm]` = https://github.com/ActivitySim/activitysim/tree/v1.6.0/activitysim/abm/models

---

## Q1. Person types: does the local code reproduce ActivitySim's 8 person types, and how are they derived from NTS?

### Takeaway
The local code has **7** person types, not 8. It merges pre-school children (ptype 8) and pre-driving school children (ptype 7, ages 6–10) into `CHILD`. It also defines `RETIRED` and `NONWORK` from NTS self-reported employment instead of ActivitySim's age-65 split, and it cannot separate the 16–19 "driving-age student" range from 10–15 or 18–21 university students, because NTS age is binned as `≤10 / 10-21 / 21-32 …`. These are data-forced approximations and the module docstring documents them. They do, however, cause downstream conflicts with ActivitySim's ptype-based availability rules (see Q2 and Q3).

### Cited Findings
- ActivitySim uses 8 integer person types: FULL=1, PART=2, UNIVERSITY=3, NONWORK=4, RETIRED=5, DRIVING=6, SCHOOL=7, PRESCHOOL=8. Source: `constants.yaml` (`PTYPE_FULL … PTYPE_PRESCHOOL`) and `cdap.yaml` `CONSTANTS`. — [prototype_mtc constants.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/constants.yaml); [cdap.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/cdap.yaml)
- In prototype_mtc, `ptype` comes with the input synthetic population. `annotate_persons.csv` only uses it, for example `has_non_worker`, `student_is_employed` and `nonstudent_to_school`; it does not derive it. — [prototype_mtc annotate_persons.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/annotate_persons.csv)
- The canonical definitions, as derived in prototype_arc:
  1. FT worker, age 16+
  2. PT worker, non-student, age 16+
  3. University student
  4. Non-worker non-student, age 16–64
  5. Non-worker non-student, age 65+ ("Retired")
  6. Student age 16–19, not a FT worker or university student
  7. Pre-driving school student, age 6–15
  8. Child under age 6

  — [prototype_arc annotate_persons.csv, lines 58–65](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_arc/configs/annotate_persons.csv)
- prototype_mwcog derives the same 8 types from `pemploy` × `pstudent` × age. For example, RETIRED = non-student, not employed, age ≥ 65; DRIVING = grade/high-school student aged ≥ 16; SCHOOL = grade-school student aged 6–15; PRESCHOOL = age < 6. — [prototype_mwcog annotate_persons.csv, lines 31–40](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mwcog/configs/annotate_persons.csv)
- CDAP's `PERSON_TYPE_MAP` groups WORKER = {1, 2} and CHILD = {6, 7, 8}. This drives household-member ranking: up to 2 workers, then up to 3 youngest children. — [cdap.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/cdap.yaml); [CDAP docs v1.6.0](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/cdap.md)
- **Local rules** (`composhed/activitysim/persontype.py:23-42`) apply in order, and the first match wins:
  1. `age == "≤10"` → CHILD
  2. ft-employed → FTW
  3. pt-employed → PTW
  4. retired → RETIRED
  5. student and age ≥ 21 → UNIV
  6. student or unknown, age 10–21 → DRIVING_STUDENT
  7. unemployed, other or unknown, age ≥ 21 → NONWORK
  8. Anything else → NONWORK, logged as "unmapped"

  `PERSON_TYPES` has 7 entries (`persontype.py:19`).
- Local person-type counts (seed 0, 56,734 persons): FTW 21,878; RETIRED 14,997; PTW 6,685; CHILD 5,770; DRIVING_STUDENT 4,375; NONWORK 2,686; UNIV 343. 142 records were unmapped: `10-21|unemployed` 88 and `10-21|other` 54. — local `stability_runs/activitysim/seed_0/validation.json` (`info.person_types`, `info.unmapped`)
- The local docs rate the person-type component as "simplified": "NTS's age bands can't separate pre-school, non-driving and driving-age children." — local `docs/activitysim.md:55`
- SPECIFICATION ASIM-1 asks for the 8 types, including separate "non-driving student" and "pre-school" types. — local `SPECIFICATION.md:231`

### Inferences
- **Matches:**
  - FTW and PTW come from employment status, as in ActivitySim. Employment is checked before student status, so a FT-employed 16–21 year old becomes FTW. This matches ARC/MWCOG, where FT employment wins.
  - UNIV comes from student status.
- **Deviations (data-forced):**
  - CHILD = ActivitySim 8 plus part of 7 (ages 6–10). DRIVING_STUDENT = the rest of 7 (ages 11–15) plus 6 (16–19) plus 18–20-year-old university students. The `≤10 / 10-21` bin edges make this unavoidable.
  - The UNIV segment is tiny (343) because most university students fall into the 10–21 bin and are typed DRIVING_STUDENT. This is a systematic mis-typing, not just a coarsening.
- **Deviations (definition choice):**
  - RETIRED is NTS self-reported retired at any age. In ActivitySim it is any non-working non-student aged 65+.
  - NONWORK includes adults with `unknown` employment (NTS employment has about 16% nulls) and unemployed or "other" people aged 65+. ActivitySim would type the latter as RETIRED.
  - PT-employed students become PTW. ActivitySim/ARC would type a PT-working student as UNIV or DRIVING, with `student_is_employed`. NTS gives a single employment category, so this cannot be separated.
- **Unmapped fallback:** `unemployed/other` aged 10–21 become NONWORK. In ActivitySim, 16–19 non-students would be NONWORK; under-16s would be children. With only 142 cases the effect is negligible, and it is logged.
- **Redundancy (not a bug):** ptype is a deterministic function of the `age` and `employment` labels, which are also covariates. Within each segment some label dummies are constant (for example all age dummies in CHILD), and in scheduling the ptype dummies are near-collinear with the label dummies. Identification relies on the L2 penalty in `MNL` / `ConditionalLogit` (`common/choice.py:51`, `:301`).

### Gaps
- I found no ActivitySim v1.6.0 doc page with a canonical, normative ptype definition table. Definitions are implementation-specific and live in each example's `annotate_persons.csv` or in the input population; the prototype_mtc population ships with `ptype` precomputed.
- Whether NTS age bins `≤10` and `10-21` are inclusive at 10 or 21 was not verified in the upstream preprocessing.

---

## Q2. CDAP: household interaction, utility terms, M/N/H definitions

### Takeaway
The local "CDAP" is a person-level MNL over M/N/H, estimated separately for each person type (`SegmentedMNL`) on all 8 NTS labels. It has none of ActivitySim's defining CDAP machinery:
- household member ranking,
- 2- and 3-way interaction terms,
- household-size-specific specs up to 5 members,
- fixed relative proportions for members beyond the 5th,
- ptype availability rules (M unavailable for non-workers and retirees).

The M/N definitions match ActivitySim. **H is never estimated or generated**, because the NTS input contains zero home-all-day person-days. This is a material behavioural difference: ActivitySim's H shares are 10–79% by ptype.

### Cited Findings
- **ActivitySim definitions:**
  - "Mandatory: the person engages in travel to at least one out-of-home mandatory activity - work, university, or school. The mandatory pattern may also include non-mandatory activities such as separate home-based tours or intermediate stops on mandatory tours."
  - "Non-mandatory: … only maintenance and discretionary tours."
  - "Home: the person does not travel outside the home."

  — [CDAP docs v1.6.0](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/cdap.md) (rendered: https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/cdap.html)
- **Household structure:**
  - Persons are ranked: FT workers (up to two), then PT workers (up to two workers in total), then children youngest to oldest (up to three), then others randomly up to five.
  - Individual M/N/H utilities are solved, then expression files are generated programmatically for household sizes 1–5 from an interaction-coefficients table.

  — [CDAP docs v1.6.0](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/cdap.md)
- **Code constants:** `MAX_HHSIZE = 5` and `MAX_INTERACTION_CARDINALITY = 3`. Spec construction: "cdap spec is same for all households of MAX_HHSIZE and greater". Persons with `cdap_rank > MAX_HHSIZE` are assigned by `extra_hh_member_choices` using `cdap_fixed_relative_proportions`. — [abm/models/util/cdap.py, lines 28–31, 406–407, 1101–1152](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/cdap.py)
- **Interaction terms (prototype_mtc):**
  - 2-way terms per activity and ptype pair (for example `H,11` and `M,12`).
  - 3-way terms (for example `H,124 / 122 / 144`).
  - "cdap_final_rules": `M,5 → coef_UNAVAILABLE` and `M,4 → coef_UNAVAILABLE`.
  - "cdap_all_people" terms for all members sharing the same pattern in 3-, 4- and 5-person households (`M,***` … `H,*****`).
  - The file has 52 H rows, 30 M rows and 51 N rows.

  — [cdap_interaction_coefficients.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/cdap_interaction_coefficients.csv)
- **Coefficient values:** `coef_UNAVAILABLE = -999`, `coef_H_11 = 1.626`, `coef_M_11 = 0.141`, `coef_H_xxxx = -3.733`. — [cdap_coefficients.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/cdap_coefficients.csv)
- **Individual utility spec (`cdap_indiv_and_hhsize1.csv`):**
  - ASCs per ptype for M and N; H is the base.
  - M is `coef_UNAVAILABLE` for ptype 4 (non-working adult) and ptype 5 (retired).
  - Age interactions: children 6–9 and 13–15; preschool 0–1 and 4–5; FT worker < 40; retired > 80.
  - Female interactions.
  - Cars vs workers (more / fewer).
  - Income bands (< 20k, 50–100k, > 100k).
  - Accessibility: `auPkTotal` for M, `auOpRetail` for N.

  — [cdap_indiv_and_hhsize1.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/cdap_indiv_and_hhsize1.csv)
- **Fixed relative proportions** for extra members (M / N / H):

  | Person type | M | N | H |
  |---|---|---|---|
  | FT worker | 0.796 | 0.094 | 0.110 |
  | PT worker | 0.617 | 0.258 | 0.126 |
  | University student | 0.692 | 0.156 | 0.151 |
  | Non-worker | 0 | 0.672 | 0.328 |
  | Retired | 0 | 0.543 | 0.457 |
  | Driving-age student | 0.776 | 0.060 | 0.164 |
  | School child | 0.685 | 0.091 | 0.223 |
  | Preschool | 0.141 | 0.065 | 0.794 |

  — [cdap_fixed_relative_proportions.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/cdap_fixed_relative_proportions.csv)
- **Joint tours in CDAP:** v1.6.0 has an optional `ADD_JOINT_TOUR_UTILITY` (default `False`) that adds a joint-tour component to CDAP. prototype_mtc does not enable it. — [abm/models/cdap.py, lines 30–37, 123–125](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/cdap.py)
- **Local label definition** (`composhed/activitysim/data.py:150`): `cdap = "M" if n_work + n_school else ("N" if tours else "H")`. A tour's purpose is work or education whenever the home-to-home segment contains any work or education episode: the priority primary rule is work > education > longest discretionary (`common/tours.py:128-149`). So M means at least one out-of-home work or education activity on the day, and N means only non-mandatory tours.
- **Local estimation** (`composhed/activitysim/models.py:297-299`):
  - Classes are only those observed among M, N, H.
  - `SegmentedMNL` fits one MNL per ptype with ≥ `n_min=200` observations, plus a pooled model (`segmented.py:193-208`).
  - Features are all 8 NTS labels one-hot encoded (`train.py:314`): age, hh_income, sex, employment, day, hh_zone, access_egress_distance, vehicles.
  - There are no ptype-specific availability rules.
- **No home-all-day days in the input.** Seed-0 validation shows CDAP classes only M and N. `info.extraction.tours_per_day` has no 0 key: 1 tour 41,954; 2 tours 11,760; 3 tours 2,475; and so on. — local `stability_runs/activitysim/seed_0/validation.json`
- The local docs say: "NTS has no home-all-day diaries, so the home option is only available where it is observed" and "One person-day: no household CDAP interactions or joint tours." — local `docs/activitysim.md:56, 92`
- **Local observed M shares by ptype (seed 0):**

  | Person type | Observed M share |
  |---|---|
  | FTW | 0.505 |
  | PTW | 0.405 |
  | DRIVING_STUDENT | 0.596 |
  | CHILD | 0.353 |
  | NONWORK | 0.095 |
  | RETIRED | 0.0069 |

  CDAP in-sample accuracy is 0.764. — local `validation.json` (`components["ASIM-2 CDAP"]`)
- Generation samples CDAP per person, returns an all-home day if H (`chain.py:36-41`), and skips mandatory tour frequency unless M (`chain.py:43-45`).

### Inferences
- **Matches:**
  - The M/N/H semantics are the same, including that M days may also contain non-mandatory tours and stops.
  - The M → mandatory tour frequency and N → non-mandatory frequency gating is the same.
  - Segment-specific coefficients by ptype: ActivitySim uses ptype-interacted terms in one utility table, which is equivalent in spirit to the local per-segment MNLs. The local version is more flexible, since every covariate is fully interacted with ptype.
  - Calibration of ASCs by segment follows ActivitySim practice.
- **Omissions justified by the data** (NTS person-days, no household linkage in this dataset, no zones):
  - household ranking;
  - 2- and 3-way interaction terms;
  - "all members same pattern" terms;
  - hhsize-specific specs;
  - fixed proportions for members beyond the 5th;
  - joint-tour CDAP;
  - car-vs-worker comparisons;
  - accessibility terms. `access_egress_distance` and `hh_zone` are only rough proxies for `auPkTotal` / `auOpRetail`.

  The local model is therefore effectively ActivitySim's household-size-1 individual-utility CDAP, without the H alternative.
- **Material deviation: no H.** ActivitySim's CDAP puts large mass on H (fixed proportions of 0.11–0.79). The local chain can never produce an at-home day. This makes the variant's day-pattern distribution structurally different from any ActivitySim run. It is an artefact of how the input data were prepared (the NTS-derived `activities.csv` appears to exclude no-travel days), not of the method. It should be stated explicitly in the paper, and the `chain.py:40` H branch is dead code under this data.
- **Deviation: ptype availability not enforced.** ActivitySim makes M unavailable for NONWORK and RETIRED. Locally, 9.5% of NONWORK and 0.7% of RETIRED days are M, and the model reproduces this. This follows from the NTS-based typing in Q1: people in "other" or "unknown" employment, or retirees, who nonetheless travel to work or education. Imposing ActivitySim's rule would require re-typing those people, not just masking M.
- **Extra covariate:** `day` (day of week) is in every CDAP model. ActivitySim models an average weekday. The local FTW M share of about 50%, versus ActivitySim's FT fixed-proportion M of 0.80, largely reflects the inclusion of weekends.
- **Doc claim check:** `docs/activitysim.md` rates CDAP "simplified", which is fair. It should add "no H alternative in practice" and "no ptype availability constraints".

### Gaps
- I did not check whether the upstream NTS preprocessing (foundata) deliberately drops zero-trip diary days. The local data simply contains none.
- The v1.6.0 docs page for CDAP has no numeric description of household-size handling beyond the code. Behaviour for households larger than 5 was taken from code comments, not docs.

---

## Q3. Mandatory tour frequency: alternatives and spec, local vs ActivitySim

### Takeaway
The alternative set is identical: `work1, work2, school1, school2, work_and_school`, applied to M persons only. The local model is a per-ptype MNL with the 8 NTS labels. Differences:
- It lacks ActivitySim's **ptype availability rules**.
- It lacks the location- and household-based terms.
- RETIRED and UNIV fall back to the pooled model.
- Days with 3 or more mandatory tours are truncated.
- `work_and_school` always schedules work first, whereas ActivitySim schedules school first for non-workers.

### Cited Findings
- **Alternatives:** `mandatory_tour_frequency_alternatives.csv` lists work1 (1, 0), work2 (2, 0), school1 (0, 1), school2 (0, 2) and work_and_school (1, 1). — [prototype_mtc](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/mandatory_tour_frequency_alternatives.csv)
- **Choosers:** only persons with `cdap_activity == "M"`. — [abm/models/mandatory_tour_frequency.py, line 88](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/mandatory_tour_frequency.py)
- **ptype availability rules** in `mandatory_tour_frequency.csv`:
  - FT and PT workers: school2 unavailable.
  - Non-worker and retired: school2 and work_and_school unavailable.
  - Driving-age child: work1 and work2 unavailable.
  - Pre-driving school child: work2 and work_and_school unavailable.
  - Preschool: only school1.
  - Work tours unavailable without a usual work location; school tours unavailable without a usual school location.

  — [mandatory_tour_frequency.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/mandatory_tour_frequency.csv)
- **Other utility terms** in the same file:
  - ptype ASCs;
  - female × ptype;
  - under 35;
  - can walk to work or school (distance < 3);
  - round-trip auto time to work or school;
  - student is employed;
  - non-student goes to school;
  - no cars, or fewer cars than drivers;
  - number of preschool children, non-workers, and under-16s not at school in the household;
  - income > 50k;
  - non-family household;
  - home is urban.

  — same file
- **Tour ordering:** "by default work tours are first for work_and_school tours; swap tour_nums for non-workers so school tour is 1 and work is 2". In other words, the work tour comes first for workers and second for non-workers. — [abm/models/util/tour_frequency.py, `process_mandatory_tours`, around lines 201–250](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/tour_frequency.py)
- **Doc error:** the v1.6.0 docs for MTF (and for mandatory scheduling) contain the boilerplate sentence "This model generates only True or False outcomes, and is structured as a binary logit model". This is incorrect for a 5-alternative MNL and is evidently copy-paste. — [MTF docs v1.6.0](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/mandatory_tour_frequency.md)
- **Local label mapping** (`composhed/activitysim/data.py:73-80`): any work plus any school → `work_and_school`. Otherwise work ≥ 2 → work2, 1 → work1; likewise for school. Tours are extracted with `collapse_mandatory_tours=False` (`data.py:145`).
- **Local model** (`models.py:300-304`): `SegmentedMNL` over `mtf_classes`, which are the alternatives observed anywhere among M persons. The same class set applies to every ptype; there is no availability matrix.
- **Generation** (`chain.py:43-45`): samples an MTF label, then `mtf_tours(label)` gives `["work", "school"]` for work_and_school (`data.py:83-85`). Tours are scheduled in that order.
- **Local evidence (seed 0):**
  - 18,921 M persons; MTF in-sample accuracy 0.959.
  - RETIRED and UNIV fall below `n_min=200` and use the pooled model (`pooled_segments: ['RETIRED','UNIV']`).
  - Observed shares:

    | Person type | work1 | work2 | school1 | school2 | work_and_school |
    |---|---|---|---|---|---|
    | FTW | 0.973 | 0.023 | 0.0038 | — | 0.0001 |
    | PTW | 0.866 | 0.032 | 0.098 | 0.0004 | 0.0044 |
    | DRIVING_STUDENT | 0.028 | — | 0.962 | 0.009 | 0.0012 |
    | CHILD | — | — | 0.995 | 0.005 | — |
    | NONWORK | 0.729 | 0.028 | 0.239 | 0.004 | — |

  — local `validation.json`
- **Mandatory tours per day:** 0: 37,813; 1: 18,515; 2: 392; 3: 13; 4: 1. 11 tours mixed work and education in a single home-to-home tour. — local `validation.json` `info.extraction`

### Inferences
- **Matches:**
  - The alternative set and M-only choosers are the same.
  - Segment-by-ptype structure is the same in spirit.
  - The local "school" includes university, as in ActivitySim's `school` tour_type (MTC splits `univ` only for scheduling and logsums).
- **Deviations:**
  1. **No ptype availability.** In ActivitySim a FTW cannot choose school2, and a driving-age child cannot choose work1 or work2. The local model can assign any observed alternative to any ptype, with probabilities from the segment MNL. Observed NTS behaviour also contradicts the rules: DRIVING_STUDENT work1 is 2.8%, NONWORK work1 is 73%. Hard-coding ActivitySim's availability would therefore conflict with the NTS typing. This is a justified simplification, but `docs/activitysim.md:57` labels MTF "full", which overstates fidelity. "Simplified" is more accurate.
  2. **Covariates:** none of ActivitySim's location, travel-time, household-composition or car-sufficiency terms are present; the 8 NTS labels stand in.
  3. **Pooling:** RETIRED and UNIV M choosers are predicted by the pooled all-ptype MNL. That model has no ptype indicator, but the `employment` and `age` dummies largely recover it.
  4. **Truncation:** days with 3–4 mandatory tours (14 persons) are mapped to work2, school2 or work_and_school. When work ≥ 2 and school ≥ 1, the label is work_and_school, so one work tour is dropped from the frequency label. Scheduling estimation (`models.py:276-291`) still uses all observed tours, so the estimation data includes some third and later tours that generation can never produce. The impact is negligible but inconsistent.
  5. **Mixed work+education single tour:** typed by its first mandatory episode (`tours.py:131`), so 11 such tours count as work or school only, not work_and_school.
- **Bug-level item:** the work-first ordering for `work_and_school` (`data.py:67`, `TOUR_ORDER = ["work", "school"] + NM_ORDER`; `chain.py:52`) differs from ActivitySim, which puts school first for non-workers (students). Only about 0.1% of DRIVING_STUDENT and 0.4% of PTW M-days are affected, so the practical impact is tiny. It is a fidelity error rather than a simplification, and easy to fix: order by `is_worker`, i.e. ptype in {FTW, PTW}.

### Gaps
- I did not retrieve `mandatory_tour_frequency_coefficients.csv` values. They are not needed for the structural comparison.

---

## Q4. Mandatory tour scheduling: alternatives, joint choice, time windows, previous-tour terms, logsums

### Takeaway
The local `TourTOD` reproduces the core ActivitySim design: a conditional logit over an enumerated table of all (departure period, arrival period) pairs with start ≤ end, hourly from 05:00, with availability masked by a person timetable and tours scheduled sequentially (work before school). It differs as follows:
1. It adds an explicit 00:00–05:00 early period: 20 periods and 210 alternatives, versus ActivitySim MTC's 19 hourly periods (5–23) and 190 alternatives, where times before 05:00 are shifted to period 5.
2. It uses a minute-level timetable with a 10-minute home buffer instead of ActivitySim's period-level start/end/middle window states.
3. It replaces ActivitySim's grouped departure, arrival and duration constants and linear "shift" terms with per-period start and end dummies, 9 duration-bin dummies, and start, end, duration and duration² × covariates.
4. It omits:
   - mode-choice logsums (justified: no skims);
   - the timetable-derived terms: previous tour ends in this departure hour, previous tour begins in this arrival hour, adjacent-window terms, remaining tours ÷ unscheduled hours;
   - first-of-2+-tour terms;
   - `work_and_school` duration terms.

   The omissions in item 4, apart from the logsums, are not forced by the data.

### Cited Findings
- **Alternatives:** `tour_departure_and_duration_alternatives.csv` (prototype_mtc) lists every `start,end` pair from `5,5` to `23,23` with start ≤ end: 19 hours and 190 alternatives. — [tour_departure_and_duration_alternatives.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_departure_and_duration_alternatives.csv)
- **Time periods:** "For `prototype_mtc`, the modeled time periods for all submodels are hourly from 3 am to 3 am the next day, and any times before 5 am are shifted to time period 5, and any times after 11 pm are shifted to time period 23." — [mandatory scheduling docs v1.6.0](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/mandatory_scheduling.md) (rendered: https://activitysim.github.io/activitysim/v1.6.0/dev-guide/components/mandatory_scheduling.html)
- **Skim periods:** `network_los.yaml` has `skim_time_periods: time_window 1440, period_minutes 60`, with skim labels EA / AM / MD / PM / EV used for logsums. — [network_los.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/network_los.yaml)
- **Model description:** the model "selects a tour departure and duration period (and therefore a start and end period as well) for each mandatory tour. The primary drivers … are accessibility-based parameters such as the mode choice logsum for the departure/arrival hour combination, demographics, and time pattern characteristics such as the time windows available from previously scheduled tours." An optional `tour_departure_and_duration_segments.csv` enables "representative logsums". — [mandatory scheduling docs v1.6.0](https://github.com/ActivitySim/activitysim/blob/v1.6.0/docs/dev-guide/components/mandatory_scheduling.md)
- **Segmentation:**
  - `mandatory_tour_scheduling.yaml` uses separate specs for work and school; `univ` reuses the school spec but has its own logsum segment.
  - `LOGSUM_SETTINGS: tour_mode_choice.yaml`.
  - In code, `mandatory_tour_seg` = tour_type, with school tours of university students relabelled `univ`.

  — [mandatory_tour_scheduling.yaml](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/mandatory_tour_scheduling.yaml); [abm/models/mandatory_scheduling.py, lines 47–61](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/mandatory_scheduling.py)
- **Sequencing and availability:**
  - `vectorize_tour_scheduling` loops `for tour_num, nth_tours in tours.groupby("tour_num", sort=True)`: all persons' first tours, then second tours, with the timetable updated in between.
  - `tdd_interaction_dataset` drops unavailable alternatives via `timetable.tour_available(window_row_ids, alts_ids)`.
  - Choice is made by `interaction_sample_simulate`.
  - `start_previous` / `end_previous` are joined for tour_num ≥ 2.

  — [abm/models/util/vectorize_tour_scheduling.py, lines 570–650, 723–883, 1083](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/abm/models/util/vectorize_tour_scheduling.py)
- **Timetable mechanics:**
  - Each period is coded EMPTY / START / END / MIDDLE / START_END.
  - Collisions are only START+START, END+END, and any overlap with MIDDLE. So a new tour may depart in the same hour a previous tour arrived (END + START = START_END).
  - Helpers: `previous_tour_ends`, `previous_tour_begins`, `adjacent_window_before/after`, `remaining_periods_available`.

  — [core/timetable.py, lines 21–43, 475–620, 655–771](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/core/timetable.py)
- **Work scheduling spec terms** (`tour_scheduling_work.csv`, prototype_mtc):
  - Shift effects, linear in start or duration hours: round-trip auto time; PT worker departure; non-worker duration; university departure; income; CBD destination.
  - "Subsequent tour must start after previous tour ends" (coefficient −100).
  - First-of-2+ and second work tour departure and duration shifts.
  - Early-departure and late-arrival interactions with income > 100k, CBD and rural home.
  - FT worker duration < 9 h; FT worker 10–12 departure; PT worker 13–15 arrival.
  - Duration < 8 h for first or subsequent of 2+ tours and for work_and_school workers or students.
  - **Mode choice logsum** (coefficient 1.027).
  - Previously scheduled tour ends in this departure hour (−0.8935) and begins in this arrival hour (−1.334).
  - Adjacent-window-before/after × first or second+ tour.
  - Remaining tours ÷ unscheduled hours (−18.68).
  - **Departure constants** in 10 groups: ≤5, 6, 7, 8, 9, 10–12, 13–15, 16–18, 19–21, ≥22.
  - **Arrival constants** in 10 groups: ≤6, 7–9, 10–12, 13–14, 15, 16, 17, 18, 19–21, ≥22.
  - **Duration constants** in 9 groups: 0–2, 3–4, 5–6, 7–8, 9, 10, 11, 12–13, 14–18 h.

  — [tour_scheduling_work.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_scheduling_work.csv); [tour_scheduling_work_coefficients.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_scheduling_work_coefficients.csv)
- **School spec:** similar, with ptype shift terms, logsum, previous and adjacent window terms. Note an upstream labelling inconsistency: "Student driving age duration shift effects" is coded `(ptype == 7)`, which is the pre-driving type. — [tour_scheduling_school.csv](https://github.com/ActivitySim/activitysim/blob/v1.6.0/activitysim/examples/prototype_mtc/configs/tour_scheduling_school.csv)
- **Local grid** (`common/periods.py:212-214`): `PeriodGrid.with_early_bin(60)` gives edges `[0, 300, 360, …, 1440]`: one 00:00–05:00 period plus 19 hourly periods (05:00–24:00), 20 in total. `pairs()` enumerates start ≤ end, giving 210 alternatives (`periods.py:236-239`). Arrival uses the end-period convention: an end on a period edge belongs to the earlier period (`periods.py:229-231`).
- **Local utility** (`common/tod.py:32-51`): `ConditionalLogit` with
  - (a) sparse generic dummies: start period (19), end period (19) and 9 implied-duration bins with edges 0, 1, 2, 3, 4, 6, 8, 10, 12, 24 h (`DUR_BIN_EDGES`, `tod.py:29`), first of each block dropped;
  - (b) dense attributes `[mid_start, mid_end, dur, dur²]/1440` interacted with `z`.

  For scheduling, `z = [1, label one-hots, ptype one-hots, NM-purpose one-hots, nth>0]` (`activitysim/models.py:202-204`). There are separate `TourTOD` models for the groups work, school and nonmandatory (`models.py:189, 324-335`).
- **Local availability:**
  - `TourTOD.mask` → `TimeWindow.period_availability_mask` (`tod.py:66-69`; `timewindow.py:375-392`). A period pair is available if some minute-level start/end inside one usable free window exists, where usable windows are shrunk by a 10-minute buffer next to existing tours and by 1 minute at the day edges (`timewindow.py:341-353`).
  - Training masks come from observed earlier tours in scheduling order (`models.py:276-291`). Generation reserves each placed tour (`tourgen.py:446-470`).
  - Within-period minutes come from the empirical `IntraPeriodSampler` (`tod.py:107-115`; `periods.py:242-295`).
- **Local doc claim** (`docs/activitysim.md:58`): "ActivitySim uses piecewise departure, arrival and duration shift terms; here period constants plus interactions." Fidelity is rated "simplified".
- **Local fit quality (seed 0):**

  | Group | Tours | McFadden ρ² | Mean observed start | Mean predicted start | forced_available |
  |---|---|---|---|---|---|
  | work | 14,204 | 0.238 | 499.8 min | 506.3 min | 0 |
  | school | 5,138 | 0.576 | — | — | 0 |

  `forced_available = 0` means no observed choice was masked out. — local `validation.json`

### Inferences
- **Matches:**
  - Joint (departure, arrival) choice from an enumerated table with start ≤ end.
  - Hourly resolution from 05:00.
  - MNL with availability from a person timetable.
  - Sequential scheduling, with work before school and mandatory before non-mandatory.
  - Constants on departure, arrival and duration.
  - ptype interactions.
  - Calibration of time-of-day constants (local `train.py:269-276` calibrates start, end and duration-bin shares, as ActivitySim calibration does).
- **Deviations, justified or neutral:**
  - **Early bin:** local keeps a separate 00:00–05:00 period and keeps 23:00–24:00 as hour 23. ActivitySim MTC shifts times before 05:00 into period 5. Local is a strict superset, so every NTS day stays representable (SPECIFICATION ASIM-4, `SPECIFICATION.md:237`). Neutral.
  - **Constants granularity:** local has a full set of per-hour start and end dummies, finer than ActivitySim's grouped departure and arrival constants. Its duration bins are coarser than ActivitySim's in the 8–12 h range: local 8–10 and 10–12 h, versus ActivitySim 9, 10 and 11 h singletons. This could under-resolve the FT-work 9-hour peak; the dense `dur, dur²` terms partly compensate.
  - **Shift terms:** local dense terms `start, end, dur, dur²` × (labels, ptype) are functionally the same as ActivitySim's "shift effects" (covariate × start or duration), plus an arrival shift and a quadratic. This is a reasonable equivalent, and the docs describe it fairly.
  - **No mode-choice logsum, CBD, auto-time, or rural early/late terms:** these are forced by the absence of locations and skims. `hh_zone` enters only via `z`.
  - **Timetable:** local works at minute level with a 10-minute home buffer (`HOME_INTER`). ActivitySim works at period level and allows departing in the same hour as the previous arrival. The two give similar but not identical feasibility. Locally, two short tours may start in the same hour (ActivitySim forbids START+START); and a tour may be infeasible locally where ActivitySim would allow it, if less than 10 minutes remain.
- **Simplifications not forced by the data** (could be added with the existing timetable):
  1. Previous tour ends in this departure hour / begins in this arrival hour.
  2. Adjacent-window-before/after × first or second+ tour.
  3. Remaining mandatory tours ÷ remaining available periods.
  4. "First of 2+ tours" departure and duration shifts. Local has only an `nth > 0` flag (`models.py:204`). `tour_count` is known from the MTF choice but not used.
  5. `work_and_school` × worker/student duration < 8 h terms.

  These are what make ActivitySim's second-tour timing respond to the first. Their absence matters most for work2, school2 and work_and_school days, which are about 2% of M persons.
- **Segmentation:** local has no separate `univ` scheduling segment. ActivitySim only separates it for logsums and reuses the school spec. With no logsums, the local merge is equivalent apart from the ptype dummy.
- **Ordering error (as in Q3):** for `work_and_school`, local always schedules work first. ActivitySim schedules school first for non-workers.
- **Minor design note:** ptype dummies in `z_tod` are near-collinear with the age and employment label dummies, also in `z`. This is harmless under L2 but makes ptype scheduling coefficients uninterpretable for the paper.
- **No outright correctness bugs** found in the mandatory scheduling path:
  - masks are built from the same ordered tours used for estimation;
  - `forced_available = 0` for work and school;
  - the period-pair mask correctly requires a feasible minute realisation within a single window.

### Gaps
- I did not verify whether any ActivitySim v1.6.0 example besides prototype_mtc uses 30-minute tdd periods. Several newer regional examples, such as SANDAG ABM3 and SEMCOG, use 48 half-hour periods; that is not confirmed from v1.6.0 files here. The local `--time-resolution 30` option would correspond to those.
- I did not inspect the prototype_mtc `settings.yaml` for any global time-period override beyond `network_los.yaml`. It was downloaded but not grepped in detail.
- How the local early 00:00–05:00 bin interacts with the `with_early_bin(15/30)` ablations was not checked.
