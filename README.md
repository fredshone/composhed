# Composhed

Econometric baselines for generating 24-hour activity schedules, built as
comparison points for the deep generative models in
[Caveat](https://github.com/big-ucl/caveat) (ActVAE). Evaluation uses
[acteval](https://github.com/fredshone/acteval).

Every model is estimated on the UK National Travel Survey 2023 (NTS). Each one
generates a day for a person as a sequence of `(activity, start, end, duration)`
rows, conditioned on that person's attributes:

* **Attributes:** age, sex, employment, household income, vehicles per adult,
  area type, distance band and day of week.
* **Activity types:** home, work, education, shop, visit, escort, leisure and
  other.

The input is the foundata NTS 2023 output
(`binned_attributes.parquet`, `activity_schedules.parquet`): 91,706 person-days,
41% of them spent at home all day. The loaders read parquet or CSV.

Like Caveat, the models cover activity type and timing only, with no locations
or travel modes.

## Models

Five models are implemented. They range from a deliberately simple chain to
re-implementations that follow the structure of three published activity-based
modelling frameworks.

| Model | In one line | Docs |
|---|---|---|
| **Compositional** (baseline) | Six small models (day type, durations, counts, types, timing, tour placement), assembled by rules | [docs/compositional.md](docs/compositional.md) |
| **MDCEV** | One joint model of time allocation across activity types, then a timing layout | [docs/mdcev.md](docs/mdcev.md) |
| **CEMDAP-aligned** | Continuous-time chain with hazard-based start and duration models | [docs/cemdap.md](docs/cemdap.md) |
| **DaySim-aligned** | Day pattern, then tours timed in priority order within free time windows | [docs/daysim.md](docs/daysim.md) |
| **ActivitySim-aligned** | Person-type-specific chain of pattern, tour, scheduling and stop models | [docs/activitysim.md](docs/activitysim.md) |

### Compositional baseline

The original CompoShed model, loosely following DaySim and CEMDAP. Six models
are estimated independently:

1. A logit picks the day type: home, work, education, discretionary only, or
   work / education plus discretionary.
2. Regressions draw the work or education duration.
3. An ordered logit draws the number of other activities.
4. A logit draws each activity's type.
5. Regressions draw each activity's duration.
6. Timing comes from kernel density estimates, and tour placement from two
   logistic models.

A rule-based algorithm assembles the pieces, shrinking them proportionally if
they overflow the day. There is no calibration.
→ [docs/compositional.md](docs/compositional.md)

### MDCEV

A Multiple Discrete-Continuous Extreme Value model, estimated with Biogeme,
splits the 1,440 minutes across the eight activity types in one joint draw.
Zero minutes means the person doesn't do that activity. Per-type episode
counts and start-time regressions then split each total into episodes. A
constrained optimisation lays the episodes out as close to their predicted
start times as possible. It uses five of the eight attributes. On the full NTS
data it must be estimated on a subsample (`--max-records`) to fit in memory.
→ [docs/mdcev.md](docs/mdcev.md)

### CEMDAP-aligned

Follows CEMDAP's sequence:

1. Does the person work or study today?
2. When do they start, and for how long, given that start?
3. How many stops on the way there, on the way back, and on separate trips
   before and after?
4. Each stop's type and duration, and the home stays between trips.

Durations come from hazard-based (AFT) models, and each draw is limited to the
time still free, so days never overflow.
→ [docs/cemdap.md](docs/cemdap.md)

### DaySim-aligned

Follows DaySim's model chain, with each model's utility terms transcribed
from the DaySim source code (keeping those whose variables NTS has) and
DaySim's worker / student availability rules:

1. A day-pattern choice over all 1,324 combinations says which tour purposes
   and which stop purposes the day has; a second model sets the exact number
   of tours of each purpose.
2. Tours are scheduled in priority order (work, education, then the rest).
3. Each tour chooses the half-hours of arrival at and departure from its main
   activity together, from the time earlier tours have left free.
4. Each half of the tour then gets its stops one at a time, working outwards
   from the main activity: the stop's purpose, then the half-hour it starts
   (or ends).

→ [docs/daysim.md](docs/daysim.md)

### ActivitySim-aligned

Follows ActivitySim's person-level chain, in ActivitySim's order. The choice
models read ActivitySim's own `prototype_mtc` spec files (vendored in
`composhed/activitysim/specs/`), with one non-mandatory frequency coefficient
set per person type as there:

1. Daily activity pattern.
2. Mandatory tour frequency, then scheduling.
3. Non-mandatory tour frequency (seeing how much of the day is still free),
   then scheduling.
4. Stop frequency, trip purpose and trip timing anchored on the tour's hours.

Scheduling works on an hourly grid against a timetable of free hours, with
the utility terms of ActivitySim's `prototype_mtc` spec files. Household and
joint models are left out (one person-day).
→ [docs/activitysim.md](docs/activitysim.md)

### Common ground for the source-aligned variants

The CEMDAP, DaySim and ActivitySim variants share several features:

* the same scope: no locations or modes, zero travel time, and one person-day
  at a time;
* exact minutes drawn uniformly inside chosen time periods (DaySim and
  ActivitySim);
* DaySim and ActivitySim use their sources' own variables, mapped from the
  NTS labels, with day of week as the only addition
  ([docs/source_differences.md](docs/source_differences.md));
* coefficients re-estimated on NTS, with no calibration;
* a report of every repair made during generation;
* byte-identical output for a fixed seed.

They are described in [docs/source_aligned.md](docs/source_aligned.md). How
NTS days are split into tours is in [docs/tour_rules.md](docs/tour_rules.md),
and sources for every component are in [docs/citations.md](docs/citations.md).

---

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync                                           # create the venv and install dependencies
uv sync --group dev                               # optional: Jupyter etc.
```

---

## Usage

Every model has a train and a generate command:

| Model | Train | Generate | Model file |
|---|---|---|---|
| Compositional | `composhed-train` | `composhed-generate` | `composhed_models.pkl` |
| MDCEV | `composhed-train-mdcev` | `composhed-generate-mdcev` | `mdcev_models.pkl` |
| CEMDAP | `composhed-train-cemdap` | `composhed-generate-cemdap` | `cemdap_models.pkl` |
| DaySim | `composhed-train-daysim` | `composhed-generate-daysim` | `daysim_models.pkl` |
| ActivitySim | `composhed-train-activitysim` | `composhed-generate-activitysim` | `activitysim_models.pkl` |

```bash
uv run composhed-train-cemdap \
  --attributes /path/to/binned_attributes.parquet \
  --schedules /path/to/activity_schedules.parquet \
  --output-dir models/ --seed 0

uv run composhed-generate-cemdap \
  --attributes /path/to/binned_attributes.parquet \
  --models models/cemdap_models.pkl \
  --out-attributes synthetic_attributes.csv \
  --out-schedules synthetic_schedules.csv --seed 0
```

The output files follow the Caveat synthetic data format:

* **Schedules:** `pid, act, start, end, duration`.
* **Attributes:** `pid` plus the eight attributes.

Model-specific options are listed on each model's docs page.

### Evaluate

```bash
uv run composhed-evaluate \
  --target-schedules /path/to/activity_schedules.parquet \
  --modelled-schedules synthetic_schedules.csv other_schedules.csv \
  --output-dir results/
```

### Stability study

`run_stability.py` runs every chosen model and seed through three stages:

1. Train.
2. Generate one day for every person in `--attributes`.
3. Evaluate the result with acteval against `--schedules`.

`scripts/summarise_runs.py` then averages the results over seeds.

Stability runs usually share the machine, so their times are not clean.
`scripts/time_models.py` takes the same `--attributes`, `--schedules`,
`--variants` and `--ablations` options and
trains and generates one model at a time, without evaluation. It writes wall
time, CPU time, peak memory and the load average before each stage to
`timing_runs/timings.csv` (default one seed, one BLAS thread). Run it on an
otherwise idle machine; it warns when the load average is above 1.

```bash
uv run python run_stability.py \
  --attributes /path/to/binned_attributes.parquet \
  --schedules /path/to/activity_schedules.parquet \
  --variants compositional,mdcev,cemdap,daysim,activitysim \
  --ablations daysim:res15,activitysim:no-segmentation \
  --mdcev-max-records 5000 \
  --out-dir stability_runs/
uv run python scripts/summarise_runs.py --runs-dir stability_runs/ [--external actvae.csv]
```

| Option | Default | Effect |
|---|---|---|
| `--seeds` | `0,1,2,3,4` | Seeds to run |
| `--variants` | all five | Models to run |
| `--ablations` | none | Extra runs as `<model>:<preset>`. Presets: `independent-timing`, `rescale`, `res15`, `res30`, `res60`, `no-segmentation` |
| `--target-schedules` | `--schedules` | Schedules to evaluate against |
| `--evaluate / --no-evaluate` | on | Run acteval for each run |
| `--split-order` | 2 | Score each slice of the eight labels at this acteval order (2: every pair of labels). 0: whole population only. |
| `--skip-existing` | off | Skip runs that already have results |
| `--mdcev-max-records` | all | MDCEV estimation sample size. Use 5000 on full NTS, which otherwise runs out of memory. |

Each run writes to `stability_runs/<model>[__<ablation>]/seed_<n>/`:

* the model files;
* the synthetic CSVs;
* `run.log`;
* `metrics.csv` (one row per acteval metric);
* for the source-aligned variants, `validation.json` and
  `generation_report.json`.

The summary script writes `summary.csv` (one row per run) and
`summary_table.csv` (mean ± standard deviation per model).

---

## Results

These results use the foundata NTS 2023 data: 91,706 person-days, of which 41%
are stay-at-home days. Each run generates one synthetic day per person-day.
Scores are acteval 0.5 distances within label pairs (`split_order=2`), so lower
is better. Each value is the mean ± standard deviation over 5 seeds. **No model
is calibrated** (calibration has been removed from the code). Feasibility is 0
for every model.

acteval scores each combination of two labels on its own (for example
`age = 18-27 & employment = student`) and then averages over the slices.
There are 676 slices; the 40 with fewer than 30 observed person-days are
skipped. The question is whether each model reproduces schedules *for each
kind of person*, not just overall.

| Model | Participations | Transitions | Timing | Creativity |
|---|---|---|---|---|
| Compositional | **0.057 ± 0.001** | 0.0201 ± 0.0004 | 0.052 ± 0.000 | 0.089 ± 0.001 |
| MDCEV¹ | 0.223 ± 0.013 | 0.0439 ± 0.0007 | 0.078 ± 0.003 | **0.084 ± 0.007** |
| CEMDAP | 0.067 ± 0.000 | 0.0204 ± 0.0003 | 0.030 ± 0.000 | 0.088 ± 0.001 |
| DaySim | 0.066 ± 0.001 | **0.0152 ± 0.0001** | **0.0266 ± 0.0002** | 0.091 ± 0.001 |
| ActivitySim | 0.079 ± 0.002 | 0.0183 ± 0.0001 | 0.0290 ± 0.0002 | 0.089 ± 0.001 |

| Ablation | Participations | Transitions | Timing | Creativity |
|---|---|---|---|---|
| DaySim, 15-min periods | 0.067 ± 0.002 | 0.0153 ± 0.0002 | 0.0263 ± 0.0002 | 0.092 ± 0.000 |
| ActivitySim, no segmentation | 0.086 ± 0.001 | 0.0241 ± 0.0001 | 0.0311 ± 0.0002 | 0.089 ± 0.001 |

¹ MDCEV was estimated on 500 person-days with free gamma
(`--mdcev-max-records 500`). Its episode-count and timing models use all
person-days.

* **They measure fit, not generalisation.** Generation uses the training
  attributes and is compared with the training schedules.
* **No repairs were needed.** No source-aligned run needed a person-level
  fallback or a rescale. About 4% of days in each source-aligned model have
  two neighbouring episodes of one type merged; ActivitySim drops about 300
  stops a run that do not fit their tour.
* **DaySim and ActivitySim now use their sources' variables.** Each model has
  only the terms its source uses whose variables NTS has, plus Saturday and
  Sunday; the sources' availability rules apply; no rules were added for our
  output format. ActivitySim reads `prototype_mtc`'s own spec files. See
  [docs/source_differences.md](docs/source_differences.md).
* **That removed their advantage on participations.** With every label in
  every model, ActivitySim and DaySim were best on participations; now the
  compositional model, which still uses all eight labels, is. DaySim remains
  best on transitions and timing, which come from its tour and stop structure.
* **ActivitySim's per-person-type coefficients help.** Non-mandatory tour
  frequency has one coefficient set per person type, as in the source. Without
  it, participations go from 0.079 to 0.086.
* **15-minute periods barely change DaySim** and take much longer to train.
* **Earlier rounds are not comparable.** acteval 0.5, the
  `wt_travel_time_to_cities` label (which replaced `access_egress_distance`)
  and so the slices all changed; even the unchanged compositional model moved
  (transitions 0.005 → 0.020, creativity 0.218 → 0.089). Earlier results are
  in `stability_runs_nts2023/_previous_version_2/` (all labels in every
  model, older acteval), `_previous_version/` (minutes copied from NTS) and
  `stability_runs/` (older NTS extract, calibrated).
* **Result files:** `stability_runs_nts2023/summary_table.csv` (per-seed
  values are in `summary.csv`). The command was:

```bash
uv run python run_stability.py --attributes .../binned_attributes.parquet \
  --schedules .../activity_schedules.parquet --out-dir stability_runs_nts2023 \
  --seeds 0,1,2,3,4 --mdcev-max-records 500 \
  --ablations daysim:res15,activitysim:no-segmentation
```

---

## Todo

- Check the CEMDAP components against their primary sources ([docs/citations.md](docs/citations.md)); DaySim and ActivitySim follow their source code and spec files ([docs/source_differences.md](docs/source_differences.md))
- ActivitySim variant: per-person random-number streams (as ActivitySim)
- Household CDAP and joint tours for the ActivitySim variant
- Evaluation on a held-out test split, matching ActVAE
- Tours from work (currently merged into the work block)
- Location and mode choice (omitted by design, to match Caveat's scope)
