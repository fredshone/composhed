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

Follows DaySim's model chain, with each model's terms taken from the DaySim
source code:

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

Follows ActivitySim's person-level chain, in ActivitySim's order, with
separate models for each person type (full-time worker, retiree, student, ...):

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
* a simulation-based calibration stage;
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
`--variants`, `--ablations` and `--calibrate / --no-calibrate` options and
trains and generates one model at a time, without evaluation. It writes wall
time, CPU time, peak memory and the load average before each stage to
`timing_runs/timings.csv` (default one seed, one BLAS thread). Run it on an
otherwise idle machine; it warns when the load average is above 1.

```bash
uv run python run_stability.py \
  --attributes /path/to/binned_attributes.parquet \
  --schedules /path/to/activity_schedules.parquet \
  --variants compositional,mdcev,cemdap,daysim,activitysim \
  --ablations cemdap:no-calibrate,daysim:res15,activitysim:no-segmentation \
  --mdcev-max-records 5000 \
  --out-dir stability_runs/
uv run python scripts/summarise_runs.py --runs-dir stability_runs/ [--external actvae.csv]
```

| Option | Default | Effect |
|---|---|---|
| `--seeds` | `0,1,2,3,4` | Seeds to run |
| `--variants` | all five | Models to run |
| `--ablations` | none | Extra runs as `<model>:<preset>`. Presets: `no-calibrate`, `independent-timing`, `rescale`, `res15`, `res30`, `res60`, `no-segmentation` |
| `--target-schedules` | `--schedules` | Schedules to evaluate against |
| `--evaluate / --no-evaluate` | on | Run acteval for each run |
| `--calibrate / --no-calibrate` | on | `--no-calibrate` skips calibration for every model that has it (MDCEV, CEMDAP, DaySim, ActivitySim) |
| `--split-order` | none | Score each slice of the eight labels at this acteval order (e.g. `2` for label pairs). Default: whole population only. |
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
Scores are acteval distances within label pairs (`split_order=2`), so lower is
better. Each value is the mean ± standard deviation over 5 seeds. **No model
is calibrated** (`--no-calibrate`). Feasibility is 0 for every model.

acteval scores each combination of two labels on its own (for example
`age = 18-27 & employment = student`) and then averages over the slices.
There are 749 slices. The 42 with fewer than 30 observed person-days are
skipped. The question is whether each model reproduces schedules *for each
kind of person*, not just overall.

| Model | Participations | Transitions | Timing | Creativity |
|---|---|---|---|---|
| Compositional | 0.043 ± 0.001 | 0.0050 ± 0.0000 | 0.052 ± 0.000 | 0.218 ± 0.001 |
| MDCEV¹ | 0.157 ± 0.009 | 0.0098 ± 0.0001 | 0.081 ± 0.003 | **0.202 ± 0.009** |
| CEMDAP | 0.048 ± 0.001 | 0.0048 ± 0.0000 | 0.031 ± 0.000 | 0.207 ± 0.001 |
| DaySim | 0.065 ± 0.001 | **0.0038 ± 0.0000** | 0.032 ± 0.000 | 0.258 ± 0.001 |
| ActivitySim | **0.031 ± 0.001** | 0.0042 ± 0.0000 | **0.023 ± 0.000** | 0.267 ± 0.001 |

| Ablation | Participations | Transitions | Timing | Creativity |
|---|---|---|---|---|
| DaySim, 15-min periods | 0.065 ± 0.001 | 0.0038 ± 0.0000 | 0.032 ± 0.000 | 0.258 ± 0.001 |
| ActivitySim, no segmentation | 0.035 ± 0.001 | 0.0044 ± 0.0000 | 0.024 ± 0.000 | 0.267 ± 0.001 |

¹ MDCEV was estimated on 500 person-days with free gamma
(`--mdcev-max-records 500`). Its episode-count and timing models use all
person-days.

* **They measure fit, not generalisation.** Generation uses the training
  attributes and is compared with the training schedules.
* **No repairs were needed.** No source-aligned run needed a person-level
  fallback or a rescale.
* **Without calibration, DaySim's participations score is weak.** Its
  day-pattern model, which has 62 patterns, under-predicts stay-at-home days.
* **ActivitySim's per-person-type coefficients help.** Its participations
  score is 0.031 with them and 0.035 without.
* **DaySim and ActivitySim copy observed days most often.** About 7% of their
  days exactly match a training day, against under 0.5% for the other models.
  This is why their creativity scores are high.
* **Earlier results are not comparable.** Those used the older NTS extract
  (56,734 days, no stay-at-home days, `medical` and `vehicles`) and calibrated
  models. They are kept in `stability_runs/`.
* **Result files:** `stability_runs_nts2023/summary_table.csv` (per-seed
  values are in `summary.csv`). The command was:

```bash
uv run python run_stability.py --attributes .../binned_attributes.parquet \
  --schedules .../activity_schedules.parquet --out-dir stability_runs_nts2023 \
  --seeds 0,1,2,3,4 --no-calibrate --split-order 2 --mdcev-max-records 500 \
  --ablations daysim:res15,activitysim:no-segmentation
```

---

## Todo

- Check the CEMDAP and DaySim components against their primary sources ([docs/citations.md](docs/citations.md)); ActivitySim was checked against v1.6.0 (`reports/ActivitySim variant fidelity.md`)
- ActivitySim variant: per-person random-number streams (as ActivitySim), so calibration passes reuse the same draws exactly
- Household CDAP and joint tours for the ActivitySim variant
- Seed the MDCEV calibration sample so calibrated MDCEV runs are reproducible
- Evaluation on a held-out test split, matching ActVAE
- Tours from work (currently merged into the work block)
- Location and mode choice (omitted by design, to match Caveat's scope)
