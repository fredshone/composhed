# Composhed

Econometric baselines for generating 24-hour activity schedules, built as
comparison points for the deep generative models in
[Caveat](https://github.com/big-ucl/caveat) (ActVAE). Evaluation uses
[acteval](https://github.com/fredshone/acteval).

Every model is estimated on the UK National Travel Survey 2023 (NTS). Each one
generates a day for a person as a sequence of `(activity, start, end, duration)`
rows, conditioned on that person's attributes:

* **Attributes:** age, sex, employment, household income, car access, area
  type, distance band and day of week.
* **Activity types:** home, work, education, shop, visit, escort, medical and
  other.

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

Follows DaySim's tour-based design:

1. A day-pattern choice fixes how many tours of each purpose the person makes.
2. Tours are scheduled in priority order (work, education, then the rest).
3. Each tour chooses its departure and return half-hour together, from the
   time earlier tours have left free.
4. Stops are added to each half of the tour.

→ [docs/daysim.md](docs/daysim.md)

### ActivitySim-aligned

Follows ActivitySim's person-level chain, with separate coefficients for each
person type (full-time worker, retiree, student, ...):

1. Daily activity pattern.
2. Mandatory tour frequency and scheduling.
3. Non-mandatory tour frequency and scheduling.
4. Stop frequency, trip purpose and trip timing.

Scheduling works on an hourly grid against a timetable of free hours.
→ [docs/activitysim.md](docs/activitysim.md)

### Common ground for the source-aligned variants

The CEMDAP, DaySim and ActivitySim variants share several features:

* the same scope: no locations or modes, zero travel time, and one person-day
  at a time;
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
  --attributes /path/to/attributes_binned.csv \
  --schedules /path/to/activities.csv \
  --output-dir models/ --seed 0

uv run composhed-generate-cemdap \
  --attributes /path/to/attributes_binned.csv \
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
  --target-schedules /path/to/activities.csv \
  --modelled-schedules synthetic_schedules.csv other_schedules.csv \
  --output-dir results/
```

### Stability study

`run_stability.py` runs every chosen model and seed through three stages:

1. Train.
2. Generate one day for every person in `--attributes`.
3. Evaluate the result with acteval against `--schedules`.

`scripts/summarise_runs.py` then averages the results over seeds.

```bash
uv run python run_stability.py \
  --attributes /path/to/attributes_binned.csv \
  --schedules /path/to/activities.csv \
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

NTS 2023, 5 seeds, one synthetic day per person (56,734). Values are acteval
distances, so lower is better, shown as mean ± standard deviation across
seeds. Feasibility is 0 for every model.

| Model | Participations | Transitions | Timing | Creativity |
|---|---|---|---|---|
| Compositional | 0.045 ± 0.001 | 0.0031 ± 0.0001 | 0.053 ± 0.000 | 0.062 ± 0.001 |
| MDCEV¹ | 0.265 ± 0.007 | 0.0098 ± 0.0002 | 0.081 ± 0.001 | 0.099 ± 0.005 |
| CEMDAP | 0.041 ± 0.002 | 0.0027 ± 0.0001 | 0.023 ± 0.000 | **0.015 ± 0.000** |
| DaySim | 0.046 ± 0.003 | 0.0015 ± 0.0001 | **0.018 ± 0.000** | 0.207 ± 0.001 |
| ActivitySim | **0.031 ± 0.003** | **0.0014 ± 0.0000** | 0.019 ± 0.001 | 0.209 ± 0.001 |

| Ablation | Participations | Transitions | Timing | Creativity |
|---|---|---|---|---|
| CEMDAP, no calibration | 0.043 ± 0.001 | 0.0027 ± 0.0001 | 0.022 ± 0.000 | 0.014 ± 0.000 |
| DaySim, 15-min periods | 0.044 ± 0.007 | 0.0015 ± 0.0000 | 0.019 ± 0.000 | 0.201 ± 0.002 |
| ActivitySim, no segmentation | 0.032 ± 0.008 | 0.0014 ± 0.0000 | 0.019 ± 0.001 | 0.209 ± 0.002 |

¹ MDCEV was estimated on 5,000 people with `--free-gamma` (the harness setting); its episode and timing models use everyone.

**How to read these results:**

* **They measure fit, not generalisation.** Generation uses the training
  attributes and is compared with the training schedules.
* **No repairs were needed.** No source-aligned run needed a person-level
  fallback or a rescale.
* **DaySim and ActivitySim score poorly on creativity.** They copy observed
  days more often: about 13% of their days exactly match a training day,
  against about 1% for the other models.

---

## Todo

- Check each source-aligned component against its primary source ([docs/citations.md](docs/citations.md))
- Household CDAP and joint tours for the ActivitySim variant
- Seed the MDCEV calibration sample so calibrated MDCEV runs are reproducible
- Evaluation on a held-out test split, matching ActVAE
- Tours from work (currently merged into the work block)
- Location and mode choice (omitted by design, to match Caveat's scope)
