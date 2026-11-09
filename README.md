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
  --ablations daysim:res15,activitysim:no-segmentation,activitysim:res30 \
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
is calibrated** (calibration has been removed from the code), and **no output is
post-processed**: every generated episode is written as generated.

acteval scores each combination of two labels on its own (for example
`age = 18-27 & employment = student`) and then averages over the slices.
There are 676 slices; the 40 with fewer than 30 observed person-days are
skipped. The question is whether each model reproduces schedules *for each
kind of person*, not just overall.

| Model | Participations | Transitions | Timing | Creativity | Infeasible |
|---|---|---|---|---|---|
| Compositional | **0.050 ± 0.001** | 0.0189 ± 0.0004 | 0.051 ± 0.000 | 0.089 ± 0.001 | 0 |
| MDCEV¹ | 0.186 ± 0.009 | 0.0503 ± 0.0021 | 0.069 ± 0.002 | **0.084 ± 0.007** | 0.091 |
| CEMDAP | 0.061 ± 0.000 | 0.0192 ± 0.0002 | 0.0276 ± 0.0001 | 0.088 ± 0.001 | 0 |
| DaySim | 0.059 ± 0.001 | **0.0142 ± 0.0001** | **0.0259 ± 0.0001** | 0.092 ± 0.001 | 0.0014 |
| ActivitySim | 0.069 ± 0.001 | 0.0159 ± 0.0001 | 0.0322 ± 0.0002 | 0.105 ± 0.001 | 0.0021 |

| Ablation | Participations | Transitions | Timing | Creativity | Infeasible |
|---|---|---|---|---|---|
| DaySim, 15-min periods | 0.059 ± 0.002 | 0.0141 ± 0.0003 | 0.0254 ± 0.0004 | 0.092 ± 0.001 | 0.0014 |
| ActivitySim, no segmentation² | 0.078 ± 0.002 | 0.0224 ± 0.0002 | 0.0333 ± 0.0002 | 0.105 ± 0.001 | 0.0021 |
| ActivitySim, 30-min periods² | 0.069 ± 0.001 | 0.0160 ± 0.0001 | 0.0290 ± 0.0002 | 0.096 ± 0.000 | 0.0020 |

¹ MDCEV was estimated on 500 person-days with free gamma
(`--mdcev-max-records 500`). Its episode-count and timing models use all
person-days.

² Run with the ActivitySim version before the November 2026 check against its
code (`stability_runs_nts2023/_previous_version_4/` has that version's main
runs). Compare them with that version's ActivitySim row: 0.069 ± 0.001,
0.0161 ± 0.0002, 0.0314 ± 0.0001, 0.105 ± 0.001, 0.0021.

"Infeasible" is acteval's share of days that do not start and end at home or
that have two neighbouring home, work or education episodes. The observed data
has none.

* **They measure fit, not generalisation.** Generation uses the training
  attributes and is compared with the training schedules.
* **No repairs.** No source-aligned run needed a person-level fallback or a
  rescale, and nothing is dropped or merged: zero-length episodes are kept,
  neighbouring episodes of one type are kept, and there is no minimum home stay
  between tours. Interrupted work or education is not merged before
  estimation: DaySim and ActivitySim model it with work / school stops, CEMDAP
  with work-based tours.
* **Each source-aligned model keeps its source's time precision.** CEMDAP is
  continuous (rounded to minutes when written), DaySim chooses half-hour
  periods and then a uniform free minute, and ActivitySim works in
  `prototype_mtc`'s 19 hourly periods (05:00–23:00, earlier times in the
  first), written at hh:30 (11% of its rows are zero-length).
* **Infeasible days are now visible.** MDCEV puts two home episodes next to
  each other on 9% of days (earlier rounds merged them). In DaySim and
  ActivitySim, 0.1–0.2% of days have a work or school stop next to the main
  work or school episode.
* **ActivitySim's creativity got worse (0.089 → 0.105) because of its hourly
  times.** Every time it writes is hh:30, and 19% of observed starts
  are at hh:30. So more generated days copy a training day exactly (novel
  0.997 → 0.967).
* **Most scores improved over the previous round**
  (`stability_runs_nts2023/_previous_version_3/`), because the observed data
  also has neighbouring episodes of one type. Participations fell for every
  model (compositional 0.057 → 0.050, CEMDAP 0.067 → 0.061, DaySim 0.066 → 0.059, ActivitySim 0.079 →
  0.069), and so did CEMDAP's timing (0.030 → 0.028). ActivitySim's timing got
  worse (0.029 → 0.031) with the coarser times.
* **ActivitySim now follows its source code, not only its configs.** It was
  checked in November 2026 against the v1.6.0 code (timetable, tour
  numbering, trip scheduling, trip purpose, frequency extension) and every
  difference found was fixed: `prototype_mtc`'s 19 periods (no 00:00–05:00
  period), only the spec files' person-type terms and reference constants,
  ActivitySim's own timetable terms (adjacent windows as 0/1 flags), the
  "subsequent tour must start after previous tour ends" rule, trip departures
  drawn and redrawn as `trip_scheduling.py` (`logic_version: 2`), and the
  probability tables' own keys. Together they changed little: timing got
  worse (0.0314 → 0.0322), transitions slightly better (0.0161 → 0.0159),
  the rest is unchanged. About 90
  tours per run (of 91,706 days) get an outbound departure after an inbound
  one, as ActivitySim allows when only one leg is redrawn; their times are
  clamped and counted (`trip_departure_out_of_order`). See
  [docs/source_differences.md](docs/source_differences.md).
* **DaySim and ActivitySim use their sources' variables.** Each model has
  only the terms its source uses whose variables NTS has, plus Saturday and
  Sunday; the sources' availability rules apply; no rules were added for our
  output format. ActivitySim reads `prototype_mtc`'s own spec files. See
  [docs/source_differences.md](docs/source_differences.md).
* **ActivitySim's per-person-type coefficients help.** Non-mandatory tour
  frequency has one coefficient set per person type, as in the source. Without
  it, participations go from 0.069 to 0.078 (previous version, ²).
* **15-minute periods barely change DaySim** and take much longer to train.
* **Half-hour periods help ActivitySim's timing (0.031 → 0.029) and
  creativity (0.105 → 0.096)** (previous version, ²) but not participations or transitions. The
  ablation uses half-hour periods from 03:00 with one 00:00–03:00 period, as
  ActivitySim's half-hour examples do, and keeps `prototype_mtc`'s terms. Its
  times are written at hh:15 and hh:45, so fewer days copy a training day
  exactly (novel 0.967 → 0.984).
* **Rounds before `_previous_version_3/` are not comparable.** acteval 0.5,
  the `wt_travel_time_to_cities` label (which replaced
  `access_egress_distance`) and so the slices all changed; even the
  compositional model moved (transitions 0.005 → 0.020, creativity 0.218 →
  0.089). `stability_runs_nts2023/_previous_version_3/` has the previous round
  (merging, minimum home stays, ActivitySim stops dropped, minutes for every
  model), `_previous_version_2/` all labels in every model with an older
  acteval, `_previous_version/` minutes copied from NTS, and `stability_runs/`
  an older NTS extract with calibration.
* **Result files:** `stability_runs_nts2023/summary_table.csv` (per-seed
  values are in `summary.csv`). The command was:

```bash
uv run python run_stability.py --attributes .../binned_attributes.parquet \
  --schedules .../activity_schedules.parquet --out-dir stability_runs_nts2023 \
  --seeds 0,1,2,3,4 --mdcev-max-records 500 \
  --ablations daysim:res15,activitysim:no-segmentation,activitysim:res30
```

---

## Todo

- Check the CEMDAP components against their primary sources ([docs/citations.md](docs/citations.md)); DaySim and ActivitySim follow their source code and spec files ([docs/source_differences.md](docs/source_differences.md))
- ActivitySim variant: per-person random-number streams (as ActivitySim)
- Household CDAP and joint tours for the ActivitySim variant
- Evaluation on a held-out test split, matching ActVAE
- Tours from work (currently merged into the work block)
- Location and mode choice (omitted by design, to match Caveat's scope)
