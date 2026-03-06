# Composhed

A compositional baseline model for 24-hour activity schedule generation, built as a theory-driven comparison point for [Caveat](https://github.com/big-ucl/caveat).

## What it is

Composhed is an econometric activity-scheduling model estimated from the UK National Travel Survey (NTS). It generates synthetic 24-hour sequences of `(activity_type, duration)` pairs, conditioned on person attributes (gender, age, car access, work status, household income). The architecture is explicitly compositional — six separately-estimated statistical models assembled by a rule-based algorithm — making it a concrete comparison against the deep generative approaches in Caveat.

It is loosely inspired by CEMDAP/DaySim-style tour-based models, deliberately simplified to the same scope as Caveat: activity type and duration only, no location or mode choice.

## Install

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync

# Optional: install dev dependencies (Jupyter etc.)
uv sync --group dev
```

A `uv.lock` lockfile is generated on first `uv sync` and should be committed.

For the Caveat evaluation library (not a runtime dependency):

```bash
uv pip install git+https://github.com/big-ucl/caveat
```

## Model overview

Schedule generation proceeds in six sequential steps:

1. **Daily Activity Pattern (DAP)** — multinomial logit classifies the day as home-only (`H`), mandatory-only (`W`), mandatory + discretionary (`WD`), or discretionary-only (`D`).
2. **Mandatory duration** — log-normal OLS predicts work/education activity duration (if DAP ∈ {W, WD}).
3. **Number of non-mandatory tours** — ordered logit predicts how many discretionary activities occur (if DAP ∈ {WD, D}).
4. **Activity type per slot** — multinomial logit predicts the type of each discretionary activity (shop, visit, escort, medical, other); separate models for slots 1, 2, and 3+.
5. **Activity duration per type** — log-normal OLS predicts duration of each discretionary activity, estimated separately per activity type.
6. **Schedule assembly** — rule-based algorithm anchors timing using KDE-sampled work-start or first-departure times, places activities, and enforces the 24-hour budget.

Every step samples stochastically from predicted distributions (never argmax) to preserve distributional diversity across identical inputs.

## Data

Expected input format:

- **Attributes csv** — one row per person with columns: `pid`, `gender`, `age`, `car_access`, `work_status`, `household_income`
- **Schedules csv** — NTS schedules as sequences of activities, with types and durations, adding to 24 hours

The pre-processing pipeline for NTS data is available from [Caveat](https://github.com/big-ucl/caveat) as an [exaple notebook](https://github.com/big-ucl/caveat/tree/main/examples).

## Usage

`uv run` uses the project's `.venv` automatically — no need to activate it manually.

**Train** all six sub-models (original compositional model):

```bash
uv run compsched-train \
  --attributes tmp/nts_attributes_2023.csv \
  --schedules tmp/nts_schedules_2023.csv \
  --output-dir models/
```

This saves a single bundle to `models/compsched_models.pkl`.

**Generate** synthetic schedules (original model):

```bash
uv run compsched-generate \
  --attributes tmp/nts_attributes_2023.csv \
  --models models/compsched_models.pkl \
  --out-attributes output/synthetic_attributes.csv \
  --out-schedules output/synthetic_schedules.csv
```

**Train** the MDCEV variant:

```bash
uv run compsched-train-mdcev \
  --attributes tmp/nts_attributes_2023.csv \
  --schedules tmp/nts_schedules_2023.csv \
  --output-dir models/
```

**Generate** synthetic schedules (MDCEV variant):

```bash
uv run compsched-generate-mdcev \
  --attributes tmp/nts_attributes_2023.csv \
  --models models/mdcev_models.pkl \
  --out-attributes output/synthetic_mdcev_attributes.csv \
  --out-schedules output/synthetic_mdcev_schedules.csv
```

Output csvs have columns `pid, act, start, end, duration` (schedules) and `pid, gender, age, car_access, work_status, household_income` (attributes), matching the Caveat synthetic data format.

## Todo

- Evaluation notebook comparing against Caveat baselines (EMD, feasibility rate, creativity)
- Calibration plots for each sub-model
- Work-based subtour decomposition (currently collapsed into the work activity)
- Location/mode choice modules (currently omitted by design to match Caveat scope)
- Config file support as an alternative to CLI args
