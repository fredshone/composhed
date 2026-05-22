# Composhed

A compositional baseline model for 24-hour activity schedule generation, built as a theory-driven comparison point for [Caveat](https://github.com/big-ucl/caveat). Evaluations are intended to be made using [acteval](https://github.com/fredshone/acteval).

## What it is

Composhed is an econometric activity-scheduling model estimated from the UK National Travel Survey (NTS). It generates synthetic 24-hour sequences of `(activity_type, duration)` pairs, conditioned on person attributes (age, sex, employment, household income, car access, area type, and day of week).

The architecture is explicitly compositional — six separately-estimated statistical models assembled by a rule-based algorithm — making it a concrete, interpretable comparison against the deep generative approaches in Caveat. It is loosely inspired by CEMDAP/DaySim-style tour-based models, deliberately simplified to the same scope as Caveat: activity type and duration only, no location or mode choice.

Two variants are implemented:

| Variant | Steps 1–5 | Step 6 |
|---------|-----------|--------|
| **Compositional** (baseline) | Five sequential sub-models (MNL, ordered logit, log-normal OLS) | Rule-based assembly with budget rescaling |
| **MDCEV** | Single joint time-allocation model (Biogeme + PyTorch) | MDCEV-specific assembly — no rescaling; episode splitting per activity type |

---

## Models

Both variants share the same **input variables** and **output format**.

**Inputs (conditioning variables):** `age`, `sex`, `employment`, `hh_income`, `hh_zone`, `day`, `vehicles`, `access_egress_distance` — all one-hot encoded, with nulls filled as `"unknown"`.

**Outputs:** A 24-hour sequence of `(activity_type, duration_minutes)` pairs, where activity types are drawn from `{home, work, education, shop, visit, escort, medical, other}` and durations sum to 1440 minutes.

**Scaling:** The core models use raw one-hot encoding. A `StandardScaler` is fit only for the shared anchor-timing logistic regression (before-work placement classifier); the fitted scaler is saved in the model bundle and reused at generation time.

### Compositional baseline

Follows the sequential tour-based paradigm of DaySim (Bowman & Ben-Akiva, 2001) and CEMDAP (Bhat et al., 2004). Six independently-estimated models are assembled in sequence:

| Step | Model | Predicts |
|------|-------|----------|
| 1. DAP classification | Multinomial logit (`MNLogit`) | Day structure: home-only `H`, work-only `W`, work + discretionary `WD`, education-only `E`, education + discretionary `ED`, or discretionary-only `D` |
| 2. Mandatory duration | Log-normal OLS | Work/education duration in minutes; active if DAP ∈ {W, WD, E, ED}; conditioned on person features plus two DAP flags: `dap_WD` (has discretionary) and `is_education` (education vs work) |
| 3. Number of tours | Ordered logit (`OrderedModel`) | Count of discretionary activities (0–4); active if DAP ∈ {WD, ED, D} |
| 4. Activity type per slot | Multinomial logit, separate models for slots 1, 2, 3+ | Discretionary activity type: shop, visit, escort, medical, or other |
| 5. Activity duration per type | Log-normal OLS, separate model per activity type | Duration of each discretionary activity |
| 6. Schedule assembly | KDE + rule-based algorithm | Work-start / first-departure timing (KDE per employment category); home-time split (Beta distribution); before/after-work placement (logistic regression) |

Every step samples stochastically from predicted distributions (never argmax) to preserve distributional diversity across identical inputs.

### MDCEV variant

A single Multiple Discrete-Continuous Extreme Value (MDCEV) model (Bhat, 2005; 2008), estimated with Biogeme (Bierlaire, 2003), replaces Steps 1–5. It jointly predicts time allocation across all 8 activity types simultaneously — zero allocation means that type is not participated in.

The MDCEV output (durations summing exactly to 1440 minutes) is passed to a dedicated assembly step that differs from the compositional pipeline:

| Assembly element | Behaviour |
|---|---|
| Non-home durations | Taken verbatim from MDCEV — never rescaled |
| Home time | Computed as `1440 − sum(non-home)` and split around the anchor start |
| Anchor timing | KDE per employment category, same as compositional (now covers work and education) |
| Episode splitting | `EpisodeCountModel`: Poisson regression per type gives n episodes; total duration is divided among them |
| Before/after placement | Logistic regression per episode, same as compositional |

The MDCEV approach captures correlations between activity participation and duration that the compositional pipeline treats as independent. The trade-off is less interpretable per-step coefficients and a dependency on Biogeme's MDCEV estimation.

---

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

---

## Usage

`uv run` uses the project's `.venv` automatically — no need to activate it manually.

### Train

**Compositional baseline:**

```bash
uv run compsched-train \
  --attributes /path/to/attributes_binned.csv \
  --schedules /path/to/activities.csv \
  --output-dir models/
```

Saves to `models/composhed_models.pkl`.

**MDCEV variant:**

```bash
uv run compsched-train-mdcev \
  --attributes /path/to/attributes_binned.csv \
  --schedules /path/to/activities.csv \
  --output-dir models/
```

Saves to `models/mdcev_models.pkl`.

### Generate

**Compositional baseline:**

```bash
uv run compsched-generate \
  --attributes /path/to/attributes_binned.csv \
  --models models/composhed_models.pkl \
  --out-attributes synthetic_attributes.csv \
  --out-schedules synthetic_schedules.csv
```

**MDCEV variant:**

```bash
uv run compsched-generate-mdcev \
  --attributes /path/to/attributes_binned.csv \
  --models models/mdcev_models.pkl \
  --out-attributes synthetic_mdcev_attributes.csv \
  --out-schedules synthetic_mdcev_schedules.csv
```

Output CSVs have columns `pid, act, start, end, duration` (schedules) and `pid, age, hh_income, sex, employment, day, hh_zone, access_egress_distance, vehicles` (attributes), matching the Caveat synthetic data format.

### Evaluate

```bash
uv run compsched-evaluate \
  --target-schedules /path/to/activities.csv \
  --modelled-schedules synthetic_schedules.csv synthetic_mdcev_schedules.csv \
  --output-dir results/
```

### Stability check

`run_stability.py` trains and generates across multiple random seeds to assess result stability. It runs both variants by default and writes per-seed output to a structured directory tree.

```bash
uv run python run_stability.py \
  --attributes /path/to/attributes_binned.csv \
  --schedules /path/to/activities.csv \
  --out-dir stability_runs/
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds` | `0,1,2,3,4` | Comma-separated list of seeds to run |
| `--out-dir` | `stability_runs` | Root output directory |
| `--mdcev-max-records` | `5000` | Cap training records for MDCEV (reduces runtime) |
| `--skip-mdcev` | off | Run compositional variant only |

**Output layout:**

```
stability_runs/
└── seed_0/
    ├── compositional/
    │   ├── models/composhed_models.pkl
    │   ├── synthetic_attributes.csv
    │   ├── synthetic_schedules.csv
    │   └── run.log
    └── mdcev/
        ├── models/mdcev_models.pkl
        ├── synthetic_mdcev_attributes.csv
        ├── synthetic_mdcev_schedules.csv
        └── run.log
```

A summary table is printed on completion showing train/generate status and wall-clock time for each seed × variant combination.

---

## Results

Trained and evaluated on NTS 2023. Lower is better.

| Domain | Compositional | MDCEV | ActVAE |
|--------|---------------|-------|--------|
| Creativity | 0.246 | 0.053 | **0.018** |
| Feasibility | **0.000** | **0.000** | 0.030 |
| Participations | 0.602 | 0.257 | **0.072** |
| Timing | 0.163 | 0.114 | **0.029** |
| Transitions | 0.015 | 0.017 | **0.005** |

---

## References

- Bowman, J.L. & Ben-Akiva, M.E. (2001). Activity-based disaggregate travel demand model system with activity schedules. *Transportation Research Part A*, 35(1), 1–28.
- Bhat, C.R., Guo, J.Y., Srinivasan, S. & Sivakumar, A. (2004). Comprehensive econometric microsimulator for daily activity-travel patterns. *Transportation Research Record*, 1894, 57–66.
- Hilgert, T., Heilig, M., Kagerbauer, M. & Vortisch, P. (2017). Modeling week activity schedules for travel demand models. *Transportation Research Record*, 2666, 69–77.
- Bhat, C.R. (2005). A multiple discrete-continuous extreme value model: formulation and application to discretionary time-use decisions. *Transportation Research Part B*, 39(8), 679–707.
- Bhat, C.R. (2008). The multiple discrete-continuous extreme value (MDCEV) model: role of utility function parameters, identification considerations, and model extensions. *Transportation Research Part B*, 42(3), 274–303.
- Bierlaire, M. (2003). BIOGEME: a free package for the estimation of discrete choice models. *Proceedings of the 3rd Swiss Transportation Research Conference (STRC)*, Ascona, Switzerland.

---

## Todo

- Train and generation timing benchmarks
- Evaluation notebook comparing against Caveat baselines (EMD, feasibility rate, creativity)
- Calibration plots for each sub-model
- Work-based subtour decomposition (currently collapsed into the work activity)
- Location/mode choice modules (currently omitted by design to match ActVAE/Caveat scope)
- Config file support as an alternative to CLI args
