# Composhed

A compositional baseline model for 24-hour activity schedule generation, built as a theory-driven comparison point for [Caveat](https://github.com/big-ucl/caveat).

## What it is

Composhed is an econometric activity-scheduling model estimated from the UK National Travel Survey (NTS). It generates synthetic 24-hour sequences of `(activity_type, duration)` pairs, conditioned on person attributes (gender, age, car access, work status, household income). The architecture is explicitly compositional - six separately-estimated statistical models assembled by a rule-based algorithm - making it a concrete comparison against the deep generative approaches in Caveat.

It is loosely inspired by CEMDAP/DaySim-style tour-based models, deliberately simplified to the same scope as Caveat: activity type and duration only, no location or mode choice.

# Model Descriptions

## Compositional baseline

CompSched follows the sequential tour-based paradigm of DaySim (Bowman & Ben-Akiva, 2001) and CEMDAP (Bhat et al., 2004), decomposing schedule generation into a hierarchy of independently estimated sub-models - multinomial logit for daily activity pattern and activity type, ordered logit for number of tours, and log-normal OLS regression for durations - assembled into valid 24-hour sequences by a rule-based algorithm. The architecture follows DaySim and CEMDAP most directly, applying the same sequential discrete choice hierarchy to 24-hour schedules. actiTopp (Hilgert et al., 2017) is architecturally similar in its stepwise regression approach but generates weekly rather than daily schedules.

## MDCEV variant

The MDCEV variant replaces the five sequential sub-models with a single Multiple Discrete-Continuous Extreme Value model (Bhat, 2005; 2008), estimated using Biogeme (Bierlaire, 2003). MDCEV treats 24-hour time allocation as a simultaneous portfolio choice, jointly predicting both activity participation and duration within a fixed 1 to 440-minute budget, before passing outputs to the same assembly step as the compositional baseline.

## References

- Bowman, J.L. & Ben-Akiva, M.E. (2001). Activity-based disaggregate travel demand model system with activity schedules. *Transportation Research Part A*, 35(1), 1–28.
- Bhat, C.R., Guo, J.Y., Srinivasan, S. & Sivakumar, A. (2004). Comprehensive econometric microsimulator for daily activity-travel patterns. *Transportation Research Record*, 1894, 57–66.
- Hilgert, T., Heilig, M., Kagerbauer, M. & Vortisch, P. (2017). Modeling week activity schedules for travel demand models. *Transportation Research Record*, 2666, 69–77.
- Bhat, C.R. (2005). A multiple discrete-continuous extreme value model: formulation and application to discretionary time-use decisions. *Transportation Research Part B*, 39(8), 679–707.
- Bhat, C.R. (2008). The multiple discrete-continuous extreme value (MDCEV) model: role of utility function parameters, identification considerations, and model extensions. *Transportation Research Part B*, 42(3), 274–303.
- Bierlaire, M. (2003). BIOGEME: a free package for the estimation of discrete choice models. *Proceedings of the 3rd Swiss Transportation Research Conference (STRC)*, Ascona, Switzerland.


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

Both variants share the same **input variables** and **output format**:

**Inputs (conditioning variables):** `age`, `sex`, `employment`, `hh_income`, `hh_zone`, `day`, `vehicles`, `access_egress_distance` - all one-hot encoded, with nulls filled as `"unknown"`.

**Outputs:** A 24-hour sequence of `(activity_type, duration_minutes)` pairs, where activity types are drawn from `{home, work, education, shop, visit, escort, medical, other}` and durations sum to 1440 minutes.

**Scaling:** The core compositional models and the MDCEV model itself use the raw one-hot label encoding. A single `StandardScaler` is fit only for the shared anchor-timing logistic regression, where scaled label features help stabilize the before-work classifier; the fitted scaler is saved in the model bundle and reused at generation time.

---

### Approach 1 - Compositional (baseline)

Six independently-estimated models assembled by a rule-based algorithm:

| Step | Model type | Predicts |
|------|-----------|---------|
| 1. DAP classification | Multinomial logit (statsmodels `MNLogit`) | Day structure: home-only (`H`), mandatory-only (`W`), mandatory + discretionary (`WD`), or discretionary-only (`D`) |
| 2. Mandatory duration | Log-normal OLS (`LinearRegression` on log-duration) | Work/education activity duration in minutes; active if DAP ∈ {W, WD} |
| 3. Number of tours | Ordered logit (`OrderedModel`) | Count of discretionary activities (0–4); active if DAP ∈ {WD, D} |
| 4. Activity type per slot | Multinomial logit, separate models for slots 1, 2, 3+ | Discretionary activity type: shop, visit, escort, medical, or other |
| 5. Activity duration per type | Log-normal OLS, separate model per activity type | Duration of each discretionary activity |
| 6. Schedule assembly | KDE + rule-based algorithm | Work-start / first-departure timing (KDE per employment category); home-time split (Beta distribution); before/after-work placement (logistic regression on scaled label features) |

Every step samples stochastically from predicted distributions (never argmax) to preserve distributional diversity across identical inputs.

---

### Approach 2 - MDCEV variant

A single Multiple Discrete-Continuous Extreme Value (MDCEV) model estimated with Biogeme replaces Steps 1–5. It jointly predicts time allocation across all 8 activity types simultaneously, then the same Step 6 assembly algorithm places activities in time.

| Component | Model type | Predicts |
|-----------|-----------|---------|
| Time allocation | MDCEV GammaProfile (Biogeme + PyTorch sampling) | Minutes allocated to each activity type in one joint pass; a zero allocation means the type is not participated in |
| Schedule assembly | KDE + rule-based algorithm (same as compositional) | Timing, ordering, and 24-hour budget enforcement; still uses the shared scaled anchor-timing classifier |

The MDCEV approach captures correlations between activity type choices and durations that the compositional pipeline treats as independent. The trade-off is less interpretable per-step coefficients and a dependency on Biogeme's MDCEV estimation.

## Usage

`uv run` uses the project's `.venv` automatically - no need to activate it manually.

**Train** all six sub-models (compositional):

```bash
uv run compsched-train \
  --attributes /home/fred/Data/foundata/out/nts/2023/attributes_binned.csv \
  --schedules /home/fred/Data/foundata/out/nts/2023/activities.csv \
  --output-dir models/
```

Saves to `models/composhed_models.pkl`.

**Generate** synthetic schedules (compositional):

```bash
uv run compsched-generate \
  --attributes /home/fred/Data/foundata/out/nts/2023/attributes_binned.csv \
  --models models/composhed_models.pkl \
  --out-attributes synthetic_attributes.csv \
  --out-schedules synthetic_schedules.csv
```

**Train** the MDCEV variant:

```bash
uv run compsched-train-mdcev \
  --attributes /home/fred/Data/foundata/out/nts/2023/attributes_binned.csv \
  --schedules /home/fred/Data/foundata/out/nts/2023/activities.csv \
  --output-dir models/
```

Saves to `models/mdcev_models.pkl`.

**Generate** synthetic schedules (MDCEV variant):

```bash
uv run compsched-generate-mdcev \
  --attributes /home/fred/Data/foundata/out/nts/2023/attributes_binned.csv \
  --models models/mdcev_models.pkl \
  --out-attributes synthetic_mdcev_attributes.csv \
  --out-schedules synthetic_mdcev_schedules.csv
```

**Evaluate** against a reference dataset:

```bash
uv run compsched-evaluate \
  --target-schedules /home/fred/Data/foundata/out/nts/2023/activities.csv \
  --modelled-schedules synthetic_schedules.csv synthetic_mdcev_schedules.csv \
  --output-dir results/
```

Output CSVs have columns `pid, act, start, end, duration` (schedules) and `pid, age, hh_income, sex, employment, day, hh_zone, access_egress_distance, vehicles` (attributes), matching the Caveat synthetic data format.

## Latest Results

Trained and evaluated using nts 2023. Lower is better.

| Domain          | Compositional | MDCEV variant | ACTVAE |
|-----------------|---------------|---------------|--------|
| Creativity      | 0.246         | 0.053         | **0.018**  |
| Feasibility     | **0.000**     | **0.000**     | 0.030  |
| Participations  | 0.602         | 0.257         | **0.072** |
| Timing          | 0.163         | 0.114         | **0.029** |
| Transitions     | 0.015         | 0.017         | **0.005** |


## Todo

- Train and generation timing
- Evaluation notebook comparing against Caveat baselines (EMD, feasibility rate, creativity)
- Calibration plots for each sub-model
- Work-based subtour decomposition (currently collapsed into the work activity)
- Location/mode choice modules (currently omitted by design to match ActVAE/Caveat scope)
- Config file support as an alternative to CLI args
