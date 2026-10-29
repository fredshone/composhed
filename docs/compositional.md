# Compositional baseline

**Code:** `composhed/models/`, `composhed/assembly.py`, `composhed/train.py`, `composhed/generate.py`

The compositional baseline is the original CompoShed model. It builds a day in
the spirit of DaySim and CEMDAP, but in a deliberately simple way. Six small
statistical models are estimated independently, and each one answers a single
question:

1. What kind of day is it? (home all day, work, education, discretionary only, ...)
2. How long is the work or education activity?
3. How many discretionary activities are there?
4. What type is each one?
5. How long does each one last?
6. When does the day start, and how are the activities grouped into trips from home?

A rule-based algorithm then assembles the answers into a 24-hour schedule. If
the sampled pieces don't fit in the day, they are shrunk proportionally rather
than rejected and redrawn.

Every step samples from its predicted distribution rather than taking the most
likely value. That is what lets two people with identical attributes get
different days.

## Running it

```bash
uv run composhed-train --attributes A.csv --schedules S.csv --output-dir models/ --seed 0
uv run composhed-generate --attributes A.csv --models models/composhed_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

Training writes `models/composhed_models.pkl`. There are no other options: the
model has no calibration step and no built-in ablations.

## Inputs

All eight person attributes (`age`, `hh_income`, `sex`, `employment`, `day`,
`hh_zone`, `access_egress_distance`, `veh_per_adult`) are one-hot encoded, with the
first category dropped. Missing values become their own `"unknown"` category.
The two logistic models in step 6 also standardise their features; the fitted
scaler is saved in the model bundle.

## Components

| Step | What it decides | Model | Inputs beyond the person attributes | Code |
|---|---|---|---|---|
| 1. Day pattern | One of six day types: home only (`H`), work only (`W`), education only (`E`), work + discretionary (`WD`), education + discretionary (`ED`), discretionary only (`D`) | Multinomial logit (statsmodels) | — | `models/dap.py` |
| 2. Mandatory duration | Total minutes of work or education (`W`, `WD`, `E`, `ED` days) | Log-normal regression with residual noise, clipped to 30–600 min | has-discretionary flag, education flag | `models/mandatory.py` |
| 3. Number of discretionary activities | 0–4 activities (`WD`, `ED`, `D` days) | Ordered logit (statsmodels); counts that would not fit at 30 min each are ruled out | has-discretionary flag, remaining time budget | `models/ntours.py` |
| 4. Activity type | escort, leisure, other, shop or visit, one per activity | Multinomial logit, one model each for the 1st, 2nd and 3rd+ activity | has-discretionary flag, slot number, remaining budget | `models/atype.py` |
| 5. Activity duration | Minutes for each discretionary activity | Log-normal regression per activity type with residual noise, clipped to [10 min, remaining budget] | remaining budget | `models/duration.py` |
| 6a. Anchor time | Start of work / education, or first departure on a `D` day | Kernel density estimate per employment category | — | `models/anchor.py` |
| 6b. Tour placement (`WD`, `ED`) | Whether each discretionary activity is a stop on the way to or from work / education, or its own trip from home before or after | 4-class logistic regression (scikit-learn) | anchor time, activity type | `models/anchor.py` |
| 6c. Tour grouping (`D`) | Whether consecutive activities share a trip from home | Binary logistic regression (scikit-learn) | slot, both activity types, remaining budget | `models/anchor.py` |
| 6d. Assembly | Final sequence of activities and home stays | Rules: 10 min at home between separate trips (`HOME_INTER`), at least 60 min at home in total, proportional shrinking when the day overflows | — | `assembly.py` |

The anchor model also fits a Beta distribution to the share of home time spent
in the morning. Nothing uses it at generation time.

### How training data is extracted

* **Day type:** comes from which activity types appear in the day.
* **Mandatory time:** all work and education time in the day is summed into a
  single mandatory activity, which starts at the first mandatory episode.
* **Stops around the mandatory activity:** a discretionary activity that comes
  after work in the same trip from home counts as an inbound stop. This
  includes trips made from work, such as `home → work → shop → work → home`.
* **Activity counts:** days with more than four discretionary activities are
  capped at four for step 3.

## Calibration

None. Every component is estimated once, and the generated schedules are
whatever the chain produces. The source-aligned variants each add a calibration
stage; their `--no-calibrate` ablation shows how much it matters.

## Deviations from the reference models

* **No locations, modes or travel times.** Activity type and duration only,
  matching ActVAE / Caveat. Moving between activities takes no time.
* **One mandatory activity per day.** Several work or education episodes are
  merged into one block, so a second commute in the day cannot be represented.
* **No work-based trips.** Activities made from work become inbound stops.
* **A coarse day pattern.** The six day types replace the much richer pattern
  choices of DaySim and CEMDAP.
* **Durations and timing are drawn separately.** Durations come from steps 2
  and 5; start times come from a separate density estimate. Nothing ties a late
  start to a short day except the final shrinking.
* **Shrinking instead of feasibility.** Overflowing days are rescaled
  proportionally, never rejected. This always produces a valid schedule but
  distorts durations on busy days.

## Ablations

There are no ablation switches on this model. It serves as the reference point
for two ablations of the source-aligned variants:

* `--independent-timing` swaps their joint timing models for this model's
  approach: a start time from a kernel density estimate and a duration from an
  independent log-normal regression.
* `--consistency rescale` swaps their feasibility rules for proportional
  shrinking.

## Outputs

* `models/composhed_models.pkl`: the fitted models and encoders.
* The generated schedule and attribute CSVs. No generation report is written.
