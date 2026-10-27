# MDCEV variant

**Code:** `composhed/models/mdcev.py`, `episode_count.py`, `timing.py`, `composhed/assembly.py`, `composhed/train_mdcev.py`, `composhed/generate_mdcev.py`

The MDCEV variant replaces steps 1–5 of the [compositional baseline](compositional.md)
with a single joint model of how a person splits their 1,440 minutes across the
eight activity types. The model is a Multiple Discrete-Continuous Extreme Value
(MDCEV) model (Bhat, 2005; 2008). In one draw it decides both *which* activities
happen (any type with more than zero minutes) and *how long* each one takes in
total.

Two small models then turn those totals into a sequence:

* the number of separate episodes of each type;
* when each episode tends to start.

A constrained optimisation lays the episodes out so that their start times are
as close as possible to the predicted ones. There is no explicit tour structure.

Compared with the compositional chain, participation and duration are tied
together. A day with a lot of work leaves less time for everything else,
because all the time comes from the same budget. The price is a heavy
estimation step (Biogeme) and coefficients that are harder to read step by step.

## Running it

```bash
uv run composhed-train-mdcev --attributes A.csv --schedules S.csv --output-dir models/ \
    --seed 0 --max-records 5000
uv run composhed-generate-mdcev --attributes A.csv --models models/mdcev_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

| Option | Default | Effect |
|---|---|---|
| `--max-records` | all | Estimate the MDCEV model on a random sample of this many people. **Needed on the full NTS data**: estimating on all 56,734 people runs out of memory (Biogeme tries to allocate about 170–185 GB for the Hessian). The stability study uses 5,000. Only the MDCEV model is subsampled; the episode and timing models use everyone. |
| `--fix-gamma / --free-gamma` | fixed | Fix each type's satiation parameter γ to its mean duration when done, or estimate γ freely. Free estimation tends to find degenerate solutions on this data. |
| `--calibrate / --no-calibrate` | on | Adjust the constants after estimation to match participation rates (see below). |
| `--cap-percentile` | 90 | How many episodes of each type the timing model enumerates, as a percentile of the observed counts. |

## Inputs

Five person attributes: `age`, `hh_income`, `sex`, `employment` and `day`,
one-hot encoded with the first category dropped. The spatial and car-access
attributes (`hh_zone`, `access_egress_distance`, `vehicles`) are left out. They
matter more for location and mode choice than for how time is split.

## Components

| Step | What it decides | Model | Inputs | Code |
|---|---|---|---|---|
| 1. Time allocation | Total minutes for each of the 8 types; summed to 1,440; zero means the person doesn't do it | MDCEV with a gamma utility profile, estimated in Biogeme. Forecasts use Gumbel draws and the closed-form solution of Pinjari & Bhat, vectorised in numpy / PyTorch. | the five attributes | `models/mdcev.py` |
| 2. Episode count | How many separate episodes the total for each type is split into (1–6) | Ordered logit per activity type on log(total minutes) (statsmodels). Types with fewer than 30 observations use the most common count. | total minutes for that type | `models/episode_count.py` |
| 3. Start times | Predicted start of each numbered episode (`work_0`, `shop_0`, `shop_1`, ...) | Linear regression per numbered episode, plus residual noise at generation | the five attributes, number of tours | `models/timing.py` |
| 4. Assembly | Final sequence | Episodes are ordered by predicted start, with the first home episode fixed at midnight and the last one filling the rest of the day. SLSQP then finds the durations that bring scheduled starts closest to the predicted ones, keeping each type's total fixed and every episode at least 1 min long. Results are rounded to whole minutes, preserving the totals. | outputs of steps 1–3 | `assembly.py` (`assemble_eased_mdcev_schedule`) |

There is no day-type step. The kind of day follows from which types receive
time.

## Calibration

After estimation, the activity-type constants are nudged until the simulated
share of people doing each activity matches the training data:

* Each pass simulates 100 draws for a random sample of 5,000 people.
* Each constant moves by `0.1 × (log target − log simulated)`, capped at ±0.3
  per pass.
* Passes stop once every share is within 0.005, or after 50 passes.
* Types done by almost everyone (target ≥ 0.99, i.e. home) are left alone:
  pushing their constants destabilises the shared budget.
* The offsets are stored in the model bundle.

**Caveat:** calibration draws its sample from an unseeded random generator, so
two calibrated runs with the same `--seed` give different models. Uncalibrated
runs are reproducible. That is why the regression test
(`tests/test_regression_existing.py`) trains MDCEV with `calibrate=False`.

## Deviations from the reference model

* **Satiation is fixed rather than estimated** (by default). Each type's γ is
  set to its mean duration when done, leaving the estimation to fit the
  constants and attribute effects.
* **Estimated on a subsample.** On the full data, estimation has to run on a
  sample (`--max-records`) to fit in memory.
* **No locations, modes or travel times**, as in every CompoShed variant.
* **Sequencing is not part of MDCEV.** Episode counts, start times and the
  layout optimisation are extra models layered on top of the allocation. They
  are not estimated jointly with it.
* **No tour structure.** The layout only tries to match predicted start times.
  Nothing makes activities form trips from home, and nothing guarantees a
  minimum stay at home between them.

## Ablations

| Switch | Effect |
|---|---|
| `--no-calibrate` | Estimation only. Also available in the stability harness as `mdcev:no-calibrate`. |
| `--free-gamma` | Estimate the satiation parameters too. |
| `--max-records N` | Size of the estimation sample. |
| `--cap-percentile P` | How many numbered episodes the timing model covers. |

> The stability harness (`run_stability.py`) currently trains MDCEV with
> `--free-gamma`, not the documented default of fixed γ.

## Outputs

* `models/mdcev_models.pkl`: the MDCEV, episode-count and start-time models,
  plus the feature names.
* The generated schedule and attribute CSVs. No generation report is written.
