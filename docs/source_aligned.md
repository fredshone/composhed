# Source-aligned variants: what they share

The [CEMDAP](cemdap.md), [DaySim](daysim.md) and [ActivitySim](activitysim.md)
variants each copy the *structure* of a published activity-based model: which
decisions are made, in what order, and with what kind of model. They are not
the frameworks themselves. All three:

* use only what is in the NTS data: no locations, modes, networks or travel
  times. Moving between activities takes no time, but a trip from home is
  always followed by at least 10 minutes at home (`HOME_INTER`, borrowed from
  the [compositional baseline](compositional.md));
* model one person's day, with no household interactions or joint trips;
* re-estimate every coefficient on NTS 2023. No published coefficients are
  reused;
* read their training data through one shared set of tour rules
  ([tour_rules.md](tour_rules.md));
* share their code in `composhed/common/` (choice models, duration models,
  time windows, calibration, reports). Nothing in `composhed/common/` is used by
  the compositional or MDCEV pipelines, whose outputs are pinned by
  `tests/test_regression_existing.py`.

## Running them

```bash
uv run composhed-train-<variant> --attributes A.csv --schedules S.csv --output-dir models/ --seed 0
uv run composhed-generate-<variant> --attributes A.csv --models models/<variant>_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

where `<variant>` is `cemdap`, `daysim` or `activitysim`. Training options that
all three accept:

| Option | Default | Effect |
|---|---|---|
| `--calibrate / --no-calibrate` | on | Run the calibration stage (below). |
| `--calib-iters` | 8 | Maximum calibration passes. |
| `--calib-sample` | 20,000 | Training people simulated in each pass. |
| `--independent-timing` | off | Ablation: replace the joint timing model with compositional-style timing. |
| `--consistency {constraint,rescale}` | constraint | Ablation: make days fit by construction, or by shrinking afterwards. |
| `--max-resample` | 50 | How many times a draw that doesn't fit is redrawn before giving up. |
| `--holdout-fraction`, `--split-seed` | 0, 0 | Hold out a share of people and report component fit on them in `validation.json`. |

Each variant adds its own options; see its page.

## Calibration

Estimated models rarely reproduce the training data exactly once they are
chained together, so all three variants calibrate the way practitioners
calibrate the real frameworks. Each pass:

1. Simulates the whole chain for a fixed sample of training people. The same
   random numbers are reused in every pass, so differences between passes come
   from the adjustments, not from noise.
2. Compares simulated shares and means against the training data.
3. Moves each constant by a damped log-ratio step towards its target.

Calibration stops when every target is within tolerance, or when it runs out
of passes. The best pass is kept, not the last.

* **Shares:** within max(0.01, 2 simulation standard errors) of the target.
  Small groups, such as about 120 simulated university students, therefore get
  a wider tolerance than large ones.
* **Means and spreads:** within 2% of the target.

Each variant's page lists what it calibrates. The full report, with each
target's history and tolerance, is in `validation.json` under
`info.calibration`. On NTS 2023 every target converges for all three variants.

## Feasibility and the generation report

Each variant makes days fit in its own way: truncated draws (CEMDAP) or time
windows that remove impossible choices (DaySim, ActivitySim). When that still
fails:

* the person is redrawn (up to `--max-resample`);
* if that fails too, an extra tour or stop is dropped;
* as a last resort, the person gets an all-home day.

Every one of these events is counted in `generation_report.json`, which is
written next to the generated schedules. On NTS 2023, no person needed the
all-home fallback in any variant.

Every generated day is checked before it is written:

* it covers exactly 1,440 minutes;
* it starts and ends at home;
* no two neighbouring activities have the same type;
* every duration is positive.

## Reproducibility

All sampling draws from one random generator seeded by `--seed`. Generating
twice with the same seed gives byte-identical files. A model bundle records its
variant and version, and loading it with the wrong generator fails with a clear
error.

## Outputs

| File | Contents |
|---|---|
| `models/<variant>_models.pkl` | Fitted models, coefficient tables, configuration, calibration targets and report |
| `models/validation.json` | Fit of each component, training-data extraction counts, calibration report, held-out fit if requested |
| `generation_report.json` | Counts of redraws, dropped tours and stops, shrunk days, merges and fallbacks |
