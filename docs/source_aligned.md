# Source-aligned variants: what they share

The [CEMDAP](cemdap.md), [DaySim](daysim.md) and [ActivitySim](activitysim.md)
variants each copy the *structure* of a published activity-based model: which
decisions are made, in what order, and with what kind of model. They are not
the frameworks themselves. All three:

* use only what is in the NTS data: no locations, modes, networks or travel
  times. Moving between activities takes no time. Between two tours CEMDAP
  keeps at least 10 minutes at home (`HOME_INTER`, borrowed from the
  [compositional baseline](compositional.md)); DaySim and ActivitySim keep
  1 minute, the least a schedule with a home episode between the tours can
  show;
* model one person's day, with no household interactions or joint trips;
* re-estimate every coefficient on NTS 2023, with no calibration. No
  published coefficients are reused. DaySim's and ActivitySim's utilities are
  their sources' own terms (DaySim's C# model code; ActivitySim's
  `prototype_mtc` spec files, read directly), keeping the terms whose
  variables NTS has and adding only day of week
  ([DaySim and ActivitySim compared with their sources](source_differences.md));
* turn chosen time periods into minutes by a uniform draw over the free minutes
  of the period (DaySim's `GetAvailableMinute`; ActivitySim stops at periods),
  never from NTS's observed minutes;
* read their training data through one shared set of tour rules
  ([tour_rules.md](tour_rules.md));
* share their code in `composhed/common/` (choice models, duration models,
  spec-file logits, time windows, reports). Nothing in `composhed/common/` is used by
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
| `--independent-timing` | off | Ablation: replace the joint timing model with compositional-style timing. |
| `--consistency {constraint,rescale}` | constraint | Ablation: make days fit by construction, or by shrinking afterwards. |
| `--max-resample` | 50 | How many times a draw that doesn't fit is redrawn before giving up. |
| `--holdout-fraction`, `--split-seed` | 0, 0 | Hold out a share of people and report component fit on them in `validation.json`. |

Each variant adds its own options; see its page.

## Feasibility and the generation report

Each variant makes days fit in its own way: truncated draws (CEMDAP) or time
windows that remove impossible choices (DaySim, ActivitySim). When that still
fails:

* a draw is repeated (up to `--max-resample`); DaySim, as its source,
  simulates the whole person-day again;
* if that fails too, the tour or stop is dropped;
* as a last resort, the person gets an all-home day.

Every one of these events is counted in `generation_report.json`, which is
written next to the generated schedules. On NTS 2023, no person needed the
all-home fallback in any variant.

Every generated day is checked before it is written:

* it covers exactly 1,440 minutes;
* it starts and ends at home;
* every duration is positive.

## Reproducibility

All sampling draws from one random generator seeded by `--seed`. Generating
twice with the same seed gives byte-identical files. A model bundle records its
variant and version, and loading it with the wrong generator fails with a clear
error.

## Outputs

| File | Contents |
|---|---|
| `models/<variant>_models.pkl` | Fitted models, coefficient tables and configuration |
| `models/validation.json` | Fit, parameter count and coefficients of each component, training-data extraction counts, held-out fit if requested |
| `generation_report.json` | Counts of redraws, dropped tours and stops, shrunk days and fallbacks |
