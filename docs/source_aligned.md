# Source-aligned variants: what they share

The [CEMDAP](cemdap.md), [DaySim](daysim.md) and [ActivitySim](activitysim.md)
variants each copy the *structure* of a published activity-based model: which
decisions are made, in what order, and with what kind of model. They are not
the frameworks themselves. All three:

* use only what is in the NTS data: no locations, modes, networks or travel
  times. Moving between activities takes no time, and no variant keeps a
  minimum home stay between tours: CEMDAP draws home stays from its own
  duration model, and DaySim's and ActivitySim's time windows let a tour start
  when the previous one ends. A zero-length home stay is still written between
  the two tours;
* model one person's day, with no household interactions or joint trips;
* re-estimate every coefficient on NTS 2023, with no calibration. No
  published coefficients are reused. DaySim's and ActivitySim's utilities are
  their sources' own terms (DaySim's C# model code; ActivitySim's
  `prototype_mtc` spec files, read directly), keeping the terms whose
  variables NTS has and adding only day of week
  ([DaySim and ActivitySim compared with their sources](source_differences.md));
* keep their source's time precision, never using NTS's observed minutes:
  * **CEMDAP:** continuous time; its hazard draws are rounded to whole minutes
    only when the day is written;
  * **DaySim:** chooses half-hour periods, then a minute drawn uniformly over
    the free minutes of the period, as DaySim's `GetAvailableMinute` does;
  * **ActivitySim:** hourly periods (`prototype_mtc`), and nothing finer. Each
    period is written at its midpoint, so two events in one period (a stop's
    arrival and departure, or the end of one tour and the start of the next)
    are zero minutes apart;
* keep every episode they generate: zero-length episodes and neighbouring
  episodes of one type are written as they are;
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
* if that fails too, the tour or stop is dropped (DaySim and ActivitySim; with
  ActivitySim's zero-length stops, no stop needs dropping);
* as a last resort, the person gets an all-home day.

Every one of these events is counted in `generation_report.json`, which is
written next to the generated schedules. On NTS 2023, no person needed the
all-home fallback in any variant.

Every generated day is checked before it is written:

* it covers exactly 1,440 minutes;
* it starts and ends at home;
* every duration is zero or more.

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
