# DaySim-aligned variant

**Code:** `composhed/daysim/`, which builds on `composhed/common/` (see [what the source-aligned variants share](source_aligned.md))

DaySim (Bowman & Ben-Akiva, 2001) plans a day around tours, meaning trips that
leave home and come back. This variant follows the same steps:

1. **Day pattern.** One choice fixes how many work, education and other tours
   the person makes, and whether the work / education tours have stops.
2. **Tours in priority order.** Work tours first, then education, then the
   others.
3. **Timing.** Each tour chooses its departure and return half-hour together,
   from the times that earlier tours have left free.
4. **Stops.** Each half of a tour (outbound and return) gets its own stops,
   stop types and stop durations, all fitted inside the tour's time.

Any time not spent on a tour is home.

## Running it

```bash
uv run composhed-train-daysim --attributes A.csv --schedules S.csv --output-dir models/ --seed 0
uv run composhed-generate-daysim --attributes A.csv --models models/daysim_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

Options beyond the [shared ones](source_aligned.md#running-them):

| Option | Default | Effect |
|---|---|---|
| `--time-resolution {15,30,60}` | 30 | Length of a time period in minutes |
| `--min-support` | 30 | Day patterns seen fewer times than this are merged into the nearest common pattern |

## Inputs

All eight person attributes, one-hot encoded with the first category dropped.

## Components

"Fidelity" says how closely each component follows DaySim:

* **full:** same structure and model form;
* **simplified:** same idea, reduced form;
* **absent:** not modelled.

| DaySim component | What it decides | Model | Code | Fidelity | Notes |
|---|---|---|---|---|---|
| Day activity pattern | Number of work tours (≤ 2), education tours (≤ 2) and other tours by purpose (≤ 4 in total), and whether the work / education tours have stops | Multinomial logit over the patterns seen at least 30 times (62 on NTS). Rarer patterns map to the nearest common one. | `models.DaysimModels.pattern` | simplified | DaySim splits this into separate pattern and exact-tour-count models; here they are one choice. |
| Tour priority | Order in which tours are scheduled | Fixed: work → education → escort → medical → shop → visit → other | `data.priority_order` | full | |
| Tour time of day | Departure and return period of each tour, chosen together (1,176 pairs at 30 min) | Conditional logit with constants for start period, end period and duration band, plus timing terms interacted with attributes, purpose, tour rank and tour count. Separate models for work, education and other tours. | `common.tod.TourTOD` | simplified | DaySim uses period-specific shift terms and time-window variables; here constants plus attribute interactions. |
| Time windows | Which pairs are still possible | Time taken by earlier tours, plus a 10-minute buffer, is removed from the choice set | `common.timewindow.TimeWindow` | full | |
| Stop generation | Number of stops on each half-tour (0, 1, 2, 3+) | Multinomial logit per direction | `.nstops[direction]` | simplified | DaySim generates stops per purpose. |
| Stop type | escort, medical, other, shop, visit | Multinomial logit per direction, using stop number and tour purpose | `.stop_purpose[direction]` | full | |
| Stop duration | Minutes per stop | Log-normal regression per type (compositional style), bounded by the tour. The main activity gets whatever is left. | `.stop_duration[type]` | simplified | DaySim schedules stops with half-tour period choices. |
| Home | Time between tours | Gaps are home; neighbouring episodes of the same type are merged (counted) | `common.schedule.from_episodes` | full | |
| Infeasible tours | | No free pair: tour dropped. A draw that can't be realised: redrawn up to 50 times. All counted. | `common.tourgen.place_tours` | full | |
| Location, mode, travel time | | — | — | absent | Not in the data. |

**Exact minutes within a period:** a tour that leaves "between 08:00 and
08:30" gets its exact minute from the observed distribution of minutes in that
period, limited to the feasible window. Sparse periods fall back to the
group's pooled pattern, then to uniform. This is done by
`common.periods.IntraPeriodSampler`.

**Estimating timing the DaySim way:** each observed tour is estimated against
only the times its day's higher-priority tours left free, buffer included. An
observed choice that lands inside a buffer is kept available and counted
(`forced_available` in `validation.json`).

## Calibration

[Shared method](source_aligned.md#calibration).

**What is adjusted:**
* the day-pattern constants (to match pattern shares);
* for each timing group, the start-period, end-period and duration-band
  constants (to match their shares);
* the stops-per-half-tour constants.

**Notes:**
* The three timing constants act on the same choice, so their steps are scaled
  by 1/3. Without that they overshoot and the calibration diverges.
* Duration bands are measured on the chosen period pair, because that is what
  the constants act on.

## Deviations from DaySim

* **No locations, modes or travel times**, and a 10-minute buffer at home
  between tours.
* **Coefficients are re-estimated on NTS.**
* **One person-day**: no household or joint tours.
* **Capped choice sets:** rare patterns are merged; at most 3 stops per
  half-tour, 4 other tours, and 2 tours of each mandatory type.
* **Multiple work / education tours are kept**, unlike the compositional
  baseline and the CEMDAP variant.
* **No tours from work.** Work interrupted by other activities is merged into
  one block (see [tour_rules.md](tour_rules.md)).
* **The main activity's duration is never drawn.** It is whatever time the
  tour has after its stops.
* **Unplaceable tours are dropped (counted)**, never shrunk, except under the
  `rescale` ablation.

## Ablations

| Switch | Effect |
|---|---|
| `--no-calibrate` | Estimation only. |
| `--independent-timing` | Start from a kernel density estimate and duration from a log-normal regression per timing group, redrawn until the tour fits. |
| `--consistency rescale` | No time windows: overlaps are fixed afterwards by shifting and proportional shrinking (counted). |
| `--time-resolution 15 / 30 / 60` | Period length. |

## Outputs

The [shared outputs](source_aligned.md#outputs): `models/daysim_models.pkl`,
`validation.json` and `generation_report.json`.
