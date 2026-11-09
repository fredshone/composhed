# CEMDAP-aligned variant

**Code:** `composhed/cemdap/`, which builds on `composhed/common/` (see [what the source-aligned variants share](source_aligned.md))

CEMDAP (Bhat et al., 2004) builds a person's day in continuous time, one
decision after another. This variant follows the same order:

1. Does the person go to work or school today?
2. If so, how long is their work-based period (from the start of work to its
   end), and when does it start, given that duration? As in CEMDAP, the
   duration is drawn first.
3. How many stops do they make on the way there and on the way back, how many
   separate trips from home before and after, and how many work-based tours
   (trips out from work and back)?
4. What type is each stop, and how long does it last?
5. How long do they stay at home between trips, and at work before each
   work-based tour?

People with no work or school get a separate chain: how many trips from home,
when they first leave, and how many stops on each trip.

Start times and durations come from hazard-based duration models
(accelerated failure time, AFT). Time is continuous, measured from the start of
the day, and nothing has a minimum duration. CEMDAP checks each predicted
duration against the time still available; here each draw is limited to the
time still free in its period, so the day cannot overflow.

## Running it

```bash
uv run composhed-train-cemdap --attributes A.csv --schedules S.csv --output-dir models/ --seed 0
uv run composhed-generate-cemdap --attributes A.csv --models models/cemdap_models.pkl \
    --out-attributes syn_a.csv --out-schedules syn_s.csv --seed 0
```

This variant takes only the [shared options](source_aligned.md#running-them).

## Inputs

All eight person attributes, one-hot encoded with the first category dropped,
as in the [compositional baseline](compositional.md).

## Components

"Fidelity" says how closely each component follows CEMDAP:

* **full:** same structure and model form;
* **simplified:** same idea, reduced form;
* **absent:** not modelled.

| CEMDAP component | What it decides | Model | Code | Fidelity | Notes |
|---|---|---|---|---|---|
| Work / school participation | Whether the day has work, or (failing that) education | Two binary logits | `models.CemdapModels.work`, `.edu` | simplified | No household context. A tour with both work and education is a work tour (12 NTS tours). |
| Work-based duration | From the start of work / school to its end, work-based tours included | AFT; log-logistic or Weibull, chosen by AIC | `.duration` | full | |
| Work / school start | Minutes since the start of the day, given the duration | AFT with duration and duration² as covariates | `.start` | full | CEMDAP's day starts at 03:00; the NTS day at 00:00. |
| Stops and tours by period | Number (0–3) of outbound (commute) stops, inbound stops, trips before work, trips after work and work-based tours | Five ordered logits | `.counts` | simplified | CEMDAP models the commute stops and the before-work, work-based and after-work tours in this way; here the counts are separate ordered logits. |
| Work-based tours | Trips out from work and back, inside the work-based period | Stops per tour, stop type and duration as for other tours; work stay before each tour from the stay model | `chain.CemdapChain` | full | 228 NTS tours. |
| Non-worker pattern | Number of trips from home (0–4) and stops per trip (1–4) | Ordered logits | `.nm_tours`, `.tour_size` | simplified | Trips are plain lists of stops, with no main activity. |
| Stop type | escort, leisure, other, shop or visit | Multinomial logit on position, stop number and remaining time | `.stop_purpose` | full | |
| Stop duration | Minutes per stop | AFT per stop type, drawn within the remaining time | `.stop_duration[type]` | full | |
| Home and work stays | Time at home between trips, time at work before a work-based tour, and first departure on a day with no work or school | AFT with the kind of stay as a covariate. The first home stay of a work / school day, the last work stay and the last home stay of every day take whatever time is left. | `.home`, `.first_departure` | simplified | One model for home and work stays. |
| Keeping the day feasible | | Every draw bounded by the time left in its period | `chain.CemdapChain` | full | |
| Location, mode, travel time | | — | — | absent | Not in the data. |

### How a day is built

**Work / school day:**

1. Draw the work-based duration *D*, then the start *S* given *D*, bounded so
   the period fits in the day.
2. Draw the stop and tour counts.
3. Fill the morning (*S* minutes available): outbound stops, then each earlier
   trip with its stops followed by a home stay. The first home stay of the day
   takes what's left.
4. Fill the evening (1440 − *S* − *D* minutes available) the same way:
   inbound stops, then each later trip preceded by a home stay. The last home
   stay takes what's left.
5. Fill the work-based period (*D* minutes): each work-based tour preceded by
   a work stay. The last work stay takes what's left.

**Day without work or school:** draw the number of trips, then the first
departure, then the trips and the home stays between them. The last home stay
takes what's left.

**Bounds on every draw:** each draw is bounded by the time left in its period,
and by nothing else. Any stay or stop can therefore be zero minutes long when
the time has run out. The remaining-time covariates used in training are
computed in the same order.

## Deviations from CEMDAP

* **No locations, modes or travel times.** Moving takes no time.
* **Coefficients are re-estimated on NTS**; no CEMDAP coefficients are used.
* **One person-day, not a household day**: no joint activities.
* **One work-based period per day.** Later work or school trips from home are
  folded into it (see [tour_rules.md](tour_rules.md)); CEMDAP's worker
  pattern has one commute each way.
* **No tour durations.** CEMDAP draws a tour's duration and then its stops
  within it; here a tour lasts as long as its stops.
* **Capped choice sets:** at most 3 stops or tours per position, 4 trips on a
  day without work or school, and 4 stops per trip. Larger counts are logged.
* **The day starts at 00:00**, the start of the NTS day, not at 03:00.
* **Own estimation code.** The logits and ordered logits use the repository's
  own L-BFGS code (`common/choice.py`, same likelihood as statsmodels but much
  faster). The AFT models use custom maximum likelihood (`common/hazard.py`).

## Ablations

| Switch | Effect |
|---|---|
| `--independent-timing` | Work / school start from a kernel density estimate per employment category, and duration from a log-normal regression that ignores the start time (compositional style). |
| `--consistency rescale` | Durations are drawn without bounds; a morning, evening or work-based period that overflows is shrunk proportionally (counted). |

## Outputs

The [shared outputs](source_aligned.md#outputs) (`models/cemdap_models.pkl`,
`validation.json`, `generation_report.json`).
