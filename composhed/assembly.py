"""Step 6 — Rule-based 24-hour schedule assembly."""

import numpy as np

_MDCEV_DISC_TYPES = ["escort", "medical", "other", "shop", "visit"]

# Minimum home time between separate home-based tours.
# This is a modelling assumption: represents the minimum time needed to
# return home and depart again between two distinct home-based tours.
HOME_INTER = 10


def assemble_schedule(
    dap: str,
    mandatory_duration: float,
    mandatory_type: str,
    disc_activities: list[tuple[str, int]],
    work_start: float | None,
    first_departure: float | None,
    tour_placements: list[str],
    tour_groupings: list[bool] | None = None,
) -> list[dict]:
    """Assemble a valid 24-hour schedule using tour-based structure.

    tour_placements: one label per disc activity (WD/ED):
        "mandatory_outbound" — stop before mandatory in same tour
        "mandatory_inbound"  — stop after mandatory in same tour
        "separate_before"    — own home-based tour before the mandatory tour
        "separate_after"     — own home-based tour after the mandatory tour
    tour_groupings: one bool per consecutive disc pair (D schedules only).
        True = next activity is on the same home-based tour.

    Returns list of dicts: [{act, start, end, duration}, ...]
    Sum of durations == 1440, starts and ends with 'home'.
    """
    md = int(round(mandatory_duration))
    disc = [(a, int(round(d))) for a, d in disc_activities]

    # ---- H: home all day ------------------------------------------------
    if dap == "H":
        return _to_rows([("home", 1440)])

    # ---- Budget enforcement: ensure ≥60 min home ------------------------
    disc = _enforce_budget(md if dap in ("W", "WD", "E", "ED") else 0, disc)

    # ---- W/E: home → mandatory → home -----------------------------------
    if dap in ("W", "E"):
        ws = int(np.clip(round(work_start or 480), 0, 1440 - md - 60))
        home_eve = 1440 - ws - md
        if home_eve < 30:
            ws = max(0, 1440 - md - 30)
            home_eve = 1440 - ws - md
        seq = [("home", ws), (mandatory_type, md), ("home", home_eve)]
        return _to_rows(seq)

    # ---- WD/ED: tour-based assembly -------------------------------------
    if dap in ("WD", "ED"):
        ws = int(np.clip(round(work_start or 480), 30, 1440 - md - 30))
        outbound, inbound, sep_pre, sep_post = _partition_by_placement(
            disc, tour_placements
        )

        outbound_total = sum(d for _, d in outbound)
        inbound_total = sum(d for _, d in inbound)
        sep_pre_total = sum(d for _, d in sep_pre)
        sep_post_total = sum(d for _, d in sep_post)

        # Fit separate_pre: it occupies [..., sep_pre, home(HOME_INTER), outbound, mandatory, ...]
        if sep_pre:
            avail_pre = ws - outbound_total - HOME_INTER - 30
            if sep_pre_total > avail_pre:
                if avail_pre > 0:
                    sep_pre = _scale_activities(sep_pre, avail_pre)
                    sep_pre_total = sum(d for _, d in sep_pre)
                else:
                    sep_post = sep_pre + sep_post
                    sep_pre = []
                    sep_pre_total = 0

        # Fit separate_post
        work_end = ws + md
        if sep_post:
            avail_post = 1440 - work_end - inbound_total - HOME_INTER - 30
            sep_post_total = sum(d for _, d in sep_post)
            if sep_post_total > avail_post:
                if avail_post > 0:
                    sep_post = _scale_activities(sep_post, avail_post)
                    sep_post_total = sum(d for _, d in sep_post)
                else:
                    sep_post = []
                    sep_post_total = 0

        # Fit outbound stops within available pre-work slot
        avail_out = ws - (sep_pre_total + HOME_INTER if sep_pre else 0) - 30
        if outbound_total > avail_out:
            if avail_out > 0:
                outbound = _scale_activities(outbound, avail_out)
                outbound_total = sum(d for _, d in outbound)
            else:
                inbound = outbound + inbound
                outbound = []
                outbound_total = 0

        # Fit inbound stops within available post-work slot
        avail_in = 1440 - work_end - (HOME_INTER + sep_post_total if sep_post else 0) - 30
        inbound_total = sum(d for _, d in inbound)
        if inbound_total > avail_in:
            if avail_in > 0:
                inbound = _scale_activities(inbound, avail_in)
                inbound_total = sum(d for _, d in inbound)
            else:
                inbound = []
                inbound_total = 0

        sep_pre_total = sum(d for _, d in sep_pre)
        sep_post_total = sum(d for _, d in sep_post)

        # Compute home times
        if sep_pre:
            home_morn = ws - outbound_total - HOME_INTER - sep_pre_total
        else:
            home_morn = ws - outbound_total

        if sep_post:
            home_eve = 1440 - work_end - inbound_total - HOME_INTER - sep_post_total
        else:
            home_eve = 1440 - work_end - inbound_total

        seq = [("home", max(0, home_morn))]
        if sep_pre:
            seq += sep_pre + [("home", HOME_INTER)]
        seq += outbound + [(mandatory_type, md)] + inbound
        if sep_post:
            seq += [("home", HOME_INTER)] + sep_post
        seq += [("home", max(0, home_eve))]

        return _to_rows(seq)

    # ---- D: tour-based assembly -----------------------------------------
    if dap == "D":
        if not disc:
            return _to_rows([("home", 1440)])

        fd = int(np.clip(round(first_departure or 480), 0, 1380))

        # Group into home-based tours using grouping flags
        tours = _group_into_tours(disc, tour_groupings or [])
        n_inter = max(0, len(tours) - 1)
        disc_total = sum(d for t in tours for _, d in t)

        home_eve = 1440 - fd - disc_total - n_inter * HOME_INTER
        if home_eve < 30:
            avail = 1440 - fd - n_inter * HOME_INTER - 30
            if avail > 0 and disc_total > 0:
                # Scale all disc activities proportionally then rebuild tours
                all_disc = _scale_activities(
                    [(a, d) for t in tours for a, d in t], avail
                )
                tours = [all_disc]
                n_inter = 0
                disc_total = sum(d for _, d in all_disc)
            home_eve = 1440 - fd - disc_total - n_inter * HOME_INTER

        seq: list[tuple[str, int]] = [("home", fd)]
        for i, tour in enumerate(tours):
            seq += tour
            if i < len(tours) - 1:
                seq += [("home", HOME_INTER)]
        seq += [("home", max(0, home_eve))]

        return _to_rows(seq)

    raise ValueError(f"Unknown DAP: {dap}")


def assemble_mdcev_schedule(
    alloc: dict[str, float],
    dap: str,
    mandatory_type: str,
    x_label: np.ndarray,
    employment: str,
    anchor_model,
    episode_model,
) -> list[dict]:
    """Assemble a 1440-min schedule from MDCEV allocations using tour-based structure.

    Durations for non-home activities are taken verbatim from alloc (rounded).
    Home time fills the remainder. Sum is guaranteed to equal 1440.
    """
    if dap == "H":
        return _to_rows([("home", 1440)])

    # Round non-home durations; home gets the exact remainder so sum == 1440
    mandatory = int(round(alloc.get(mandatory_type, 0.0)))
    disc_durs: dict[str, int] = {
        atype: int(round(alloc.get(atype, 0.0)))
        for atype in _MDCEV_DISC_TYPES
        if alloc.get(atype, 0.0) >= 10.0
    }
    # If the other mandatory type also has a significant allocation, treat it as a
    # discretionary activity so both appear in the schedule via the tour-placement model.
    _secondary = "education" if mandatory_type == "work" else "work"
    if alloc.get(_secondary, 0.0) >= 10.0:
        disc_durs[_secondary] = int(round(alloc[_secondary]))
    # For D schedules, the mandatory_type allocation didn't meet the mandatory threshold
    # but may still be non-trivial; include it as a discretionary episode rather than
    # silently dropping it.
    if dap == "D" and mandatory >= 10:
        disc_durs[mandatory_type] = mandatory
    home_total = 1440 - mandatory - sum(disc_durs.values())
    home_total = max(0, home_total)

    # Enforce minimum home time so every schedule has a valid home frame.
    _MIN_HOME = 60
    if home_total < _MIN_HOME:
        non_home = mandatory + sum(disc_durs.values())
        if non_home > 0:
            scale = max(0.0, (1440 - _MIN_HOME) / non_home)
            mandatory = max(0, int(round(mandatory * scale)))
            disc_durs = {k: max(0, int(round(v * scale))) for k, v in disc_durs.items()}
        home_total = 1440 - mandatory - sum(disc_durs.values())
        home_total = max(_MIN_HOME, home_total)

    # Build flat episode list for disc activities
    disc_episodes: list[tuple[str, int]] = []
    for atype, dur in disc_durs.items():
        disc_episodes.extend(_split_duration(atype, dur, episode_model))

    # ---- W / E: home → mandatory → home ------------------------------------
    if dap in ("W", "E"):
        upper = max(1, home_total - 30)
        ws = int(np.clip(round(anchor_model.sample_work_start(employment)), 1, upper))
        seq = [("home", ws), (mandatory_type, mandatory), ("home", home_total - ws)]
        return _to_rows(seq)

    # ---- WD / ED: tour-based assembly --------------------------------------
    if dap in ("WD", "ED"):
        ws_raw = int(np.clip(
            round(anchor_model.sample_work_start(employment)),
            30, max(30, home_total - 30),
        ))
        placements = anchor_model.sample_tour_placements(
            x_label, disc_episodes, float(ws_raw)
        )
        outbound, inbound, sep_pre, sep_post = _partition_by_placement(
            disc_episodes, placements
        )

        outbound_total = sum(d for _, d in outbound)
        sep_pre_total = sum(d for _, d in sep_pre)
        n_inter = (1 if sep_pre else 0) + (1 if sep_post else 0)

        # Effective "load" before work_start (used to clip ws)
        pre_load = outbound_total + (HOME_INTER + sep_pre_total if sep_pre else 0)

        # Clip ws so home_morn ≥ 30 and home_eve ≥ 30
        ws_lo = pre_load + 30
        ws_hi = pre_load + home_total - n_inter * HOME_INTER - 30
        if ws_lo > ws_hi:
            ws = pre_load + max(0, home_total - n_inter * HOME_INTER) // 2
        else:
            ws = int(np.clip(ws_raw, ws_lo, ws_hi))

        # If pre_load no longer fits, fold sep_pre into sep_post
        if ws - pre_load < 30:
            sep_post = sep_pre + sep_post
            sep_pre = []
            sep_pre_total = 0
            pre_load = outbound_total

        home_morn = ws - pre_load
        n_inter = (1 if sep_pre else 0) + (1 if sep_post else 0)
        home_eve = home_total - n_inter * HOME_INTER - home_morn

        work_end = ws + mandatory
        inbound_total = sum(d for _, d in inbound)
        sep_post_total = sum(d for _, d in sep_post)

        # Verify post side fits; if not, fold sep_post into inbound
        post_load = inbound_total + (HOME_INTER + sep_post_total if sep_post else 0)
        if home_eve < 30 and sep_post:
            inbound = inbound + sep_post
            sep_post = []
            n_inter = 1 if sep_pre else 0
            home_eve = home_total - n_inter * HOME_INTER - home_morn

        seq = [("home", max(0, home_morn))]
        if sep_pre:
            seq += sep_pre + [("home", HOME_INTER)]
        seq += outbound + [(mandatory_type, mandatory)] + inbound
        if sep_post:
            seq += [("home", HOME_INTER)] + sep_post
        seq += [("home", max(0, home_eve))]
        return _to_rows(seq)

    # ---- D: tour-based assembly --------------------------------------------
    if dap == "D":
        groupings = anchor_model.sample_tour_groupings(x_label, disc_episodes)
        tours = _group_into_tours(disc_episodes, groupings)
        n_inter = max(0, len(tours) - 1)

        # home_total must cover inter-tour homes + at least 30 min morning + evening
        inter_needed = n_inter * HOME_INTER
        if home_total < inter_needed + 60:
            # Not enough room for separate tours; collapse to one
            tours = [disc_episodes]
            n_inter = 0
            inter_needed = 0

        upper = max(1, home_total - inter_needed - 30)
        fd = int(np.clip(
            round(anchor_model.sample_first_departure(employment)),
            1, upper,
        ))

        seq = [("home", fd)]
        for i, tour in enumerate(tours):
            seq += tour
            if i < len(tours) - 1:
                seq += [("home", HOME_INTER)]
        seq += [("home", home_total - inter_needed - fd)]
        return _to_rows(seq)

    raise ValueError(f"Unknown DAP for MDCEV assembly: {dap}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_alloc_to_enumerated(
    alloc: dict[str, float],
    episode_counts: dict[str, int],
) -> dict[str, float]:
    """Convert per-type MDCEV totals into enumerated episode keys (equal split).

    Each type is split equally across its sampled episode count.
    E.g. {"shop": 120, "home": 800} + {"shop": 2, "home": 3}
    → {"shop_0": 60.0, "shop_1": 60.0, "home_0": 267.0, "home_1": 267.0, "home_2": 267.0}

    Types with total <= 0 or n_episodes == 0 are omitted.
    Note: use assemble_greedy_mdcev_schedule for home-aware splitting.
    """
    result: dict[str, float] = {}
    for atype, total in alloc.items():
        if total <= 0:
            continue
        n = max(1, int(episode_counts.get(atype, 1)))
        per = total / n
        for i in range(n):
            result[f"{atype}_{i}"] = per
    return result


def assemble_eased_mdcev_schedule(
    alloc: dict[str, float],
    episode_counts: dict[str, int],
    x_person: np.ndarray,
    timing_model,
    min_dur: float = 1.0,
) -> list[dict]:
    """Assemble a 1440-min schedule via timing-constrained duration optimisation.

    (i)  Build one episode per type per episode_count[type].
    (ii) Predict a start time for every episode (home_0 fixed at 0, all others
         via timing_model keyed as type_N so the model captures 1st/2nd etc.
         participation).
    (iii) Order all episodes by predicted start time, home_0 always first.
    (iv) Find episode durations d that minimise
             sum_i ( scheduled_start_i  -  predicted_start_i )^2
         subject to:
             sum_{i: type_i == t}  d_i  ==  alloc[t]   for every type t
             d_i  >=  min_dur
         where scheduled_start_i = sum(d_j for j < i) (greedy left-to-right).

    Optimisation uses SLSQP (fast for K < ~20 activities).
    Falls back to equal splits per type if the solver fails.
    """
    from scipy.optimize import minimize as _sp_min

    # ---- (i) enumerate episodes ---------------------------------------------
    non_home_keys: list[str] = []
    for atype in sorted(alloc):
        if atype == "home":
            continue
        if alloc[atype] <= 0:
            continue
        n = max(1, int(episode_counts.get(atype, 1)))
        for idx in range(n):
            non_home_keys.append(f"{atype}_{idx}")

    if not non_home_keys:
        return _to_rows([("home", 1440)])

    n_home = max(2, int(episode_counts.get("home", 2)))
    n_tours = max(1, n_home - 1)
    home_keys = [f"home_{idx}" for idx in range(n_home)]

    # Integer per-type totals that sum exactly to 1440
    nh_int: dict[str, int] = {}
    for key in non_home_keys:
        atype = key.rsplit("_", 1)[0]
        if atype not in nh_int:
            nh_int[atype] = int(round(alloc[atype]))
    home_int = max(n_home * int(np.ceil(min_dur)), 1440 - sum(nh_int.values()))
    int_totals: dict[str, int] = {**nh_int, "home": home_int}

    # ---- (ii) predict start times ------------------------------------------
    pred: dict[str, float] = {"home_0": 0.0}
    for key in non_home_keys:
        pred[key] = timing_model.sample(key, x_person, n_tours)
    for key in home_keys[1:]:  # home_1 … home_{n-1} (including last home)
        pred[key] = timing_model.sample(key, x_person, n_tours)

    # ---- (iii) order by predicted start: home_0 first, home_{n-1} last ------
    # The last home episode is fixed at the end regardless of its predicted start
    # so the schedule always ends with home. Intermediate homes + non-home
    # activities sort freely between home_0 and home_{n-1}.
    last_home = home_keys[-1]
    middle_keys = non_home_keys + home_keys[1:-1]  # intermediate homes only
    middle_sorted = sorted(middle_keys, key=lambda k: pred[k])
    ordered = ["home_0"] + middle_sorted + [last_home]
    K = len(ordered)

    ep_types = [k.rsplit("_", 1)[0] for k in ordered]
    type_idx: dict[str, list[int]] = {}
    for pos, atype in enumerate(ep_types):
        type_idx.setdefault(atype, []).append(pos)

    t_pred = np.array([pred[k] for k in ordered], dtype=np.float64)
    float_totals = {t: float(v) for t, v in int_totals.items()}

    # ---- (iv) SLSQP optimisation -------------------------------------------
    # Initial guess: equal split per type
    d0 = np.zeros(K)
    for atype, positions in type_idx.items():
        per = float_totals[atype] / len(positions)
        for pos in positions:
            d0[pos] = max(min_dur, per)

    def _obj(d: np.ndarray) -> float:
        # scheduled_start[i] = sum(d[0:i])
        s = np.empty(K)
        s[0] = 0.0
        np.cumsum(d[:-1], out=s[1:])
        diff = s - t_pred
        return float(diff @ diff)

    def _jac(d: np.ndarray) -> np.ndarray:
        s = np.empty(K)
        s[0] = 0.0
        np.cumsum(d[:-1], out=s[1:])
        errors = s - t_pred
        # df/dd_k = 2 * sum_{i > k} errors[i]  (d_k shifts all starts after it)
        suffix = np.cumsum(errors[::-1])[::-1]
        grad = np.zeros(K)
        grad[:-1] = 2.0 * suffix[1:]
        return grad

    eq_cons = [
        {
            "type": "eq",
            "fun": lambda d, ii=positions, tt=float_totals[atype]: sum(d[j] for j in ii) - tt,
        }
        for atype, positions in type_idx.items()
    ]
    bounds = [(min_dur, None)] * K

    try:
        res = _sp_min(
            _obj, d0, method="SLSQP", jac=_jac, bounds=bounds,
            constraints=eq_cons,
            options={"maxiter": 500, "ftol": 1e-8, "disp": False},
        )
        d_opt = np.maximum(min_dur, res.x)
    except Exception:
        d_opt = d0.copy()

    # ---- Round to integers preserving per-type totals -----------------------
    d_fl = np.floor(d_opt).astype(int)
    for atype, positions in type_idx.items():
        target = int_totals[atype]
        deficit = target - sum(d_fl[j] for j in positions)
        if deficit > 0:
            # Add 1 to episodes with largest fractional parts first
            by_frac = sorted(positions, key=lambda j: -(d_opt[j] - d_fl[j]))
            for j in by_frac[:deficit]:
                d_fl[j] += 1
        elif deficit < 0:
            # Remove 1 from largest episodes (keep >= 1)
            for j in sorted(positions, key=lambda j: -d_fl[j]):
                if d_fl[j] > 1 and deficit < 0:
                    d_fl[j] -= 1
                    deficit += 1
    d_int = np.maximum(1, d_fl)

    def _strip(key: str) -> str:
        return key.rsplit("_", 1)[0]

    seq = [(_strip(k), int(d_int[i])) for i, k in enumerate(ordered)]
    return _to_rows(seq)


def assemble_greedy_mdcev_schedule(
    alloc: dict[str, float],
    episode_counts: dict[str, int],
    x_person: np.ndarray,
    timing_model,
) -> list[dict]:
    """Assemble a 1440-min schedule by greedy start-time ordering.

    Algorithm:
    1. Build non-home episodes: each type split equally across episode_counts[type].
       Episodes keyed as type_N (e.g. shop_0, shop_1) so timing model captures
       1st vs 2nd participation.
    2. Build home episodes: intermediate homes (home_1 … home_{n-2}) each get
       HOME_INTER minutes; remaining home time is split between home_0 (morning)
       and the final home (evening, fills 1440 remainder).
    3. Predict start time for every non-home episode and every intermediate home
       episode via timing_model.sample(key, x_person, n_tours).
    4. home_0 is fixed at t=0. Last home fills the 1440 remainder.
       Sort everything else by predicted start time and place greedily.

    Returns list of {act, start, end, duration} dicts summing to 1440.
    """
    MIN_NON_HOME = 10.0

    # 1. Build non-home episodes (equal split, filter tiny)
    non_home_eps: list[tuple[str, int]] = []
    for atype, total in alloc.items():
        if atype == "home" or total < MIN_NON_HOME:
            continue
        n = max(1, int(episode_counts.get(atype, 1)))
        per = max(1, int(round(total / n)))
        for i in range(n):
            non_home_eps.append((f"{atype}_{i}", per))

    if not non_home_eps:
        return _to_rows([("home", 1440)])

    # 2. Build home episodes
    non_home_total = sum(d for _, d in non_home_eps)
    home_available = max(0, 1440 - non_home_total)

    n_home = max(2, int(episode_counts.get("home", 2)))
    n_intermediate = max(0, n_home - 2)
    n_tours = max(1, n_home - 1)

    # Intermediate homes get HOME_INTER; remainder shared between morning and last home
    inter_total = n_intermediate * HOME_INTER
    remaining_home = max(0, home_available - inter_total)
    morning = max(0, remaining_home // 2)

    home_eps: list[tuple[str, int]] = [("home_0", morning)]
    for i in range(1, n_home - 1):
        home_eps.append((f"home_{i}", HOME_INTER))
    # last home is appended at the end of the sequence with 1440-used

    # 3. Predict start times: non-home episodes + intermediate homes
    start_preds: dict[str, float] = {}
    for key, _ in non_home_eps:
        start_preds[key] = timing_model.sample(key, x_person, n_tours)
    for key, _ in home_eps[1:]:  # intermediate homes (home_1 … home_{n-2})
        start_preds[key] = timing_model.sample(key, x_person, n_tours)

    # 4. Sort non-home + intermediate homes by predicted start time; place greedily
    to_order: list[tuple[str, int]] = non_home_eps + list(home_eps[1:])
    to_order.sort(key=lambda x: start_preds.get(x[0], 720.0))

    def _strip(key: str) -> str:
        return key.rsplit("_", 1)[0]

    seq: list[tuple[str, int]] = [("home", morning)]
    for key, dur in to_order:
        seq.append((_strip(key), dur))
    used = sum(d for _, d in seq)
    seq.append(("home", max(0, 1440 - used)))

    return _to_rows(seq)


def assemble_enumerated_mdcev_schedule(
    alloc: dict[str, float],
    x_person: np.ndarray,
    timing_model,
) -> list[dict]:
    """Assemble a 1440-min schedule from enumerated MDCEV allocations.

    Args:
        alloc: dict mapping enumerated keys (home_0, home_1, work_0, shop_0, ...)
               to allocated minutes (output of MDCEVModel.sample_batch).
        x_person: 1-D person feature vector (aligned to training feature_names).
        timing_model: StartTimeModel used to predict activity start times.

    Returns:
        List of row dicts [{act, start, end, duration}] summing to 1440,
        with enumeration suffixes stripped (home_0 → home, work_0 → work, etc.).
    """
    MIN_NON_HOME = 10.0

    # 1. Separate home and non-home; filter non-home below threshold
    home_alloc: dict[str, float] = {}
    non_home_alloc: list[tuple[str, float]] = []
    for key, dur in alloc.items():
        base = key.rsplit("_", 1)[0]
        if base == "home":
            if dur > 0:
                home_alloc[key] = dur
        elif dur >= MIN_NON_HOME:
            non_home_alloc.append((key, dur))

    if "home_0" not in home_alloc:
        home_alloc["home_0"] = 0.0

    # Sort home keys by suffix index
    home_keys = sorted(home_alloc, key=lambda k: int(k.rsplit("_", 1)[1]))

    # Non-home activities rounded to int
    non_home_items: list[tuple[str, int]] = [
        (k, int(round(v))) for k, v in non_home_alloc
    ]
    non_home_total = sum(d for _, d in non_home_items)

    # H schedule: no non-home activities
    if not non_home_items:
        return _to_rows([("home", 1440)])

    # Ensure at least two home episodes (need a closing home after activities)
    if len(home_keys) == 1:
        home_alloc["home_1"] = max(0.0, 1440.0 - home_alloc["home_0"] - non_home_total)
        home_keys = ["home_0", "home_1"]

    n_home = len(home_keys)
    n_tours = n_home - 1

    # 2. Distribute home budget proportionally among home episodes
    home_available = max(0, 1440 - non_home_total)
    home_raw = [home_alloc[k] for k in home_keys]
    home_raw_sum = sum(home_raw)
    if home_raw_sum > 0:
        home_durs = [home_available * v / home_raw_sum for v in home_raw]
    else:
        home_durs = [home_available / n_home] * n_home

    # 3. Enforce minimum duration on intermediate homes (home_1 ... home_{n-2})
    for i in range(1, n_home - 1):
        if home_durs[i] < HOME_INTER:
            deficit = HOME_INTER - home_durs[i]
            home_durs[i] = float(HOME_INTER)
            # Subtract from the largest remaining home
            others = [(j, home_durs[j]) for j in range(n_home) if j != i]
            largest_j = max(others, key=lambda x: x[1])[0]
            home_durs[largest_j] = max(0.0, home_durs[largest_j] - deficit)

    home_durs_int = [max(0, int(round(d))) for d in home_durs]

    # 4. Predict start times for all non-home activities and inter-tour homes
    start_preds: dict[str, float] = {}
    for key, _ in non_home_items:
        start_preds[key] = timing_model.sample(key, x_person, n_tours)
    # Inter-tour homes (home_1 ... home_{n-2}) act as tour boundaries
    for i in range(1, n_home - 1):
        key = home_keys[i]
        start_preds[key] = timing_model.sample(key, x_person, n_tours)

    # 5. Greedy assignment: partition non-home activities into tour slots
    if n_tours == 1:
        tours_list = [sorted(non_home_items, key=lambda x: start_preds.get(x[0], 0.0))]
    else:
        inter_home_keys = home_keys[1:-1]  # home_1 ... home_{n-2}
        boundaries = sorted(start_preds[k] for k in inter_home_keys)

        tours_list: list[list[tuple[str, int]]] = [[] for _ in range(n_tours)]
        for key, dur in non_home_items:
            pred = start_preds.get(key, 0.0)
            slot = n_tours - 1
            for idx, boundary in enumerate(boundaries):
                if pred < boundary:
                    slot = idx
                    break
            tours_list[slot].append((key, dur))

        for i in range(len(tours_list)):
            tours_list[i] = sorted(tours_list[i], key=lambda x: start_preds.get(x[0], 0.0))

    # 6. Build sequence and strip enumeration suffixes
    def _strip(key: str) -> str:
        return key.rsplit("_", 1)[0]

    seq: list[tuple[str, int]] = [("home", home_durs_int[0])]
    for i, tour in enumerate(tours_list):
        for key, dur in tour:
            seq.append((_strip(key), dur))
        seq.append(("home", home_durs_int[i + 1]))

    return _to_rows(seq)


def _partition_by_placement(
    disc: list[tuple[str, int]],
    placements: list[str],
) -> tuple[
    list[tuple[str, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
    list[tuple[str, int]],
]:
    """Split disc activities into outbound stops, inbound stops, sep_pre, sep_post."""
    outbound: list[tuple[str, int]] = []
    inbound: list[tuple[str, int]] = []
    sep_pre: list[tuple[str, int]] = []
    sep_post: list[tuple[str, int]] = []

    pl = list(placements) if placements else []
    # Any disc activity beyond the placements list defaults to inbound
    for i, (a, d) in enumerate(disc):
        p = pl[i] if i < len(pl) else "mandatory_inbound"
        if p == "mandatory_outbound":
            outbound.append((a, d))
        elif p == "separate_before":
            sep_pre.append((a, d))
        elif p == "separate_after":
            sep_post.append((a, d))
        else:
            inbound.append((a, d))

    return outbound, inbound, sep_pre, sep_post


def _group_into_tours(
    disc: list[tuple[str, int]],
    tour_groupings: list[bool],
) -> list[list[tuple[str, int]]]:
    """Group disc activities into home-based tours using grouping flags."""
    if not disc:
        return []
    tours: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = [disc[0]]
    for i in range(len(disc) - 1):
        same = tour_groupings[i] if i < len(tour_groupings) else False
        if same:
            current.append(disc[i + 1])
        else:
            tours.append(current)
            current = [disc[i + 1]]
    tours.append(current)
    return tours


def _split_duration(
    atype: str, total: int, episode_model
) -> list[tuple[str, int]]:
    """Split total minutes into n episodes; last episode absorbs rounding."""
    n = episode_model.sample(atype, float(total))
    per = total // n
    last = total - per * (n - 1)
    return [(atype, per)] * (n - 1) + [(atype, last)]


def _enforce_budget(
    mandatory_dur: int, disc: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """Proportionally rescale disc activities if total non-home time > 1380."""
    total_nonhome = mandatory_dur + sum(d for _, d in disc)
    if total_nonhome > 1380 and total_nonhome > 0:
        disc_total = sum(d for _, d in disc)
        allowed_disc = max(0, 1380 - mandatory_dur)
        if disc_total > 0:
            disc = _scale_activities(disc, allowed_disc)
    return disc


def _scale_activities(
    activities: list[tuple[str, int]], budget: int
) -> list[tuple[str, int]]:
    """Proportionally scale activity durations to fit within budget."""
    total = sum(d for _, d in activities)
    if total <= 0:
        return activities
    scale = budget / total
    scaled = [(a, max(10, int(d * scale))) for a, d in activities]
    # Fix rounding: adjust last item
    diff = budget - sum(d for _, d in scaled)
    if scaled:
        last_a, last_d = scaled[-1]
        scaled[-1] = (last_a, max(10, last_d + diff))
    return scaled


def _to_rows(seq: list[tuple[str, int]]) -> list[dict]:
    """Convert (act, dur) sequence to row dicts, fix rounding, assert 1440."""
    # Remove zero-duration activities
    seq = [(a, d) for a, d in seq if d > 0]

    # Merge consecutive same-type activities (e.g. two "visit" slots → one)
    merged: list[tuple[str, int]] = []
    for act, dur in seq:
        if merged and merged[-1][0] == act:
            merged[-1] = (act, merged[-1][1] + dur)
        else:
            merged.append((act, dur))
    seq = merged

    # Fix rounding so sum == 1440
    total = sum(d for _, d in seq)
    if total != 1440:
        # Find last home and adjust
        for i in reversed(range(len(seq))):
            if seq[i][0] == "home":
                new_dur = seq[i][1] + (1440 - total)
                if new_dur >= 0:
                    seq[i] = ("home", new_dur)
                    break
        else:
            # Fallback: adjust last activity
            seq[-1] = (seq[-1][0], seq[-1][1] + (1440 - total))

    # Remove any newly zeroed entries
    seq = [(a, d) for a, d in seq if d > 0]

    # Build rows
    rows = []
    cursor = 0
    for act, dur in seq:
        rows.append({"act": act, "start": cursor, "end": cursor + dur, "duration": dur})
        cursor += dur

    return rows
