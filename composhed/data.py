"""Data loading, DAP classification, and training dataset construction."""

import numpy as np
import polars as pl

LABEL_COLS = ["age", "hh_income", "sex", "employment", "day", "hh_zone", "access_egress_distance", "vehicles"]
# Reduced feature set for MDCEV: drop spatial/mode-choice features irrelevant to time allocation
MDCEV_LABEL_COLS = ["age", "hh_income", "sex", "employment", "day"]
MANDATORY_ACTS = {"work", "education"}
DISC_ACTS = {"shop", "visit", "escort", "medical", "other"}
ACTIVITY_TYPES = ["home", "work", "education", "shop", "visit", "escort", "medical", "other"]


def load_attributes(path: str) -> pl.DataFrame:
    schema_overrides = {c: pl.Utf8 for c in LABEL_COLS}
    df = pl.read_csv(path, schema_overrides=schema_overrides)
    df = df.with_columns([pl.col(c).fill_null("unknown") for c in LABEL_COLS])
    return df.select(["pid"] + LABEL_COLS)


def load_schedules(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    for col in ("hid", "zone", "seq"):
        if col in df.columns:
            df = df.drop(col)
    if "duration" not in df.columns:
        df = df.with_columns((pl.col("end") - pl.col("start")).alias("duration"))
    return df.sort(["pid", "start"])


def classify_dap(acts: list[str]) -> str:
    act_set = set(acts)
    has_work = "work" in act_set
    has_edu = "education" in act_set
    has_mandatory = has_work or has_edu
    has_disc = bool(act_set & DISC_ACTS)
    if not has_mandatory and not has_disc:
        return "H"
    elif has_mandatory and not has_disc:
        return "W" if has_work else "E"
    elif has_mandatory and has_disc:
        return "WD" if has_work else "ED"
    else:
        return "D"


def extract_tour_memberships(activities: list[dict]) -> list[str]:
    """Return tour placement label for each discretionary activity in schedule order.

    Labels:
      "mandatory_outbound"  — stop before mandatory in same tour (no home between)
      "mandatory_inbound"   — stop after mandatory in same tour
      "separate_before"     — own home-based tour occurring before the mandatory tour
      "separate_after"      — own home-based tour occurring after the mandatory tour
      "standalone"          — D-schedule activity (no mandatory tour in day)
    """
    # Split on home boundaries to get raw tours
    tours: list[list[dict]] = []
    current: list[dict] = []
    for a in activities:
        if a["act"] == "home":
            if current:
                tours.append(current)
                current = []
        else:
            current.append(a)
    if current:
        tours.append(current)

    # Find the first tour containing a mandatory activity
    mandatory_tour_idx: int | None = None
    for i, tour in enumerate(tours):
        if any(a["act"] in MANDATORY_ACTS for a in tour):
            mandatory_tour_idx = i
            break

    result: list[str] = []
    for i, tour in enumerate(tours):
        disc_in_tour = [a for a in tour if a["act"] in DISC_ACTS]
        if not disc_in_tour:
            continue

        if mandatory_tour_idx is None:
            for _ in disc_in_tour:
                result.append("standalone")
        elif i == mandatory_tour_idx:
            mand_start = next(a["start"] for a in tour if a["act"] in MANDATORY_ACTS)
            for a in disc_in_tour:
                result.append(
                    "mandatory_outbound" if a["start"] < mand_start else "mandatory_inbound"
                )
        elif i < mandatory_tour_idx:
            for _ in disc_in_tour:
                result.append("separate_before")
        else:
            for _ in disc_in_tour:
                result.append("separate_after")

    return result


def build_training_dataset(
    attr_df: pl.DataFrame, sched_df: pl.DataFrame
) -> tuple[list[dict], list[dict]]:
    """Build per-pid training records and per-disc-activity slot records.

    Returns:
        records: list of dicts, one per pid (scalar features + nested lists)
        disc_slot_records: list of dicts, one per disc activity (for Steps 4/5)
    """
    # Index attributes by pid
    attr_by_pid = {row["pid"]: row for row in attr_df.iter_rows(named=True)}

    # Group schedule rows by pid (already sorted by pid, start)
    sched_by_pid: dict[int, list[dict]] = {}
    for row in sched_df.iter_rows(named=True):
        sched_by_pid.setdefault(row["pid"], []).append(row)

    records: list[dict] = []
    disc_slot_records: list[dict] = []

    for pid, activities in sched_by_pid.items():
        if pid not in attr_by_pid:
            continue
        attrs = attr_by_pid[pid]
        acts = [a["act"] for a in activities]
        dap = classify_dap(acts)

        # Mandatory duration, work_start, mandatory_type
        work_duration = 0
        education_duration = 0
        work_start = None
        mandatory_type = "work"
        for a in activities:
            if a["act"] == "work":
                if work_start is None:
                    work_start = int(a["start"])
                    mandatory_type = "work"
                work_duration += int(a["duration"])
            elif a["act"] == "education":
                if work_start is None:
                    work_start = int(a["start"])
                    mandatory_type = "education"
                education_duration += int(a["duration"])
        mandatory_duration = work_duration + education_duration

        # First departure (first non-home activity start)
        first_departure = None
        for a in activities:
            if a["act"] != "home":
                first_departure = int(a["start"])
                break

        # Discretionary activities in schedule order: (type, duration, start)
        disc_raw = [
            (a["act"], int(a["duration"]), int(a["start"]))
            for a in activities
            if a["act"] in DISC_ACTS
        ]
        n_disc = len(disc_raw)

        # Home time
        home_rows = [a for a in activities if a["act"] == "home"]
        total_home = sum(int(a["duration"]) for a in home_rows)
        home_morning = int(home_rows[0]["duration"]) if home_rows else 0
        home_ratio = home_morning / total_home if total_home > 0 else 0.5

        # Tour membership for each disc activity
        tour_placement: list[str] = extract_tour_memberships(activities)

        # Before-work flags (WD and ED only) — kept for backward compatibility
        before_work_flags: list[bool] = []
        if dap in ("WD", "ED") and work_start is not None:
            before_work_flags = [start < work_start for _, _, start in disc_raw]

        # For D schedules: whether each consecutive pair of disc activities share a tour
        same_tour_as_next: list[bool] = []
        if dap == "D" and len(disc_raw) >= 2:
            tours_raw: list[list[dict]] = []
            _cur: list[dict] = []
            for a in activities:
                if a["act"] == "home":
                    if _cur:
                        tours_raw.append(_cur)
                        _cur = []
                else:
                    _cur.append(a)
            if _cur:
                tours_raw.append(_cur)
            disc_tour_idx: list[int] = [
                i
                for i, t in enumerate(tours_raw)
                for a in t
                if a["act"] in DISC_ACTS
            ]
            same_tour_as_next = [
                disc_tour_idx[j] == disc_tour_idx[j + 1]
                for j in range(len(disc_tour_idx) - 1)
            ]

        # Disc slot records (for atype and duration models)
        disc_durs = [d for _, d, _ in disc_raw]
        label_vals = {k: attrs[k] for k in LABEL_COLS}
        for i, (atype, dur, _) in enumerate(disc_raw):
            slot_idx = i + 1
            slot_key = str(slot_idx) if slot_idx <= 2 else "3+"
            slot_numeric = min(slot_idx, 3)
            remaining_budget = sum(disc_durs[i:])
            disc_slot_records.append(
                {
                    "pid": pid,
                    "slot_key": slot_key,
                    "slot_numeric": slot_numeric,
                    "remaining_budget": float(remaining_budget),
                    "atype": atype,
                    "duration": dur,
                    "dap": dap,
                    "dap_WD": int(dap in ("WD", "ED")),
                    **label_vals,
                }
            )

        records.append(
            {
                "pid": pid,
                "dap": dap,
                "mandatory_duration": float(mandatory_duration),
                "work_duration": float(work_duration),
                "education_duration": float(education_duration),
                "mandatory_type": mandatory_type,
                "work_start": float(work_start) if work_start is not None else None,
                "first_departure": float(first_departure)
                if first_departure is not None
                else None,
                "n_disc": n_disc,
                "disc_activities": [(a, d) for a, d, _ in disc_raw],
                "tour_placement": tour_placement,
                "before_work_flags": before_work_flags,
                "same_tour_as_next": same_tour_as_next,
                "total_home": float(total_home),
                "home_morning": float(home_morning),
                "home_ratio": float(home_ratio),
                **label_vals,
            }
        )

    return records, disc_slot_records


def encode_features(
    records: list[dict],
    label_cols: list[str] = LABEL_COLS,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode label columns from a list of record dicts.

    Returns (X, feature_names) where X is float64 numpy array.
    """
    rows = [{k: str(r[k]) for k in label_cols} for r in records]
    df = pl.DataFrame(rows)
    df_enc = df.to_dummies(columns=label_cols, drop_first=True)
    if feature_names is not None:
        # Align to stored feature names (add missing=0, drop extras)
        for col in feature_names:
            if col not in df_enc.columns:
                df_enc = df_enc.with_columns(pl.lit(0).alias(col))
        df_enc = df_enc.select(feature_names)
    cols = df_enc.columns
    X = df_enc.to_numpy(allow_copy=True).astype(np.float64)
    return X, cols


def encode_for_generation(
    attr_df: pl.DataFrame,
    label_cols: list[str],
    feature_names: list[str],
) -> np.ndarray:
    """Encode attributes DataFrame to aligned numpy feature matrix."""
    df = attr_df.select(label_cols).with_columns(
        [pl.col(c).cast(pl.Utf8) for c in label_cols]
    )
    df_enc = df.to_dummies(columns=label_cols, drop_first=True)
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc = df_enc.with_columns(pl.lit(0).alias(col))
    df_enc = df_enc.select(feature_names)
    return df_enc.to_numpy(allow_copy=True).astype(np.float64)


def compute_mean_home_times(records: list[dict]) -> dict[str, float]:
    """Mean total home duration per DAP type."""
    from collections import defaultdict

    totals: dict[str, list[float]] = defaultdict(list)
    for r in records:
        totals[r["dap"]].append(r["total_home"])
    return {dap: float(np.mean(vals)) for dap, vals in totals.items()}


def build_enumerated_dataset(
    attr_df: pl.DataFrame,
    sched_df: pl.DataFrame,
    min_disc_dur: int = 5,
    cap_percentile: float = 99.0,
) -> tuple[list[dict], dict[str, int]]:
    """Build per-pid records with per-occurrence enumerated activity durations.

    Each activity occurrence gets its own key: home_0, home_1, work_0, shop_0, etc.
    Non-home activities with duration < min_disc_dur are excluded; their time is
    redistributed into home so each person's durations sum to 1440.
    Home occurrences are capped at the cap_percentile of the training distribution
    (per type) to avoid fitting rare multi-tour patterns with no statistical support.

    Returns:
        records: list of dicts with enumerated_durations (sum=1440),
                 enumerated_starts, n_tours, and label cols
        max_counts: dict mapping base activity type to max occurrence index (inclusive)
    """
    attr_by_pid = {row["pid"]: row for row in attr_df.iter_rows(named=True)}

    sched_by_pid: dict[int, list[dict]] = {}
    for row in sched_df.iter_rows(named=True):
        sched_by_pid.setdefault(row["pid"], []).append(row)

    # First pass: count per-type occurrences per pid to determine max_counts
    raw_counts: dict[str, list[int]] = {}
    for pid, activities in sched_by_pid.items():
        if pid not in attr_by_pid:
            continue
        type_count: dict[str, int] = {}
        for a in activities:
            atype = a["act"]
            dur = int(a["duration"])
            if atype == "home" or dur >= min_disc_dur:
                type_count[atype] = type_count.get(atype, 0) + 1
        for atype, cnt in type_count.items():
            raw_counts.setdefault(atype, []).append(cnt)

    max_counts: dict[str, int] = {}
    for atype, counts in raw_counts.items():
        max_counts[atype] = max(1, int(np.percentile(counts, cap_percentile)))

    # Second pass: build enumerated records
    records: list[dict] = []
    for pid, activities in sched_by_pid.items():
        if pid not in attr_by_pid:
            continue
        attrs = attr_by_pid[pid]

        enumerated_durations: dict[str, float] = {}
        enumerated_starts: dict[str, float] = {}
        type_counter: dict[str, int] = {}

        for a in activities:
            atype = a["act"]
            dur = int(a["duration"])
            start = int(a["start"])
            if atype != "home" and dur < min_disc_dur:
                continue  # Skip short non-home activities

            cap = max_counts.get(atype, 1)
            idx = type_counter.get(atype, 0)

            if idx < cap:
                key = f"{atype}_{idx}"
                enumerated_durations[key] = float(dur)
                enumerated_starts[key] = float(start)
            else:
                # Merge excess occurrences into last allowed slot
                key = f"{atype}_{cap - 1}"
                enumerated_durations[key] = enumerated_durations.get(key, 0.0) + float(dur)
                # Keep earliest start time for the key

            type_counter[atype] = idx + 1

        if "home_0" not in enumerated_durations:
            continue  # Malformed schedule — skip

        # Ensure durations sum to exactly 1440: home absorbs any filtered time
        non_home_total = sum(v for k, v in enumerated_durations.items() if not k.startswith("home_"))
        home_available = 1440.0 - non_home_total
        home_keys = [k for k in enumerated_durations if k.startswith("home_")]
        home_raw_sum = sum(enumerated_durations[k] for k in home_keys)
        if home_raw_sum > 0:
            scale = home_available / home_raw_sum
            for k in home_keys:
                enumerated_durations[k] = max(0.0, enumerated_durations[k] * scale)
        elif home_keys:
            enumerated_durations[home_keys[0]] = home_available

        n_home = len(home_keys)
        n_tours = max(0, n_home - 1)

        label_vals = {k: attrs[k] for k in LABEL_COLS}
        records.append({
            "pid": pid,
            "enumerated_durations": enumerated_durations,
            "enumerated_starts": enumerated_starts,
            "n_tours": n_tours,
            **label_vals,
        })

    return records, max_counts


def derive_type_records(records: list[dict]) -> list[dict]:
    """Convert build_training_dataset records to per-type totals and episode counts.

    Each output record has:
        total_durations: dict[str, float] — total minutes per type (sums to ~1440)
        episode_counts:  dict[str, int]  — number of episodes per type (home = n_tours + 1)
        dap, label cols

    Used to train MDCEVModel (8-type totals) and EpisodeCountModel.
    """
    result: list[dict] = []
    for r in records:
        dap = r["dap"]
        total_durations: dict[str, float] = {}
        episode_counts: dict[str, int] = {}

        # Home total
        total_durations["home"] = max(0.0, float(r["total_home"]))

        # Mandatory
        if r["work_duration"] > 0:
            total_durations["work"] = float(r["work_duration"])
            episode_counts["work"] = 1
        if r["education_duration"] > 0:
            total_durations["education"] = float(r["education_duration"])
            episode_counts["education"] = 1

        # Discretionary: aggregate total per type and count episodes
        disc_type_dur: dict[str, float] = {}
        disc_type_cnt: dict[str, int] = {}
        for atype, dur in r["disc_activities"]:
            disc_type_dur[atype] = disc_type_dur.get(atype, 0.0) + float(dur)
            disc_type_cnt[atype] = disc_type_cnt.get(atype, 0) + 1
        for atype in disc_type_dur:
            total_durations[atype] = disc_type_dur[atype]
            episode_counts[atype] = disc_type_cnt[atype]

        # Home episode count (n_home_episodes - 1 = n_home_based_tours)
        if dap == "H":
            n_home = 1
        elif dap in ("W", "E"):
            n_home = 2
        elif dap in ("WD", "ED"):
            tp = r.get("tour_placement", [])
            n_home = (
                2
                + int(any(p == "separate_before" for p in tp))
                + int(any(p == "separate_after" for p in tp))
            )
        else:  # D
            same = r.get("same_tour_as_next", [])
            n_disc_tours = (sum(1 for b in same if not b) + 1) if r["n_disc"] > 0 else 0
            n_home = n_disc_tours + 1
        episode_counts["home"] = n_home

        # Ensure totals sum to exactly 1440 (home absorbs rounding)
        non_home_total = sum(v for k, v in total_durations.items() if k != "home")
        total_durations["home"] = max(0.0, 1440.0 - non_home_total)

        label_vals = {k: r[k] for k in LABEL_COLS}
        result.append({
            "pid": r["pid"],
            "total_durations": total_durations,
            "episode_counts": episode_counts,
            "dap": dap,
            **label_vals,
        })
    return result
