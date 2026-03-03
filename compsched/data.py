"""Data loading, DAP classification, and training dataset construction."""

import numpy as np
import polars as pl

LABEL_COLS = ["gender", "age_group", "car_access", "work_status", "income"]
MANDATORY_ACTS = {"work", "education"}
DISC_ACTS = {"shop", "visit", "escort", "medical", "other"}


def load_attributes(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.with_columns(
        pl.when(pl.col("car_access") == "unknown")
        .then(pl.lit("yes"))
        .otherwise(pl.col("car_access"))
        .alias("car_access")
    )
    return df.select(["pid"] + LABEL_COLS)


def load_schedules(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    if "hid" in df.columns:
        df = df.drop("hid")
    return df.sort(["pid", "start"])


def classify_dap(acts: list[str]) -> str:
    act_set = set(acts)
    has_mandatory = bool(act_set & MANDATORY_ACTS)
    has_disc = bool(act_set & DISC_ACTS)
    if not has_mandatory and not has_disc:
        return "H"
    elif has_mandatory and not has_disc:
        return "W"
    elif has_mandatory and has_disc:
        return "WD"
    else:
        return "D"


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
        mandatory_duration = 0
        work_start = None
        mandatory_type = "work"
        for a in activities:
            if a["act"] in MANDATORY_ACTS:
                if work_start is None:
                    work_start = int(a["start"])
                    mandatory_type = a["act"]
                mandatory_duration += int(a["duration"])

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

        # Before-work flags (WD only)
        before_work_flags: list[bool] = []
        if dap == "WD" and work_start is not None:
            before_work_flags = [start < work_start for _, _, start in disc_raw]

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
                    "dap_WD": int(dap == "WD"),
                    **label_vals,
                }
            )

        records.append(
            {
                "pid": pid,
                "dap": dap,
                "mandatory_duration": float(mandatory_duration),
                "mandatory_type": mandatory_type,
                "work_start": float(work_start) if work_start is not None else None,
                "first_departure": float(first_departure)
                if first_departure is not None
                else None,
                "n_disc": n_disc,
                "disc_activities": [(a, d) for a, d, _ in disc_raw],
                "before_work_flags": before_work_flags,
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
