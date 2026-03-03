"""Generate synthetic schedules for all persons in an attributes file."""

import argparse

import joblib
import numpy as np
import polars as pl

from compsched.assembly import assemble_schedule
from compsched.data import (
    LABEL_COLS,
    encode_for_generation,
    load_attributes,
)


def generate(
    attributes_path: str,
    models_path: str,
    out_attributes: str,
    out_schedules: str,
) -> None:
    print("Loading models...")
    bundle = joblib.load(models_path)
    dap_model = bundle["dap"]
    mand_model = bundle["mandatory"]
    ntours_model = bundle["ntours"]
    atype_model = bundle["atype"]
    dur_model = bundle["duration"]
    anchor_model = bundle["anchor"]
    feature_names: list[str] = bundle["feature_names"]
    mean_home: dict[str, float] = bundle["mean_home"]

    print("Loading attributes...")
    attr_df = load_attributes(attributes_path)
    pids = attr_df["pid"].to_list()
    n = len(pids)

    # Encode all persons' label features
    X_all = encode_for_generation(attr_df, LABEL_COLS, feature_names)

    # Sample DAP for all persons at once
    print(f"Sampling DAP for {n} persons...")
    dap_list = dap_model.sample(X_all)

    # Build attribute rows (same format as reference synthetic_attributes.csv)
    attr_rows = attr_df.select(["pid"] + LABEL_COLS).to_dicts()

    # Generate schedules
    print("Generating schedules...")
    sched_rows: list[dict] = []

    for i, (pid, dap) in enumerate(zip(pids, dap_list)):
        x_label = X_all[i]
        work_status = str(attr_rows[i]["work_status"])

        try:
            rows = _generate_one(
                pid=pid,
                dap=dap,
                x_label=x_label,
                work_status=work_status,
                mand_model=mand_model,
                ntours_model=ntours_model,
                atype_model=atype_model,
                dur_model=dur_model,
                anchor_model=anchor_model,
                mean_home=mean_home,
                feature_names=feature_names,
            )
        except Exception as exc:
            # Fallback: single home activity
            print(f"  WARNING pid={pid}: {exc}; using fallback H schedule")
            rows = [{"act": "home", "start": 0, "end": 1440, "duration": 1440}]

        for row in rows:
            row["pid"] = pid
            sched_rows.append(row)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n}")

    # Write outputs — polars with pandas-style unnamed index column
    print(f"Writing {out_attributes} ...")
    out_attr_df = pl.DataFrame(attr_rows).select(["pid"] + LABEL_COLS)
    _write_csv_with_index(out_attr_df, out_attributes)

    print(f"Writing {out_schedules} ...")
    sched_df = pl.DataFrame(sched_rows).select(["pid", "act", "start", "end", "duration"])
    _write_csv_with_index(sched_df, out_schedules)

    print("Done.")


def _generate_one(
    pid: int,
    dap: str,
    x_label: np.ndarray,
    work_status: str,
    mand_model,
    ntours_model,
    atype_model,
    dur_model,
    anchor_model,
    mean_home: dict[str, float],
    feature_names: list[str],
) -> list[dict]:
    """Generate schedule for one person."""

    # ---- H: nothing to do ---------------------------------------------------
    if dap == "H":
        return assemble_schedule("H", 0, "home", [], None, None, [])

    # ---- Step 2: Mandatory duration -----------------------------------------
    mandatory_duration = 0.0
    mandatory_type = "work"
    work_start = None

    if dap in ("W", "WD"):
        if mand_model is not None:
            dap_WD = np.array([[1.0 if dap == "WD" else 0.0]])
            x_mand = np.hstack([x_label.reshape(1, -1), dap_WD])
            mandatory_duration = float(mand_model.sample(x_mand)[0])
        else:
            mandatory_duration = 480.0

        # Anchor: work start
        work_start = anchor_model.sample_work_start(work_status)
        # Ensure work fits in day
        work_start = float(np.clip(work_start, 0.0, 1440.0 - mandatory_duration - 60.0))

    # ---- Step 3: Number of disc tours ---------------------------------------
    disc_activities: list[tuple[str, int]] = []

    if dap in ("WD", "D"):
        if ntours_model is not None:
            dap_WD = np.array([[1.0 if dap == "WD" else 0.0]])
            remaining_budget = (
                1440.0 - mandatory_duration - mean_home.get(dap, 400.0)
            )
            x_ntours = np.hstack(
                [x_label.reshape(1, -1), dap_WD, [[remaining_budget]]]
            )
            avail = remaining_budget / 30.0  # min 30 min per activity
            max_allowed = np.array([max(0, int(avail))])
            n_disc = int(ntours_model.sample(x_ntours, max_allowed)[0])
        else:
            n_disc = 1

        # ---- Steps 4 & 5: Sample types and durations per slot ---------------
        remaining = max(0.0, 1440.0 - mandatory_duration - mean_home.get(dap, 400.0))

        for slot_idx in range(1, n_disc + 1):
            slot_key = str(slot_idx) if slot_idx <= 2 else "3+"
            slot_numeric = min(slot_idx, 3)

            atype = atype_model.sample_slot(
                x_label,
                slot=slot_key,
                dap_WD=int(dap == "WD"),
                slot_numeric=slot_numeric,
                remaining_budget=remaining,
            )
            dur = dur_model.sample(atype, x_label, remaining_budget=remaining)
            dur = int(max(10, round(dur)))
            remaining = max(0.0, remaining - dur)
            disc_activities.append((atype, dur))

    # ---- Step 6a: Anchor timing for D ---------------------------------------
    first_departure = None
    if dap == "D":
        first_departure = anchor_model.sample_first_departure(work_status)

    # ---- Step 6b: Before-work flags (WD only) --------------------------------
    before_work_flags: list[bool] = []
    if dap == "WD" and disc_activities and work_start is not None:
        before_work_flags = anchor_model.sample_before_work_flags(
            x_label, disc_activities, work_start
        )

    # ---- Step 6c: Assemble schedule -----------------------------------------
    return assemble_schedule(
        dap=dap,
        mandatory_duration=mandatory_duration,
        mandatory_type=mandatory_type,
        disc_activities=disc_activities,
        work_start=work_start,
        first_departure=first_departure,
        before_work_flags=before_work_flags,
    )


def _write_csv_with_index(df: pl.DataFrame, path: str) -> None:
    """Write CSV with a pandas-style unnamed integer index as first column."""
    import io

    df_idx = df.with_row_index(name="")
    buf = io.StringIO()
    df_idx.write_csv(buf)
    content = buf.getvalue()
    # Polars quotes the empty column name as `""` — strip to match pandas format
    content = content.replace('"",', ",", 1)
    with open(path, "w") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CompSched synthetic schedules")
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out-attributes", default="synthetic_attributes.csv")
    parser.add_argument("--out-schedules", default="synthetic_schedules.csv")
    args = parser.parse_args()
    generate(args.attributes, args.models, args.out_attributes, args.out_schedules)


if __name__ == "__main__":
    main()
