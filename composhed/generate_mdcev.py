"""Generate synthetic schedules using the MDCEV CompSched variant."""

import argparse

import joblib
import numpy as np
import polars as pl

from composhed.assembly import assemble_schedule
from composhed.data import (
    LABEL_COLS,
    MANDATORY_ACTS,
    DISC_ACTS,
    classify_dap,
    encode_for_generation,
    load_attributes,
)
from composhed.generate import _write_csv_with_index


def generate(
    attributes_path: str,
    models_path: str,
    out_attributes: str,
    out_schedules: str,
) -> None:
    print("Loading models...")
    bundle = joblib.load(models_path)
    mdcev_model = bundle["mdcev"]
    anchor_model = bundle["anchor"]
    feature_names: list[str] = bundle["feature_names"]

    print("Loading attributes...")
    attr_df = load_attributes(attributes_path)
    pids = attr_df["pid"].to_list()
    n = len(pids)

    X_all = encode_for_generation(attr_df, LABEL_COLS, feature_names)
    attr_rows = attr_df.select(["pid"] + LABEL_COLS).to_dicts()

    print(f"Generating {n} schedules...")
    sched_rows: list[dict] = []

    for i, pid in enumerate(pids):
        x_label = X_all[i]
        work_status = str(attr_rows[i]["work_status"])

        try:
            rows = _generate_one(
                pid=pid,
                x_label=x_label,
                work_status=work_status,
                mdcev_model=mdcev_model,
                anchor_model=anchor_model,
            )
        except Exception as exc:
            print(f"  WARNING pid={pid}: {exc}; using fallback H schedule")
            rows = [{"act": "home", "start": 0, "end": 1440, "duration": 1440}]

        for row in rows:
            row["pid"] = pid
            sched_rows.append(row)

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n}")

    print(f"Writing {out_attributes} ...")
    out_attr_df = pl.DataFrame(attr_rows).select(["pid"] + LABEL_COLS)
    _write_csv_with_index(out_attr_df, out_attributes)

    print(f"Writing {out_schedules} ...")
    sched_df = pl.DataFrame(sched_rows).select(["pid", "act", "start", "end", "duration"])
    _write_csv_with_index(sched_df, out_schedules)

    print("Done.")


def _generate_one(
    pid: int,
    x_label: np.ndarray,
    work_status: str,
    mdcev_model,
    anchor_model,
) -> list[dict]:
    """Generate a schedule for one person using the MDCEV allocation."""

    # ---- 1. Sample MDCEV time allocation ------------------------------------
    t = mdcev_model.sample(x_label)  # {type: minutes_float}

    # ---- 2. Derive chosen activities (threshold > 1 min) --------------------
    chosen = [(atype, dur) for atype, dur in t.items() if atype != "home" and dur > 1.0]

    # ---- 3. Derive DAP -------------------------------------------------------
    act_types = [a for a, _ in chosen]
    dap = classify_dap(act_types)

    if dap == "H":
        return assemble_schedule("H", 0, "home", [], None, None, [])

    # ---- 4. Split mandatory / discretionary ---------------------------------
    mandatory_acts = [(a, dur) for a, dur in chosen if a in MANDATORY_ACTS]
    disc_activities = [(a, max(1, round(dur))) for a, dur in chosen if a in DISC_ACTS]

    # Combine work + education into a single mandatory block; use dominant type
    mandatory_duration = sum(dur for _, dur in mandatory_acts)
    if mandatory_acts:
        mandatory_type = max(mandatory_acts, key=lambda x: x[1])[0]
    else:
        mandatory_type = "work"
    mandatory_duration = max(1, round(mandatory_duration))

    # ---- 5. Anchor timing ---------------------------------------------------
    work_start = None
    first_departure = None

    if dap in ("W", "WD"):
        work_start = anchor_model.sample_work_start(work_status)
        work_start = float(np.clip(work_start, 0.0, 1440.0 - mandatory_duration - 60.0))

    if dap == "D":
        first_departure = anchor_model.sample_first_departure(work_status)

    # ---- 6. Before-work flags (WD only) -------------------------------------
    before_work_flags: list[bool] = []
    if dap == "WD" and disc_activities and work_start is not None:
        before_work_flags = anchor_model.sample_before_work_flags(
            x_label, disc_activities, work_start
        )

    # ---- 7. Assemble schedule -----------------------------------------------
    return assemble_schedule(
        dap=dap,
        mandatory_duration=mandatory_duration,
        mandatory_type=mandatory_type,
        disc_activities=disc_activities,
        work_start=work_start,
        first_departure=first_departure,
        before_work_flags=before_work_flags,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic schedules using MDCEV CompSched variant"
    )
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--out-attributes", default="synthetic_mdcev_attributes.csv")
    parser.add_argument("--out-schedules", default="synthetic_mdcev_schedules.csv")
    args = parser.parse_args()
    generate(args.attributes, args.models, args.out_attributes, args.out_schedules)


if __name__ == "__main__":
    main()
