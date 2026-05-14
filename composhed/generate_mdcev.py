"""Generate synthetic schedules using the MDCEV CompSched variant."""

import click
import joblib
import numpy as np
import polars as pl
from tqdm import tqdm

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

    print(f"Sampling MDCEV allocations for {n} persons...")
    all_allocs = mdcev_model.sample_batch(X_all)

    print(f"Assembling {n} schedules...")
    sched_rows: list[dict] = []

    for i, pid in tqdm(enumerate(pids), total=n, desc="Assembling"):
        employment = str(attr_rows[i]["employment"])

        try:
            rows = _generate_one(
                pid=pid,
                alloc=all_allocs[i],
                x_label=X_all[i],
                employment=employment,
                anchor_model=anchor_model,
            )
        except Exception as exc:
            tqdm.write(f"  WARNING pid={pid}: {exc}; using fallback H schedule")
            rows = [{"act": "home", "start": 0, "end": 1440, "duration": 1440}]

        for row in rows:
            row["pid"] = pid
            sched_rows.append(row)

    print(f"Writing {out_attributes} ...")
    out_attr_df = pl.DataFrame(attr_rows).select(["pid"] + LABEL_COLS)
    _write_csv_with_index(out_attr_df, out_attributes)

    print(f"Writing {out_schedules} ...")
    sched_df = pl.DataFrame(sched_rows).select(["pid", "act", "start", "end", "duration"])
    _write_csv_with_index(sched_df, out_schedules)

    print("Done.")


def _generate_one(
    pid: int,
    alloc: dict[str, float],
    x_label: np.ndarray,
    employment: str,
    anchor_model,
) -> list[dict]:
    """Generate a schedule for one person given a pre-sampled MDCEV allocation."""

    t = alloc

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
        work_start = anchor_model.sample_work_start(employment)
        work_start = float(np.clip(work_start, 0.0, 1440.0 - mandatory_duration - 60.0))

    if dap == "D":
        first_departure = anchor_model.sample_first_departure(employment)

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


@click.command()
@click.option("--attributes", required=True, type=click.Path(exists=True))
@click.option("--models", required=True, type=click.Path(exists=True))
@click.option("--out-attributes", default="synthetic_mdcev_attributes.csv", show_default=True)
@click.option("--out-schedules", default="synthetic_mdcev_schedules.csv", show_default=True)
@click.option("--seed", default=None, type=int, help="Random seed")
def main(attributes: str, models: str, out_attributes: str, out_schedules: str, seed: int | None) -> None:
    if seed is not None:
        import random
        np.random.seed(seed)
        random.seed(seed)
    generate(attributes, models, out_attributes, out_schedules)


if __name__ == "__main__":
    main()
