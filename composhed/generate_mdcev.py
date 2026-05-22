"""Generate synthetic schedules using the MDCEV CompSched variant."""

import click
import joblib
import numpy as np
import polars as pl
from tqdm import tqdm

from composhed.assembly import assemble_mdcev_schedule
from composhed.data import (
    LABEL_COLS,
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
    episode_model = bundle["episodes"]
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
                alloc=all_allocs[i],
                x_label=X_all[i],
                employment=employment,
                anchor_model=anchor_model,
                episode_model=episode_model,
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
    alloc: dict[str, float],
    x_label: np.ndarray,
    employment: str,
    anchor_model,
    episode_model,
) -> list[dict]:
    """Assemble a schedule for one person from a pre-sampled MDCEV allocation."""
    # Derive DAP from which activity types received > 1 min
    participated = [atype for atype, dur in alloc.items() if atype != "home" and dur > 1.0]
    dap = classify_dap(participated)

    mandatory_type = "education" if dap in ("E", "ED") else "work"

    return assemble_mdcev_schedule(
        alloc=alloc,
        dap=dap,
        mandatory_type=mandatory_type,
        x_label=x_label,
        employment=employment,
        anchor_model=anchor_model,
        episode_model=episode_model,
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
