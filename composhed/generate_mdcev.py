"""Generate synthetic schedules using the collapsed MDCEV CompSched variant."""

import click
import joblib
import numpy as np
import polars as pl
from tqdm import tqdm

from composhed.assembly import assemble_eased_mdcev_schedule
from composhed.data import (
    LABEL_COLS,
    MDCEV_LABEL_COLS,
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
    episode_model = bundle["episode_count"]
    timing_model = bundle["timing"]
    feature_names: list[str] = bundle["feature_names"]
    timing_feature_names: list[str] = bundle.get("timing_feature_names", feature_names)

    print("Loading attributes...")
    attr_df = load_attributes(attributes_path)
    pids = attr_df["pid"].to_list()
    n = len(pids)

    X_mdcev = encode_for_generation(attr_df, MDCEV_LABEL_COLS, feature_names)
    X_timing = encode_for_generation(attr_df, MDCEV_LABEL_COLS, timing_feature_names)
    attr_rows = attr_df.select(["pid"] + LABEL_COLS).to_dicts()

    print(f"Sampling MDCEV allocations for {n} persons...")
    all_allocs = mdcev_model.sample_batch(X_mdcev)

    print(f"Assembling {n} schedules...")
    sched_rows: list[dict] = []

    for i, pid in tqdm(enumerate(pids), total=n, desc="Assembling"):
        try:
            alloc = all_allocs[i]
            episode_counts = {
                atype: episode_model.sample(atype, total)
                for atype, total in alloc.items()
            }
            rows = assemble_eased_mdcev_schedule(
                alloc=alloc,
                episode_counts=episode_counts,
                x_person=X_timing[i],
                timing_model=timing_model,
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


@click.command()
@click.option("--attributes", required=True, type=click.Path(exists=True))
@click.option("--models", required=True, type=click.Path(exists=True))
@click.option("--out-attributes", default="synthetic_mdcev_attributes.csv", show_default=True)
@click.option("--out-schedules", default="synthetic_mdcev_schedules.csv", show_default=True)
@click.option("--seed", default=None, type=int, help="Random seed")
def main(
    attributes: str,
    models: str,
    out_attributes: str,
    out_schedules: str,
    seed: int | None,
) -> None:
    if seed is not None:
        import random

        np.random.seed(seed)
        random.seed(seed)
    generate(attributes, models, out_attributes, out_schedules)


if __name__ == "__main__":
    main()
