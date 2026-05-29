"""Train collapsed MDCEV CompSched variant and save to disk.

Pipeline:
  1. MDCEVModel  — 8-type time allocation (home, work, education, shop, visit, escort, medical, other)
  2. EpisodeCountModel — per-type episode count given total allocated minutes
  3. StartTimeModel   — per-enumerated-key start time regression (for assembly ordering)
"""

import os
import time

import click
import joblib
from tqdm import tqdm

from composhed.data import (
    ACTIVITY_TYPES,
    LABEL_COLS,
    MDCEV_LABEL_COLS,
    build_enumerated_dataset,
    build_training_dataset,
    derive_type_records,
    encode_features,
    load_attributes,
    load_schedules,
)
from composhed.models.episode_count import EpisodeCountModel
from composhed.models.mdcev import MDCEVModel
from composhed.models.timing import StartTimeModel


def _print_mdcev_validation(metrics: dict, types: list[str], label: str = "") -> None:
    header = "  MDCEV — participation rates and mean durations (actual vs predicted)"
    if label:
        header += f"  [{label}]"
    print(header)
    if "pseudo_r2" in metrics:
        ll = metrics.get("log_likelihood", float("nan"))
        print(f"    pseudo-R² {metrics['pseudo_r2']:.3f}   log-likelihood {ll:.1f}")
    hdr = f"    {'type':<14}  {'part_act':>8}  {'part_pred':>9}  {'dur_act':>7}  {'dur_pred':>8}"
    print(hdr)
    for atype in types:
        if atype not in metrics:
            continue
        m = metrics[atype]
        print(
            f"    {atype:<14}  {m['participation_actual']:>8.3f}  "
            f"{m['participation_pred']:>9.3f}  "
            f"{m['mean_duration_actual']:>7.0f}  {m['mean_duration_pred']:>8.0f}"
        )


def _print_calibration_offsets(offsets: dict[str, float], types: list[str]) -> None:
    print("  ASC calibration offsets")
    for atype in types:
        v = offsets.get(atype, 0.0)
        bar = ("+" if v >= 0 else "") + f"{v:+.3f}"
        print(f"    {atype:<14}  {bar}")


def _print_episode_validation(metrics: dict) -> None:
    print("  EpisodeCount — mean episodes per type (actual vs predicted)")
    hdr = f"    {'type':<14}  {'n':>6}  {'mean_act':>8}  {'mean_pred':>9}  {'mae':>6}"
    print(hdr)
    for atype, m in sorted(metrics.items()):
        print(
            f"    {atype:<14}  {m['n']:>6}  {m['mean_actual']:>8.2f}  "
            f"{m['mean_pred']:>9.2f}  {m['mae']:>6.3f}"
        )


def train(
    attributes_path: str,
    schedules_path: str,
    output_dir: str,
    max_records: int | None = None,
    cap_percentile: float = 90.0,
    calibrate: bool = True,
    fix_gamma: bool = True,
) -> None:
    t0 = time.time()
    print("Loading data...")
    attr_df = load_attributes(attributes_path)
    sched_df = load_schedules(schedules_path)

    # ---- Build datasets -------------------------------------------------------
    print("Building training datasets...")
    baseline_records, _ = build_training_dataset(attr_df, sched_df)
    print(f"  {len(baseline_records)} persons (baseline records)")

    type_records = derive_type_records(baseline_records)
    print(f"  {len(type_records)} type records for MDCEV + EpisodeCount")

    print("Building enumerated dataset (for timing model)...")
    enum_records, max_counts = build_enumerated_dataset(
        attr_df, sched_df, cap_percentile=cap_percentile
    )
    if max_counts:
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(max_counts.items()))
        print(f"  Max counts per type (cap @ {cap_percentile}th pct): {summary}")
    print(f"  {len(enum_records)} enumerated records for timing model")

    # Optional subsample for faster MDCEV estimation
    if max_records is not None and len(type_records) > max_records:
        import random
        type_records = random.sample(type_records, max_records)
        print(f"  Subsampled to {max_records} type records for MDCEV estimation")

    # ---- Encode features -------------------------------------------------------
    X_mdcev, feature_names = encode_features(type_records, MDCEV_LABEL_COLS)
    print(f"  {len(feature_names)} MDCEV label features")

    X_timing, timing_feature_names = encode_features(enum_records, MDCEV_LABEL_COLS)
    print(f"  {len(timing_feature_names)} timing label features")

    steps = ["MDCEV model (biogeme)", "Episode count model", "Start time model"]
    pbar = tqdm(steps, desc="Training")

    # ---- Step 1: MDCEV (8-type totals) ----------------------------------------
    pbar.set_description(f"Step 1: {steps[0]}")
    print(f"\nStep 1: MDCEV (K={len(ACTIVITY_TYPES)})", flush=True)
    mdcev_model = MDCEVModel().fit(
        type_records, feature_names, types=ACTIVITY_TYPES, X=X_mdcev, fix_gamma=fix_gamma
    )
    _print_mdcev_validation(mdcev_model.validate(type_records, X_mdcev), ACTIVITY_TYPES, label="pre-calibration")
    print("  Biogeme estimates (intercepts / gammas / scale)")
    print(f"    scale = {mdcev_model._scale:.4f}")
    hdr2 = f"    {'type':<14}  {'intercept':>10}  {'gamma':>10}"
    print(hdr2)
    for atype, intercept, gamma in zip(
        mdcev_model.TYPES, mdcev_model._intercepts, mdcev_model._gammas
    ):
        print(f"    {atype:<14}  {intercept:>10.4f}  {gamma:>10.2f}")
    pbar.update(1)

    # ---- Step 1b: ASC calibration ---------------------------------------------
    if calibrate:
        print("\nStep 1b: ASC calibration", flush=True)
        target_participation = {
            atype: sum(1 for r in type_records if r["total_durations"].get(atype, 0.0) > 0)
            / len(type_records)
            for atype in ACTIVITY_TYPES
        }
        offsets = mdcev_model.calibrate(X_mdcev, target_participation, verbose=True)
        _print_calibration_offsets(offsets, ACTIVITY_TYPES)
        _print_mdcev_validation(mdcev_model.validate(type_records, X_mdcev), ACTIVITY_TYPES, label="post-calibration")

    # ---- Step 2: Episode count model ------------------------------------------
    pbar.set_description(f"Step 2: {steps[1]}")
    print(f"\nStep 2: Episode count model", flush=True)
    episode_model = EpisodeCountModel().fit(type_records)
    _print_episode_validation(episode_model.validate(type_records))
    pbar.update(1)

    # ---- Step 3: Start time model (for assembly ordering) ---------------------
    pbar.set_description(f"Step 3: {steps[2]}")
    print(f"\nStep 3: Start time model", flush=True)
    timing_model = StartTimeModel().fit(enum_records, timing_feature_names, X=X_timing)
    pbar.update(1)
    pbar.close()

    # ---- Save -----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        "mdcev": mdcev_model,
        "episode_count": episode_model,
        "timing": timing_model,
        "feature_names": feature_names,
        "timing_feature_names": timing_feature_names,
        "label_cols": LABEL_COLS,
        "activity_types": ACTIVITY_TYPES,
    }
    out_path = os.path.join(output_dir, "mdcev_models.pkl")
    joblib.dump(bundle, out_path, compress=3)
    print(f"\nModels saved to {out_path} ({time.time() - t0:.1f}s)")


@click.command()
@click.option("--attributes", required=True, type=click.Path(exists=True))
@click.option("--schedules", required=True, type=click.Path(exists=True))
@click.option("--output-dir", default="models", show_default=True)
@click.option("--max-records", default=None, show_default=True, type=int)
@click.option(
    "--cap-percentile",
    default=90.0,
    show_default=True,
    type=float,
    help="Percentile cap for timing model enumeration (does not affect MDCEV)",
)
@click.option(
    "--calibrate/--no-calibrate",
    default=True,
    show_default=True,
    help="Run ASC calibration after MDCEV estimation to match training participation rates",
)
@click.option(
    "--fix-gamma/--free-gamma",
    default=True,
    show_default=True,
    help=(
        "Fix gamma to conditional mean duration per type (recommended). "
        "Free gamma allows Biogeme to estimate satiation jointly but tends "
        "to produce degenerate solutions with this dataset."
    ),
)
@click.option("--seed", default=None, type=int, help="Random seed")
def main(
    attributes: str,
    schedules: str,
    output_dir: str,
    max_records: int | None,
    cap_percentile: float,
    calibrate: bool,
    fix_gamma: bool,
    seed: int | None,
) -> None:
    if seed is not None:
        import random
        import numpy as np

        np.random.seed(seed)
        random.seed(seed)
    train(attributes, schedules, output_dir, max_records, cap_percentile, calibrate, fix_gamma)


if __name__ == "__main__":
    main()
