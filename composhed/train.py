"""Train all Composhed models and save to disk."""

import os
import time

import click

import joblib
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from composhed.data import (
    LABEL_COLS,
    build_training_dataset,
    compute_mean_home_times,
    encode_features,
    load_attributes,
    load_schedules,
)
from composhed.models.anchor import AnchorTimingModel
from composhed.models.atype import ActivityTypeModel
from composhed.models.dap import DAPModel
from composhed.models.duration import ActivityDurationModel
from composhed.models.mandatory import MandatoryDurationModel
from composhed.models.ntours import NToursModel


def train(attributes_path: str, schedules_path: str, output_dir: str) -> None:
    t0 = time.time()
    print("Loading data...")
    attr_df = load_attributes(attributes_path)
    sched_df = load_schedules(schedules_path)

    print("Building training dataset...")
    records, disc_slot_records = build_training_dataset(attr_df, sched_df)
    print(f"  {len(records)} persons, {len(disc_slot_records)} disc-slot records")

    # ---- Encode base label features -------------------------------------------
    X_base, feature_names = encode_features(records, LABEL_COLS)
    print(f"  {len(feature_names)} label features: {feature_names[:5]}...")
    # Fit scaler on label features for use in the anchor logistic model only
    scaler = StandardScaler().fit(X_base)

    mean_home = compute_mean_home_times(records)
    print(f"  Mean home times by DAP: {mean_home}")

    steps = [
        "DAP model",
        "Mandatory duration",
        "N disc tours",
        "Activity type",
        "Activity duration",
        "Anchor timing",
    ]
    pbar = tqdm(steps, desc="Training")

    # ---- Step 1: DAP MNLogit ---------------------------------------------------
    pbar.set_description(f"Step 1: {steps[0]}")
    print(f"\nStep 1: DAP model", flush=True)
    y_dap = [r["dap"] for r in records]
    dap_model = DAPModel().fit(X_base, y_dap)
    pbar.update(1)

    # ---- Step 2: Mandatory duration --------------------------------------------
    pbar.set_description(f"Step 2: {steps[1]}")
    print(f"\nStep 2: Mandatory duration", flush=True)
    mand_records = [r for r in records if r["dap"] in ("W", "WD")]
    if mand_records:
        X_mand, _ = encode_features(mand_records, LABEL_COLS, feature_names=feature_names)
        dap_WD_col = np.array(
            [1.0 if r["dap"] == "WD" else 0.0 for r in mand_records]
        ).reshape(-1, 1)
        X_mand = np.hstack([X_mand, dap_WD_col])
        y_mand = np.array([r["mandatory_duration"] for r in mand_records])
        mand_model = MandatoryDurationModel().fit(X_mand, y_mand)
    else:
        mand_model = None
    pbar.update(1)

    # ---- Step 3: Number of disc tours -----------------------------------------
    pbar.set_description(f"Step 3: {steps[2]}")
    print(f"\nStep 3: N disc tours", flush=True)
    disc_records = [r for r in records if r["dap"] in ("WD", "D")]
    if disc_records:
        X_disc, _ = encode_features(disc_records, LABEL_COLS, feature_names=feature_names)
        dap_WD_col = np.array(
            [1.0 if r["dap"] == "WD" else 0.0 for r in disc_records]
        ).reshape(-1, 1)
        rem_budget = np.array(
            [
                (1440.0 - r["mandatory_duration"] - mean_home.get(r["dap"], 400.0)) / 1440.0
                for r in disc_records
            ]
        ).reshape(-1, 1)
        X_ntours = np.hstack([X_disc, dap_WD_col, rem_budget])
        y_ntours = np.array([r["n_disc"] for r in disc_records], dtype=int)
        ntours_model = NToursModel().fit(X_ntours, y_ntours)
    else:
        ntours_model = None
    pbar.update(1)

    # ---- Step 4: Activity type per slot ----------------------------------------
    pbar.set_description(f"Step 4: {steps[3]}")
    print(f"\nStep 4: Activity type", flush=True)
    atype_model = ActivityTypeModel().fit(disc_slot_records, feature_names)
    pbar.update(1)

    # ---- Step 5: Activity duration per type ------------------------------------
    pbar.set_description(f"Step 5: {steps[4]}")
    print(f"\nStep 5: Activity duration", flush=True)
    dur_model = ActivityDurationModel().fit(disc_slot_records, feature_names)
    pbar.update(1)

    # ---- Step 6: Anchor timing -------------------------------------------------
    pbar.set_description(f"Step 6: {steps[5]}")
    print(f"\nStep 6: Anchor timing", flush=True)
    anchor_model = AnchorTimingModel().fit(records, feature_names, scaler)
    pbar.update(1)
    pbar.close()

    # ---- Save ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        "dap": dap_model,
        "mandatory": mand_model,
        "ntours": ntours_model,
        "atype": atype_model,
        "duration": dur_model,
        "anchor": anchor_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "mean_home": mean_home,
        "label_cols": LABEL_COLS,
    }
    out_path = os.path.join(output_dir, "composhed_models.pkl")
    joblib.dump(bundle, out_path, compress=3)
    print(f"\nModels saved to {out_path} ({time.time()-t0:.1f}s)")


@click.command()
@click.option("--attributes", required=True, type=click.Path(exists=True))
@click.option("--schedules", required=True, type=click.Path(exists=True))
@click.option("--output-dir", default="models", show_default=True)
@click.option("--seed", default=None, type=int, help="Random seed")
def main(attributes: str, schedules: str, output_dir: str, seed: int | None) -> None:
    if seed is not None:
        import random
        np.random.seed(seed)
        random.seed(seed)
    train(attributes, schedules, output_dir)


if __name__ == "__main__":
    main()
