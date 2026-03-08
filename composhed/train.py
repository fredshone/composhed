"""Train all Composhed models and save to disk."""

import argparse
import os
import time

import joblib
import numpy as np
import statsmodels.api as sm

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

    mean_home = compute_mean_home_times(records)
    print(f"  Mean home times by DAP: {mean_home}")

    # ---- Step 1: DAP MNLogit ---------------------------------------------------
    print("Step 1: Fitting DAP model...")
    y_dap = [r["dap"] for r in records]
    dap_model = DAPModel().fit(X_base, y_dap)
    print("  Done.")

    # ---- Step 2: Mandatory duration --------------------------------------------
    print("Step 2: Fitting mandatory duration model...")
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
    print("  Done.")

    # ---- Step 3: Number of disc tours -----------------------------------------
    print("Step 3: Fitting n_disc model...")
    disc_records = [r for r in records if r["dap"] in ("WD", "D")]
    if disc_records:
        X_disc, _ = encode_features(disc_records, LABEL_COLS, feature_names=feature_names)
        dap_WD_col = np.array(
            [1.0 if r["dap"] == "WD" else 0.0 for r in disc_records]
        ).reshape(-1, 1)
        rem_budget = np.array(
            [
                1440.0 - r["mandatory_duration"] - mean_home.get(r["dap"], 400.0)
                for r in disc_records
            ]
        ).reshape(-1, 1)
        X_ntours = np.hstack([X_disc, dap_WD_col, rem_budget])
        y_ntours = np.array([r["n_disc"] for r in disc_records], dtype=int)
        ntours_model = NToursModel().fit(X_ntours, y_ntours)
    else:
        ntours_model = None
    print("  Done.")

    # ---- Step 4: Activity type per slot ----------------------------------------
    print("Step 4: Fitting activity type models...")
    atype_model = ActivityTypeModel().fit(disc_slot_records, feature_names)
    print("  Done.")

    # ---- Step 5: Activity duration per type ------------------------------------
    print("Step 5: Fitting activity duration models...")
    dur_model = ActivityDurationModel().fit(disc_slot_records, feature_names)
    print("  Done.")

    # ---- Step 6: Anchor timing -------------------------------------------------
    print("Step 6: Fitting anchor timing models...")
    anchor_model = AnchorTimingModel().fit(records, feature_names)
    print("  Done.")

    # ---- Save ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        "dap": dap_model,
        "mandatory": mand_model,
        "ntours": ntours_model,
        "atype": atype_model,
        "duration": dur_model,
        "anchor": anchor_model,
        "feature_names": feature_names,
        "mean_home": mean_home,
        "label_cols": LABEL_COLS,
    }
    out_path = os.path.join(output_dir, "composhed_models.pkl")
    joblib.dump(bundle, out_path, compress=3)
    print(f"\nModels saved to {out_path} ({time.time()-t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Composhed models")
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    train(args.attributes, args.schedules, args.output_dir)


if __name__ == "__main__":
    main()
