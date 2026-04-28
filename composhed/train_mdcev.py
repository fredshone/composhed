"""Train MDCEV CompSched variant and save to disk."""

import argparse
import os
import time

import joblib
from tqdm import tqdm

from composhed.data import (
    LABEL_COLS,
    build_training_dataset,
    encode_features,
    load_attributes,
    load_schedules,
)
from composhed.models.anchor import AnchorTimingModel
from composhed.models.mdcev import MDCEVModel


def train(attributes_path: str, schedules_path: str, output_dir: str, max_records: int | None = None) -> None:
    t0 = time.time()
    print("Loading data...")
    attr_df = load_attributes(attributes_path)
    sched_df = load_schedules(schedules_path)

    print("Building training dataset...")
    records, _ = build_training_dataset(attr_df, sched_df)
    print(f"  {len(records)} persons")

    if max_records is not None and len(records) > max_records:
        import random
        random.seed(42)
        records = random.sample(records, max_records)
        print(f"  Subsampled to {max_records} persons for MDCEV estimation")

    X_base, feature_names = encode_features(records, LABEL_COLS)
    print(f"  {len(feature_names)} label features: {feature_names[:5]}...")

    steps = ["MDCEV model (biogeme, slow)", "Anchor timing"]
    pbar = tqdm(steps, desc="Training")

    pbar.set_description(f"Step 1: {steps[0]}")
    mdcev_model = MDCEVModel().fit(records, feature_names, X=X_base)
    pbar.update(1)

    pbar.set_description(f"Step 2: {steps[1]}")
    anchor_model = AnchorTimingModel().fit(records, feature_names)
    pbar.update(1)
    pbar.close()

    # ---- Save -------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        "mdcev": mdcev_model,
        "anchor": anchor_model,
        "feature_names": feature_names,
        "label_cols": LABEL_COLS,
    }
    out_path = os.path.join(output_dir, "mdcev_models.pkl")
    joblib.dump(bundle, out_path, compress=3)
    print(f"\nModels saved to {out_path} ({time.time() - t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MDCEV CompSched variant")
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--max-records", type=int, default=5000)
    args = parser.parse_args()
    train(args.attributes, args.schedules, args.output_dir, args.max_records)


if __name__ == "__main__":
    main()
