"""Shared train / generate plumbing for the source-aligned variants."""

import os

import numpy as np
from tqdm import tqdm

from composhed.common.report import GenerationReport
from composhed.common.schedule import home_day
from composhed.common.split import holdout_split
from composhed.common.tours import group_schedules
from composhed.data import (
    LABEL_COLS,
    encode_features,
    encode_for_generation,
    load_attributes,
    load_schedules,
)


def load_training_data(attributes: str, schedules: str, holdout_fraction: float, split_seed: int):
    """Returns (attr_df, schedules grouped by pid, train pids, holdout pids)."""
    attr_df = load_attributes(attributes)
    sched_by_pid = group_schedules(load_schedules(schedules))
    pids = [p for p in attr_df["pid"].to_list() if p in sched_by_pid]
    train_pids, hold_pids = holdout_split(pids, holdout_fraction, split_seed)
    print(f"  {len(train_pids):,} training persons, {len(hold_pids):,} held out")
    return attr_df, sched_by_pid, train_pids, hold_pids


def encode_labels(label_dicts: list[dict], feature_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if not label_dicts:
        return np.zeros((0, len(feature_names or []))), list(feature_names or [])
    return encode_features(label_dicts, LABEL_COLS, feature_names=feature_names)


def generate_population(chain, attributes: str, feature_names: list[str], rng: np.random.Generator,
                        report: GenerationReport) -> tuple[list[dict], list[dict]]:
    """Generate one schedule per person in input pid order (SR-20); failures fall back to home (I7)."""
    attr_df = load_attributes(attributes)
    X = encode_for_generation(attr_df, LABEL_COLS, feature_names)
    attr_rows = attr_df.select(["pid"] + LABEL_COLS).to_dicts()
    sched_rows: list[dict] = []
    for i, attrs in tqdm(enumerate(attr_rows), total=len(attr_rows), desc="Generating"):
        report.n_persons += 1
        rescales_before = report.rescales
        try:
            rows, _ = chain.generate(X[i], attrs, rng, report)
            rescaled = report.rescales > rescales_before
        except Exception as exc:  # noqa: BLE001 — I7: never abort the run
            report.fallback(type(exc).__name__ + (f": {exc}" if str(exc) else ""))
            rows, rescaled = home_day(), False
            report.persons_repaired += 1
        if rescaled:
            report.persons_rescaled += 1
            report.persons_repaired += 1
        for r in rows:
            sched_rows.append({"pid": attrs["pid"], **r})
    return attr_rows, sched_rows


def output_paths(output_dir: str, variant: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{variant}_models.pkl"), os.path.join(output_dir, "validation.json")
