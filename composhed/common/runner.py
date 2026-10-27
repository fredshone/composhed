"""Shared train / generate / calibrate plumbing for the source-aligned variants."""

import os
import time

import numpy as np
from tqdm import tqdm

from composhed.common.calibration import CalibrationReport, Target, calibrate
from composhed.common.report import GenerationReport
from composhed.common.schedule import home_day
from composhed.common.split import holdout_split
from composhed.common.tours import group_schedules
from composhed.data import LABEL_COLS, encode_features, encode_for_generation, load_attributes, load_schedules


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


def simulate_traces(chain, X: np.ndarray, attr_rows: list[dict], seed: int) -> list[dict]:
    """Run the chain on the given persons with common random numbers; return traces."""
    rng = np.random.default_rng(seed)
    report = GenerationReport()
    traces = []
    for x, attrs in zip(X, attr_rows):
        try:
            traces.append(chain.generate(x, attrs, rng, report)[1])
        except Exception:  # noqa: BLE001
            continue
    return traces


def run_calibration(chain, targets: list[Target], stats_fn, X: np.ndarray, attr_rows: list[dict],
                    sample: int, iters: int, seed: int | None) -> CalibrationReport:
    """SR-15/16/17: calibrate on a sample of *training* persons."""
    rng = np.random.default_rng(seed)
    n = len(attr_rows)
    reps = max(1, int(np.ceil(sample / max(n, 1)))) if n < sample else 1
    idx = rng.choice(n, size=min(sample, n), replace=False) if n > sample else np.tile(np.arange(n), reps)
    Xs = X[idx]
    rows = [attr_rows[i] for i in idx]
    sim_seed = int(rng.integers(2**31))
    t0 = time.time()

    def simulate():
        return stats_fn(simulate_traces(chain, Xs, rows, sim_seed))

    print(f"  calibrating on {len(idx):,} simulated persons (max {iters} iterations)...")
    rep = calibrate(targets, simulate, max_iter=iters)
    print(f"  calibration done in {time.time() - t0:.0f}s")
    rep.print()
    return rep


def targets_from_specs(observed: dict[str, dict], specs: dict[str, tuple]) -> list[Target]:
    """Build Target objects for every spec whose observed statistic exists.

    specs: name -> (kind, get, set[, lr_scale])
    """
    out = []
    for name, spec in specs.items():
        if name in observed and observed[name]:
            out.append(Target(name, observed[name], *spec))
    return out


def scale_accessors(model):
    """get/set closures exposing log(scale multiplier) of a duration model (spread targets)."""

    def get() -> dict:
        return {"mean": float(np.log(model.scale_mult))}

    def set_(d: dict) -> None:
        model.scale_mult = float(np.exp(d["mean"]))

    return get, set_


def output_paths(output_dir: str, variant: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{variant}_models.pkl"), os.path.join(output_dir, "validation.json")
