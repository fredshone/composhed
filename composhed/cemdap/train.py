"""Train the CEMDAP-aligned variant and save ``cemdap_models.pkl``."""

import time

import click
import numpy as np

from composhed.cemdap.chain import HOME_INTER, CemdapChain, record_trace, trace_stats
from composhed.cemdap.data import build_records
from composhed.cemdap.models import COUNT_POSITIONS, PURPOSES, CemdapModels
from composhed.common.bundle import save_bundle
from composhed.common.calibration import location_accessors, offset_accessors
from composhed.common.cli import make_rng, train_options
from composhed.common.report import ValidationReport
from composhed.common.runner import (
    encode_labels,
    load_training_data,
    output_paths,
    run_calibration,
    scale_accessors,
    targets_from_specs,
)

VARIANT = "cemdap"


def _calibration_specs(m: CemdapModels) -> dict:
    specs = {
        "work_participation": ("share", *offset_accessors(m.work)),
        "education_participation": ("share", *offset_accessors(m.edu)),
        "nm_tours": ("share", *offset_accessors(m.nm_tours)),
        "tour_size": ("share", *offset_accessors(m.tour_size)),
        "stop_purpose": ("share", *offset_accessors(m.stop_purpose)),
    }
    for k, model in m.counts.items():
        specs[f"stops_{k[2:]}"] = ("share", *offset_accessors(model))
    if m.start is not None:
        dur = m.indep_duration if m.independent_timing else m.duration
        specs["mand_duration_mean"] = ("mean", *location_accessors(dur))
        specs["mand_duration_spread"] = ("mean", *scale_accessors(dur))
        if not m.independent_timing:
            specs["mand_start_mean"] = ("mean", *location_accessors(m.start))
            specs["mand_start_spread"] = ("mean", *scale_accessors(m.start))
    seen = set()
    for p in PURPOSES:
        model = m.stop_duration.get(p)
        if model is not None and id(model) not in seen:
            seen.add(id(model))
            specs[f"stop_duration_{p}"] = ("mean", *location_accessors(model))
    if m.home is not None:
        specs["home_stay_mean"] = ("mean", *location_accessors(m.home))
    if m.first_departure is not None:
        specs["first_departure_mean"] = ("mean", *location_accessors(m.first_departure))
    return specs


def train(
    attributes: str,
    schedules: str,
    output_dir: str,
    seed: int | None = None,
    holdout_fraction: float = 0.0,
    split_seed: int = 0,
    calibrate: bool = True,
    calib_iters: int = 8,
    calib_sample: int = 20000,
    independent_timing: bool = False,
    consistency: str = "constraint",
    max_resample: int = 50,
) -> str:
    t0 = time.time()
    print("Loading data...")
    attr_df, sched_by_pid, train_pids, hold_pids = load_training_data(
        attributes, schedules, holdout_fraction, split_seed)

    print("Extracting CEMDAP day structures (mandatory tours collapsed)...")
    records, counters = build_records(attr_df, sched_by_pid, train_pids)
    counters.log()
    X, feature_names = encode_labels([r.labels for r in records])
    hold_records = []
    if hold_pids:
        hold_records, _ = build_records(attr_df, sched_by_pid, hold_pids)
    X_hold = encode_labels([r.labels for r in hold_records], feature_names)[0] if hold_records else None

    print("\nFitting CEMDAP components...")
    models = CemdapModels(independent_timing=independent_timing).fit(records, X, feature_names)

    print("\nValidation (in-sample)")
    report = ValidationReport()
    train_metrics = models.validate(records, X)
    hold_metrics = models.validate(hold_records, X_hold) if hold_records else {}
    for name, metrics in train_metrics.items():
        report.add(name, metrics, hold_metrics.get(name))
    gaps = [h["duration"] for r in records for h in r.home_stays]
    report.info = {
        "extraction": counters.as_dict(),
        "home_inter": HOME_INTER,
        "between_tour_home_p05": float(np.percentile(gaps, 5)) if gaps else None,
        "n_train": len(records),
        "n_holdout": len(hold_records),
        "aic": {k: getattr(v, "aic_all_", None) for k, v in
                {"start": models.start, "duration": models.duration, "home": models.home}.items() if v is not None},
    }

    chain = CemdapChain(models, consistency=consistency, max_resample=max_resample)
    calib = None
    observed = trace_stats([record_trace(r) for r in records], independent_timing)
    if calibrate:
        print("\nCalibration (targets from training data)")
        targets = targets_from_specs(observed, _calibration_specs(models))
        attr_rows = [{"pid": r.pid, **r.labels} for r in records]
        calib = run_calibration(chain, targets, lambda tr: trace_stats(tr, independent_timing), X, attr_rows,
                                calib_sample, calib_iters, seed)
        report.info["calibration"] = calib.to_dict()

    config = {
        "independent_timing": independent_timing,
        "consistency": consistency,
        "max_resample": max_resample,
        "home_inter": HOME_INTER,
        "calibrate": calibrate,
        "calib_iters": calib_iters,
        "calib_sample": calib_sample,
        "holdout_fraction": holdout_fraction,
        "split_seed": split_seed,
        "seed": seed,
        "time_resolution": 1,
    }
    path, val_path = output_paths(output_dir, VARIANT)
    save_bundle(output_dir, VARIANT, {"cemdap": models}, feature_names, config,
                calibration={"targets": observed, "report": calib.to_dict() if calib else None},
                coefficients=models.coef_tables())
    report.write(val_path)
    print(f"\nModels saved to {path} ({time.time() - t0:.1f}s); validation written to {val_path}")
    return path


@click.command()
@train_options
def main(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, calibrate, calib_iters,
         calib_sample, independent_timing, consistency, max_resample) -> None:
    make_rng(seed)
    train(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, calibrate, calib_iters,
          calib_sample, independent_timing, consistency, max_resample)


if __name__ == "__main__":
    main()
