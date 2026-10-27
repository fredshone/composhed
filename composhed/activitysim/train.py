"""Train the ActivitySim-aligned variant and save ``activitysim_models.pkl``."""

import time
from collections import Counter

import click

from composhed.activitysim.chain import AsimChain, record_trace, trace_stats
from composhed.activitysim.data import build_records
from composhed.activitysim.models import GROUPS, AsimModels
from composhed.common.bundle import save_bundle
from composhed.common.calibration import location_accessors, offset_accessors
from composhed.common.cli import make_rng, train_options
from composhed.common.report import ValidationReport
from composhed.common.runner import encode_labels, load_training_data, output_paths, run_calibration, targets_from_specs
from composhed.daysim.chain import array_accessors

VARIANT = "activitysim"


def _calibration_specs(m: AsimModels, timing: str) -> dict:
    specs = {}
    for prefix, seg in (("cdap", m.cdap), ("mtf", m.mtf), ("nmtf", m.nmtf), ("stopfreq", m.stopfreq)):
        if seg is None:
            continue
        for key, model in seg.models.items():
            specs[f"{prefix}:{key}"] = ("share", *offset_accessors(model))
    for g in GROUPS:
        tod = m.tod.get(g)
        if tod is None:
            continue
        if timing == "tod":
            specs[f"tod_start_{g}"] = ("share", *array_accessors(tod.start_offset), 1 / 3)
            specs[f"tod_end_{g}"] = ("share", *array_accessors(tod.end_offset), 1 / 3)
            specs[f"tod_duration_{g}"] = ("share", *array_accessors(tod.dur_offset), 1 / 3)
        else:
            specs[f"tour_duration_mean_{g}"] = ("mean", *location_accessors(m.indep[g].dur_))
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
    time_resolution: int = 60,
    segmentation: bool = True,
    n_min: int = 200,
    min_support: int = 30,
) -> str:
    t0 = time.time()
    print("Loading data...")
    attr_df, sched_by_pid, train_pids, hold_pids = load_training_data(
        attributes, schedules, holdout_fraction, split_seed)
    print("Extracting ActivitySim tours (multiple mandatory tours kept)...")
    unmapped: Counter = Counter()
    records, counters, caps = build_records(attr_df, sched_by_pid, train_pids, unmapped)
    counters.log()
    ptypes = Counter(r.ptype for r in records)
    print(f"  person types: {dict(ptypes)}")
    if unmapped:
        print(f"  unmapped age × employment (→ NONWORK): {dict(unmapped)}")
    if caps:
        print(f"  non-mandatory tours removed from frequency counts (cap 4): {caps}")
    X, feature_names = encode_labels([r.labels for r in records])
    hold_records = build_records(attr_df, sched_by_pid, hold_pids)[0] if hold_pids else []
    X_hold = encode_labels([r.labels for r in hold_records], feature_names)[0] if hold_records else None

    print(f"\nFitting ActivitySim components (time resolution {time_resolution} min, "
          f"segmentation {'on' if segmentation else 'off'})...")
    models = AsimModels(time_resolution=time_resolution, segmented=segmentation, n_min=n_min,
                        min_support=min_support).fit(records, X, feature_names)
    for name, seg in (("CDAP", models.cdap), ("MTF", models.mtf), ("NMTF", models.nmtf)):
        if seg is not None and seg.pooled_segments:
            print(f"  {name}: segments pooled (n < {n_min}): {seg.pooled_segments}")

    print("\nValidation (in-sample)")
    report = ValidationReport()
    hold = models.validate(hold_records, X_hold) if hold_records else {}
    for name, metrics in models.validate(records, X).items():
        report.add(name, metrics, hold.get(name))
    report.info = {"extraction": counters.as_dict(), "person_types": dict(ptypes),
                   "unmapped": {f"{a}|{e}": n for (a, e), n in unmapped.items()}, "nm_caps": caps,
                   "n_train": len(records), "n_holdout": len(hold_records),
                   "nm_alternatives": len(models.nm_vectors), "nm_persons_mapped": models.nm_mapped,
                   "trip_purpose_lookup": models.trip_purpose.as_dict()}

    timing = "independent" if independent_timing else "tod"
    chain = AsimChain(models, timing=timing, consistency=consistency, max_resample=max_resample)
    observed = trace_stats([record_trace(r, models) for r in records], models, timing)
    calib = None
    if calibrate:
        print("\nCalibration (targets from training data; constants per segment)")
        targets = targets_from_specs(observed, _calibration_specs(models, timing))
        attr_rows = [{"pid": r.pid, **r.labels} for r in records]
        calib = run_calibration(chain, targets, lambda tr: trace_stats(tr, models, timing), X, attr_rows,
                                calib_sample, calib_iters, seed)
        report.info["calibration"] = calib.to_dict()

    config = {"timing": timing, "consistency": consistency, "max_resample": max_resample,
              "time_resolution": time_resolution, "segmentation": segmentation, "n_min": n_min,
              "min_support": min_support, "home_inter": models.home_inter, "calibrate": calibrate,
              "calib_iters": calib_iters, "calib_sample": calib_sample, "holdout_fraction": holdout_fraction,
              "split_seed": split_seed, "seed": seed}
    path, val_path = output_paths(output_dir, VARIANT)
    save_bundle(output_dir, VARIANT, {"activitysim": models}, feature_names, config,
                calibration={"targets": observed, "report": calib.to_dict() if calib else None})
    report.write(val_path)
    print(f"\nModels saved to {path} ({time.time() - t0:.1f}s); validation written to {val_path}")
    return path


@click.command()
@train_options
@click.option("--time-resolution", type=click.Choice(["15", "30", "60"]), default="60", show_default=True,
              help="ASIM-10 / AB-4: tdd period length in minutes after the 00:00–05:00 early bin")
@click.option("--segmentation/--no-segmentation", default=True, show_default=True,
              help="AB-5: person-type segment-specific coefficients")
@click.option("--n-min", default=200, show_default=True, type=int, help="Minimum segment size before pooling")
@click.option("--min-support", default=30, show_default=True, type=int,
              help="Support threshold for non-mandatory frequency alternatives")
def main(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, calibrate, calib_iters,
         calib_sample, independent_timing, consistency, max_resample, time_resolution, segmentation, n_min,
         min_support) -> None:
    make_rng(seed)
    train(attributes, schedules, output_dir, seed, holdout_fraction, split_seed, calibrate, calib_iters,
          calib_sample, independent_timing, consistency, max_resample, int(time_resolution), segmentation,
          n_min, min_support)


if __name__ == "__main__":
    main()
