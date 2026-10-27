"""Integration tests for the ActivitySim-aligned variant (A1, A2a, A2d, A3, ablations)."""

import json
import os

import pytest

from composhed.common.bundle import BundleMismatch, load_bundle
from composhed.activitysim.generate import generate
from composhed.activitysim.train import train
from tests.variant_helpers import run_variant

# The 52-person fixture is far below the NTS-scale support thresholds.
DEFAULTS = {"min_support": 1, "n_min": 10}


def test_train_generate_invariants_and_determinism(tmp_path):
    bundle, df, report = run_variant(train, generate, str(tmp_path), DEFAULTS)
    b = load_bundle(bundle, "activitysim")
    assert b["spec_version"] and b["config"]["calibrate"] and b["feature_names"]
    with pytest.raises(BundleMismatch):
        load_bundle(bundle, "cemdap" if "activitysim" != "cemdap" else "daysim")


def test_calibration_report_within_tolerance(tmp_path):
    """A3: with calibration on, the report lists target, uncalibrated and calibrated values."""
    bundle, _, _ = run_variant(train, generate, str(tmp_path), DEFAULTS | {"calib_iters": 6, "calib_sample": 1500})
    val = json.load(open(os.path.join(tmp_path, "models", "validation.json")))
    targets = val["info"]["calibration"]["targets"]
    assert targets
    for name, t in targets.items():
        assert {"target", "uncalibrated", "calibrated", "max_error"} <= set(t)
    # Documented tolerance for the 52-person fixture (simulation noise dominates): 0.1 share / 10 % of mean.
    for name, t in targets.items():
        assert t["max_error"] <= 0.10, f"{name}: {t['max_error']}"


@pytest.mark.parametrize("kwargs", [
    {"calibrate": False},
    {"independent_timing": True},
    {"consistency": "rescale"},
])
def test_ablations_run(tmp_path, kwargs):
    run_variant(train, generate, str(tmp_path), DEFAULTS | kwargs)


def test_no_segmentation_and_resolution(tmp_path):
    """AB-4 / AB-5."""
    bundle, _, _ = run_variant(train, generate, str(tmp_path),
                               DEFAULTS | {"segmentation": False, "time_resolution": 30, "calibrate": False})
    cfg = load_bundle(bundle, "activitysim")["config"]
    assert cfg["segmentation"] is False and cfg["time_resolution"] == 30
