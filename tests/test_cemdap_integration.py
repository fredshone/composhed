"""Integration tests for the CEMDAP-aligned variant (A1, A2a, A2d, A3, ablations)."""

import json
import os

import pytest

from composhed.common.bundle import BundleMismatch, load_bundle
from composhed.cemdap.generate import generate
from composhed.cemdap.train import train
from tests.variant_helpers import run_variant


def test_train_generate_invariants_and_determinism(tmp_path):
    bundle, df, report = run_variant(train, generate, str(tmp_path))
    b = load_bundle(bundle, "cemdap")
    assert b["spec_version"] and b["config"]["calibrate"] and b["feature_names"]
    with pytest.raises(BundleMismatch):
        load_bundle(bundle, "cemdap" if "cemdap" != "cemdap" else "daysim")


def test_calibration_report_within_tolerance(tmp_path):
    """A3: with calibration on, the report lists target, uncalibrated and calibrated values."""
    bundle, _, _ = run_variant(train, generate, str(tmp_path), {"calib_iters": 6, "calib_sample": 1500})
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
    run_variant(train, generate, str(tmp_path), kwargs)


def test_holdout_validation(tmp_path):
    """SR-18/19: a pid holdout produces holdout metrics in validation.json."""
    run_variant(train, generate, str(tmp_path), {"holdout_fraction": 0.3, "split_seed": 1, "calibrate": False})
    val = json.load(open(os.path.join(tmp_path, "models", "validation.json")))
    assert val["info"]["n_holdout"] == 16 and "CEM-1 work participation" in val["holdout"]
