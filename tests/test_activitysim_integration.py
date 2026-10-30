"""Integration tests for the ActivitySim-aligned variant (A1, A2a, A2d, A3, ablations)."""


import pytest

from composhed.activitysim.generate import generate
from composhed.activitysim.train import train
from composhed.common.bundle import BundleMismatch, load_bundle
from tests.variant_helpers import run_variant

# The 52-person fixture is far below the NTS-scale support thresholds.
DEFAULTS = {}


def test_train_generate_invariants_and_determinism(tmp_path):
    bundle, df, report = run_variant(train, generate, str(tmp_path), DEFAULTS)
    b = load_bundle(bundle, "activitysim")
    assert b["spec_version"] and b["feature_names"]
    with pytest.raises(BundleMismatch):
        load_bundle(bundle, "cemdap" if "activitysim" != "cemdap" else "daysim")


@pytest.mark.parametrize("kwargs", [
    {"independent_timing": True},
    {"consistency": "rescale"},
])
def test_ablations_run(tmp_path, kwargs):
    run_variant(train, generate, str(tmp_path), DEFAULTS | kwargs)


def test_no_segmentation_and_resolution(tmp_path):
    """AB-4 / AB-5."""
    bundle, _, _ = run_variant(train, generate, str(tmp_path),
                               DEFAULTS | {"segmentation": False, "time_resolution": 30})
    cfg = load_bundle(bundle, "activitysim")["config"]
    assert cfg["segmentation"] is False and cfg["time_resolution"] == 30
