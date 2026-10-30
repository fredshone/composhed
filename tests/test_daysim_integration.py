"""Integration tests for the DaySim-aligned variant (A1, A2a, A2d, A3, ablations)."""


import pytest

from composhed.common.bundle import BundleMismatch, load_bundle
from composhed.daysim.generate import generate
from composhed.daysim.train import train
from tests.variant_helpers import run_variant

# The 52-person fixture is far below the NTS-scale support thresholds.
DEFAULTS = {}


def test_train_generate_invariants_and_determinism(tmp_path):
    bundle, df, report = run_variant(train, generate, str(tmp_path), DEFAULTS)
    b = load_bundle(bundle, "daysim")
    assert b["spec_version"] and b["feature_names"]
    with pytest.raises(BundleMismatch):
        load_bundle(bundle, "cemdap" if "daysim" != "cemdap" else "daysim")


@pytest.mark.parametrize("kwargs", [
    {"independent_timing": True},
    {"consistency": "rescale"},
])
def test_ablations_run(tmp_path, kwargs):
    run_variant(train, generate, str(tmp_path), DEFAULTS | kwargs)


@pytest.mark.parametrize("res", [15, 30, 60])
def test_time_resolution(tmp_path, res):
    """DAY-10 / AB-4."""
    bundle, _, _ = run_variant(train, generate, str(tmp_path), DEFAULTS | {"time_resolution": res})
    assert load_bundle(bundle, "daysim")["config"]["time_resolution"] == res
