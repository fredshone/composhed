"""Integration tests for the CEMDAP-aligned variant (A1, A2a, A2d, A3, ablations)."""

import json
import os

import pytest

from composhed.cemdap.generate import generate
from composhed.cemdap.train import train
from composhed.common.bundle import BundleMismatch, load_bundle
from tests.variant_helpers import run_variant


def test_train_generate_invariants_and_determinism(tmp_path):
    bundle, df, report = run_variant(train, generate, str(tmp_path))
    b = load_bundle(bundle, "cemdap")
    assert b["spec_version"] and b["feature_names"]
    with pytest.raises(BundleMismatch):
        load_bundle(bundle, "cemdap" if "cemdap" != "cemdap" else "daysim")


@pytest.mark.parametrize("kwargs", [
    {"independent_timing": True},
    {"consistency": "rescale"},
])
def test_ablations_run(tmp_path, kwargs):
    run_variant(train, generate, str(tmp_path), kwargs)


def test_holdout_validation(tmp_path):
    """SR-18/19: a pid holdout produces holdout metrics in validation.json."""
    run_variant(train, generate, str(tmp_path), {"holdout_fraction": 0.3, "split_seed": 1})
    val = json.load(open(os.path.join(tmp_path, "models", "validation.json")))
    assert val["info"]["n_holdout"] == 16 and "CEM-1 work participation" in val["holdout"]
