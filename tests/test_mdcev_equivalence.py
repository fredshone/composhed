"""Verify that _numpy_forecast gives the same allocations as Biogeme's forecast().

Trains a tiny MDCEV model on synthetic in-memory data, then for each of several
test persons checks that both code paths produce identical allocations when given
the same fixed epsilon draws.
"""

import io
import logging
import tempfile

import biogeme.database as bio_db
import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest

from composhed.data import LABEL_COLS, build_training_dataset, encode_features
from composhed.models.mdcev import MDCEVModel
from tests.test_integration import _make_synthetic_data

TYPES = MDCEVModel.TYPES
K = len(TYPES)
BUDGET = MDCEVModel.BUDGET


# ---------------------------------------------------------------------------
# Fixture: tiny trained MDCEV model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_mdcev():
    """Train an MDCEV model on the 42-person synthetic dataset."""
    attr_rows, sched_rows = _make_synthetic_data()

    attr_df = pl.DataFrame(attr_rows)
    sched_df = pl.DataFrame(sched_rows).sort(["pid", "start"])

    records, _ = build_training_dataset(attr_df, sched_df)
    X, feature_names = encode_features(records, LABEL_COLS)

    # Suppress Biogeme's verbose logging during the test
    bio_logger = logging.getLogger("biogeme")
    bio_logger.setLevel(logging.ERROR)

    model = MDCEVModel().fit(records, feature_names, X=X)
    return model, X, feature_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _biogeme_forecast_one(model: MDCEVModel, x: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Run Biogeme's forecast() for one person with a fixed epsilon vector.

    Returns allocation array of shape (K,) in TYPES order.
    """
    row = dict(zip(model.feature_names_, x.tolist()))
    df = pd.DataFrame([row])
    database = bio_db.Database("eq_test", df)
    # epsilons: list of one array of shape (n_draws=1, K)
    result = model.biogeme_model_.forecast(
        database, total_budget=BUDGET, epsilons=[eps.reshape(1, K)]
    )
    # result[0] is a DataFrame with columns = alternative keys (0..K-1)
    row_result = result[0].iloc[0]
    return np.array([float(row_result.iloc[i]) for i in range(K)])


def test_numpy_matches_biogeme_fixed_epsilons(tiny_mdcev):
    """_numpy_forecast matches Biogeme for 10 persons with fixed epsilons."""
    model, X, _ = tiny_mdcev
    rng = np.random.default_rng(0)

    # Suppress Biogeme's per-observation INFO logs during assertions
    bio_logger = logging.getLogger("biogeme")
    bio_logger.setLevel(logging.ERROR)

    failures = []
    for i in range(min(10, len(X))):
        x = X[i]
        eps = rng.gumbel(0, 1, size=K)

        biogeme_alloc = _biogeme_forecast_one(model, x, eps)
        numpy_alloc = model._numpy_forecast(x.reshape(1, -1), eps.reshape(1, K))[0]

        if not np.allclose(biogeme_alloc, numpy_alloc, atol=1e-2):
            failures.append(
                f"person {i}: max_diff={np.abs(biogeme_alloc - numpy_alloc).max():.6f}\n"
                f"  biogeme={biogeme_alloc}\n"
                f"  numpy  ={numpy_alloc}"
            )

    assert not failures, "Allocation mismatch:\n" + "\n".join(failures)


def test_sample_batch_budget_and_nonneg(tiny_mdcev):
    """sample_batch allocations sum to 1440 and are non-negative."""
    model, X, _ = tiny_mdcev
    allocs = model.sample_batch(X)
    for i, d in enumerate(allocs):
        vals = np.array([d[t] for t in TYPES])
        assert (vals >= 0).all(), f"person {i}: negative allocation {vals}"
        assert abs(vals.sum() - BUDGET) < 1e-6, f"person {i}: sum={vals.sum()}"


def test_pickle_roundtrip_preserves_numpy_params(tiny_mdcev):
    """Pickle round-trip restores _intercepts, _B, _gammas, _scale identically."""
    model, X, _ = tiny_mdcev
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        joblib.dump(model, f.name)
        loaded = joblib.load(f.name)

    assert hasattr(loaded, "_intercepts"), "_intercepts missing after unpickle"
    assert hasattr(loaded, "_B"), "_B missing after unpickle"
    assert hasattr(loaded, "_gammas"), "_gammas missing after unpickle"
    assert hasattr(loaded, "_scale"), "_scale missing after unpickle"

    np.testing.assert_array_equal(loaded._intercepts, model._intercepts)
    np.testing.assert_array_equal(loaded._B, model._B)
    np.testing.assert_array_equal(loaded._gammas, model._gammas)
    assert loaded._scale == model._scale
