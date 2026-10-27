"""Sub-model tests for the three source-aligned variants (A2b): fit / sample / validate."""

import numpy as np
import pytest

from composhed.common.runner import encode_labels, load_training_data
from tests.variant_helpers import write_fixture


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    attr, sched, _ = write_fixture(str(d))
    return load_training_data(attr, sched, 0.0, 0)


def test_cemdap_models(data):
    from composhed.cemdap.chain import CemdapChain, record_trace, trace_stats
    from composhed.cemdap.data import build_records
    from composhed.cemdap.models import COUNT_POSITIONS, CemdapModels

    attr_df, sched, train_pids, _ = data
    records, counters = build_records(attr_df, sched, train_pids)
    assert counters.persons == len(records) == 52
    X, names = encode_labels([r.labels for r in records])
    m = CemdapModels().fit(records, X, names)
    metrics = m.validate(records, X)
    assert "CEM-1 work participation" in metrics and metrics["CEM-1 work participation"]["n"] == 52
    assert set(m.counts) == set(COUNT_POSITIONS)
    rng = np.random.default_rng(0)
    assert set(m.work.sample(X, rng)) <= {0, 1}
    s = m.start.sample(np.hstack([X, np.zeros((52, 1))]), rng, lo=10, hi=500)
    assert s.shape == (52,) and s.min() >= 10 and s.max() <= 500
    from composhed.common.report import GenerationReport

    chain = CemdapChain(m)
    rows, trace = chain.generate(X[5], {"employment": "employed"}, rng, GenerationReport())
    assert sum(r["duration"] for r in rows) == 1440 and "stops" in trace
    stats = trace_stats([record_trace(r) for r in records])
    assert abs(sum(v for k, v in stats["work_participation"].items() if k != "__n__") - 1) < 1e-9


def test_daysim_models(data):
    from composhed.daysim.data import build_records, pattern_label, pattern_tours
    from composhed.daysim.models import DIRECTIONS, DaysimModels

    attr_df, sched, train_pids, _ = data
    records, _, _ = build_records(attr_df, sched, train_pids)
    X, names = encode_labels([r.labels for r in records])
    m = DaysimModels(min_support=1).fit(records, X, names)
    assert all(pattern_label(v) == k for k, v in m.pattern_vectors.items())
    rng = np.random.default_rng(0)
    labels = m.pattern.sample(X, rng)
    assert all(l in m.pattern_vectors for l in labels)
    assert pattern_tours((1, 1, 0, 0, 1, 0, 0, 1)) == ["work", "education", "shop"]
    metrics = m.validate(records, X)
    assert "DAY-2 TOD work" in metrics and all(f"DAY-5 stops {k}" in metrics for k in DIRECTIONS)
    # Rare patterns map to the nearest supported one.
    m2 = DaysimModels(min_support=10).fit(records, X, names)
    assert m2.map_pattern((2, 2, 4, 0, 0, 0, 0, 1)) in m2.pattern_vectors


def test_activitysim_models_and_person_types(data):
    from composhed.activitysim.data import build_records, mtf_label, nm_label
    from composhed.activitysim.models import AsimModels, Lookup
    from composhed.activitysim.persontype import person_type

    assert person_type("≤10", "ft-employed") == "CHILD"
    assert person_type("32-40", "ft-employed") == "FTW"
    assert person_type("10-21", "unknown") == "DRIVING_STUDENT"
    assert person_type("21-32", "student") == "UNIV"
    assert person_type("40-48", "unknown") == "NONWORK"
    assert mtf_label(1, 1) == "work_and_school" and mtf_label(0, 2) == "school2" and mtf_label(0, 0) is None

    attr_df, sched, train_pids, _ = data
    records, _, _ = build_records(attr_df, sched, train_pids)
    X, names = encode_labels([r.labels for r in records])
    m = AsimModels(n_min=10, min_support=1).fit(records, X, names)
    rng = np.random.default_rng(0)
    cdap = m.cdap.sample(X, [r.ptype for r in records], rng)
    assert set(cdap) <= set(m.cdap_classes)
    # N persons cannot choose the zero-tour non-mandatory alternative.
    avail = m._nm_avail(["N"])
    zero = [k for k, v in m.nm_vectors.items() if sum(v) == 0][0]
    assert not avail[0, list(m.nm_vectors).index(zero)]
    metrics = m.validate(records, X)
    assert "ASIM-2 CDAP" in metrics and "ASIM-7 stop frequency" in metrics
    lk = Lookup(["a", "b"], min_count=2).fit([("x", 1), ("x", 1), ("y", 1)], ["a", "a", "b"])
    assert lk.probs(("x", 1))[0] == 1.0 and np.isclose(lk.probs(("z", 1)).sum(), 1.0)
    assert nm_label((1, 0, 0, 0, 1)) == "es1sh0ot0so0ot1"
