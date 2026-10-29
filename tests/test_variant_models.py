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
    from composhed.common.report import GenerationReport
    from composhed.daysim.chain import DaysimChain
    from composhed.daysim.data import (
        build_records,
        enumerate_patterns,
        feasible,
        nearest_feasible,
        pattern_label,
        pattern_tours,
    )
    from composhed.daysim.models import DaysimModels

    pats = enumerate_patterns()
    assert len(pats) == 1324 and len(set(pats)) == 1324  # DaySim's 2,080 less the work / school stop flags
    # 4 tour purposes and a stop: infeasible; the lowest-priority tour flag goes first.
    v = (1, 0, 1, 1, 1, 0, 0) + (1, 0, 0, 0, 0)
    assert not feasible(v[:7], v[7:]) and nearest_feasible(v) == (1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    assert pattern_tours((1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0), {"shop": 2}) == ["work", "education", "shop", "shop"]
    attr_df, sched, train_pids, _ = data
    records, _, _ = build_records(attr_df, sched, train_pids)
    X, names = encode_labels([r.labels for r in records])
    m = DaysimModels().fit(records, X, names)
    assert all(pattern_label(v) == k for k, v in m.pattern_vectors.items())
    metrics = m.validate(records, X)
    assert "DAY-2 tour time work" in metrics and "DAY-5 stop generation" in metrics
    rng = np.random.default_rng(0)
    chain = DaysimChain(m)
    for r, x in zip(records[:20], X[:20]):
        rows, trace = chain.generate(x, {**r.labels}, rng, GenerationReport())
        assert sum(row["duration"] for row in rows) == 1440 and trace["pattern"] in m.pattern_vectors


def test_activitysim_models_and_person_types(data):
    from composhed.activitysim.data import build_records, mtf_label, nm_label
    from composhed.activitysim.models import AsimModels, Lookup
    from composhed.activitysim.persontype import person_type

    assert person_type("0-11", "void") == "PRESCHOOL"
    assert person_type("0-11", "student") == "SCHOOL"
    assert person_type("11-18", "employed") == "DRIVING"
    assert person_type("35-41", "employed") == "FTW"
    assert person_type("18-27", "student") == "UNIV"
    assert person_type("41-48", "unknown") == "NONWORK"
    assert mtf_label(1, 1) == "work_and_school" and mtf_label(0, 2) == "school2" and mtf_label(0, 0) is None

    attr_df, sched, train_pids, _ = data
    records, _, _ = build_records(attr_df, sched, train_pids)
    X, names = encode_labels([r.labels for r in records])
    m = AsimModels(n_min=10).fit(records, X, names)
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
    assert nm_label((1, 0, 0, 0, 1)) == "es1sh0om0od0so1"


def test_activitysim_ordering_and_trip_keys():
    from composhed.activitysim.data import AsimTour, mtf_tours, tour_order
    from composhed.activitysim.models import departure_keys, stop_alt, stop_attrs
    from composhed.common.periods import PeriodGrid

    assert mtf_tours("work_and_school", worker=True) == ["work", "school"]
    assert mtf_tours("work_and_school", worker=False) == ["school", "work"]
    assert tour_order("UNIV")[:2] == ["school", "work"]
    assert tour_order("FTW")[2:] == ["escort", "shopping", "othmaint", "othdiscr", "social"]
    assert stop_attrs(stop_alt(2, 1)) == [0, 1, 0, 1, 0, 0, 0, 1, 0]
    grid = PeriodGrid.with_early_bin(60)
    # 08:00 shop stop 20 min, work 08:20–17:00, two inbound stops 17:00–17:30 and 17:30–18:00.
    t = AsimTour("work", 480, 1080, outbound=[("shop", 480, 20)], inbound=[("escort", 1020, 30), ("shop", 1050, 30)],
                 primary_start=500, primary_end=1020)
    keys = departure_keys(t, grid)
    assert [(k[0], k[2], k[4], dep, ref) for k, dep, ref in keys] == [
        ("outbound", 4, 0, 500, 480),           # anchored on the start period (08:00), counted forwards
        ("inbound", 13, 1, 1020, 1050),         # anchored on the end period (17:00–18:00), counted backwards
        ("inbound", 13, 0, 1050, 1080)]
    # work 08:00–11:40, shop 11:40–12:10, work 12:10–16:40, escort 16:40–17:00: the shop sits inside the
    # merged work block (08:00–16:10 after merging), so the inbound departures are out of order.
    frag = AsimTour("work", 480, 1020, inbound=[("shop", 700, 30), ("escort", 1000, 20)],
                    primary_start=480, primary_end=970)
    assert departure_keys(frag, grid) == []
