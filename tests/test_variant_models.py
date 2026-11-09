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
    from composhed.cemdap.chain import CemdapChain
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
    s = m.start.sample(np.hstack([X, np.zeros((52, 3))]), rng, lo=10, hi=500)
    assert s.shape == (52,) and s.min() >= 10 and s.max() <= 500
    from composhed.common.report import GenerationReport

    chain = CemdapChain(m)
    rows, trace = chain.generate(X[5], {"employment": "employed"}, rng, GenerationReport())
    assert sum(r["duration"] for r in rows) == 1440 and "stops" in trace


def test_cemdap_work_based_tour_record():
    from composhed.cemdap.data import build_record
    from composhed.common.tours import extract_day_structure

    rows, t = [], 0
    for act, d in (("home", 420), ("work", 200), ("shop", 30), ("work", 280), ("home", 510)):
        rows.append({"act": act, "start": t, "end": t + d, "duration": d})
        t += d
    day = extract_day_structure(rows, collapse_mandatory_tours=True, mandatory="work_based")
    r = build_record(0, {}, day)
    assert (r.mand_type, r.mand_start, r.mand_duration, r.n_wb) == ("work", 420, 510, 1)
    # The work stay before the work-based tour, then its stop, both against the work-based period's residual.
    assert r.home_stays == [{"side": "work", "duration": 200, "residual": 510.0}]
    assert r.stops == [{"purpose": "shop", "duration": 30, "position": "wb_tour", "index": 0, "residual": 310.0}]


def test_daysim_models(data):
    from composhed.common.report import GenerationReport
    from composhed.daysim.chain import DaysimChain
    from composhed.daysim.data import (
        available,
        stop_allowed,
        build_records,
        enumerate_patterns,
        feasible,
        is_student,
        is_worker,
        pattern_label,
        pattern_tours,
    )
    from composhed.daysim.models import (
        DaysimModels,
        pattern_b_mask,
        pattern_inter_attrs,
    )

    pats = enumerate_patterns()
    assert len(pats) == 2080 and len(set(pats)) == 2080  # DaySim's TOTAL_ALTERNATIVES
    z7 = (0,) * 7
    v = (1, 0, 1, 1, 1, 0, 0) + (0, 0, 1, 0, 0, 0, 0)
    assert not feasible(v[:7], v[7:])  # 4 tour purposes
    # Work / school stops only with a work or school tour, and at most one of them.
    assert not feasible((0, 0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0, 0))
    assert feasible((0, 1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0))
    assert not feasible((1, 0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0))
    # DaySim availability: work tours and stops for workers only, school tours and stops for students only.
    work_day, school_day = (1, 0, 0, 0, 0, 0, 0) + z7, (0, 1, 0, 0, 0, 0, 0) + z7
    assert not available(work_day, False, True) and available(work_day, True, False)
    assert not available(school_day, True, False) and available(school_day, False, True)
    assert not available((0, 1, 0, 0, 0, 0, 0) + (1, 0, 0, 0, 0, 0, 0), False, True)
    assert is_worker({"employment": "employed"}) and is_student({"employment": "void"}, "SCHOOL")
    # Stop purposes a tour allows (IntermediateStopGenerationModel).
    assert stop_allowed("work", "education") and not stop_allowed("shop", "work")
    assert stop_allowed("escort", "escort") and not stop_allowed("shop", "escort") and stop_allowed("shop", "visit")
    # Two tour purposes share one person vector scaled by log(2); the tour component covers non-mandatory tours.
    a = pattern_inter_attrs((1, 0, 0, 0, 1, 0, 0) + (0, 0, 1, 0, 0, 0, 0))
    assert np.isclose(a[21], np.log(2)) and a[22] == 0 and a[23] == 1 and a[24] == 1
    assert pattern_inter_attrs((1, 0, 0, 0, 0, 0, 0) + (1, 0, 0, 0, 0, 0, 0))[24] == 0  # work stops: not the stop component
    assert pattern_b_mask().sum() == 7 * 18 + 14 + 2 * 16 + 2 * 4
    assert pattern_tours((1, 1, 0, 0, 1, 0, 0) + z7, {"shop": 2}) == ["work", "education", "shop", "shop"]
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
    m = AsimModels().fit(records, X, names)
    rng = np.random.default_rng(0)
    env = m.person_env([r.labels for r in records])
    assert set(m.cdap.sample(env, len(records), rng)) <= {"M", "N", "H"}
    # prototype_mtc: non-working adults and retirees cannot have a mandatory day.
    retired = m.person_env([{**records[0].labels, "age": "71-118", "employment": "retired"}])
    assert m.cdap.predict_proba(retired, 1)[0, m.cdap.classes_.index("M")] == 0
    # Without mandatory tours the zero-tour non-mandatory alternative is unavailable (spec row, -999).
    seg = next(iter(m.nmtf))
    zero = [k for k, v in m.nm_vectors.items() if sum(v) == 0][0]
    P = m.nmtf[seg].predict_proba(m.nm_env(m.person_env([records[0].labels]), [0], [0.0]), 1)
    assert P[0, list(m.nm_vectors).index(zero)] == 0
    metrics = m.validate(records, X)
    assert "ASIM-2 CDAP" in metrics and any(k.startswith("ASIM-7 stop frequency") for k in metrics)
    assert not any(r["missing"] == ["ptype"] for r in m.cdap.spec.dropped_rows)
    lk = Lookup(["a", "b"], min_count=2).fit([("x", 1), ("x", 1), ("y", 1)], ["a", "a", "b"])
    assert lk.probs(("x", 1))[0] == 1.0 and np.isclose(lk.probs(("z", 1)).sum(), 1.0)
    assert nm_label((1, 0, 0, 0, 1)) == "es1sh0om0od0so1"


def test_activitysim_ordering_and_trip_keys():
    from composhed.activitysim.data import AsimTour, mtf_tours, tour_order
    from composhed.activitysim.models import (
        departure_keys,
        stop_alt,
        stop_segment,
        tour_env,
    )
    from composhed.common.periods import PeriodGrid

    assert mtf_tours("work_and_school", worker=True) == ["work", "school"]
    assert mtf_tours("work_and_school", worker=False) == ["school", "work"]
    assert tour_order("UNIV")[:2] == ["school", "work"]
    assert tour_order("FTW")[2:] == ["escort", "shopping", "othmaint", "othdiscr", "social"]
    assert stop_alt(2, 5) == "2out_3in"
    assert stop_segment("school", "UNIV") == "univ" and stop_segment("school", "SCHOOL") == "school"
    from collections import Counter

    e = tour_env(["social"], [7 * 60 + 30], [19 * 60 + 10], [Counter(work=1, social=1)], [False])
    assert e["start"][0, 0] == 7 and e["end"][0, 0] == 19 and e["duration"][0, 0] == 12 and e["is_social"][0, 0]
    grid = PeriodGrid.with_early_bin(60, early_end=360, early_point=330)  # periods 5-23 -> indices 0-18
    # 08:00 shop stop 20 min, work 08:20–17:00, two inbound stops 17:00–17:30 and 17:30–18:00.
    t = AsimTour("work", 480, 1080, outbound=[("shop", 480, 20)], inbound=[("escort", 1020, 30), ("shop", 1050, 30)],
                 primary_start=500, primary_end=1020)
    assert departure_keys(t, grid, "work") == [
        (("work", True, 3, 2), 500),    # outbound trip 2, tour hour = start period (08:00)
        (("work", False, 12, 1), 1020),  # inbound trips numbered from the primary activity,
        (("work", False, 12, 2), 1050)]  # tour hour = end period (17:00–18:00)
    # work 08:00–11:40, shop 11:40–12:10, work 12:10–16:40, escort 16:40–17:00: the shop sits inside the
    # merged work block (08:00–16:10 after merging), so the inbound departures are out of order.
    frag = AsimTour("work", 480, 1020, inbound=[("shop", 700, 30), ("escort", 1000, 20)],
                    primary_start=480, primary_end=970)
    assert departure_keys(frag, grid, "work") == []
    from composhed.activitysim.models import extension_eligible, log_max_window, purpose_range

    assert purpose_range("work", True, 8) == "5-8" and purpose_range("work", False, 15) == "15-23"
    assert purpose_range("shopping", True, 8) == "5-23"
    # escort 2 and shopping 1 are at their maximum; othdiscr is not chosen; 5 tours in all is not eligible
    assert extension_eligible((2, 1, 0, 0, 0)) == [True, True, False, False, False]
    assert extension_eligible((1, 0, 0, 1, 0)) == [False, False, False, True, False]
    assert extension_eligible((2, 1, 1, 1, 0)) == [False] * 5
    # work 08:30–17:30 (periods 8–17): free blocks 5–8 (4 periods) and 17–23 (7 periods)
    assert np.isclose(log_max_window([(510, 1050)], grid), np.log1p(7))
