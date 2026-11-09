"""Unit tests for composhed.common (A2b/c): time windows, tours, choice, spec and duration models."""

import numpy as np
import pytest

from composhed.common.choice import MNL, AltAttrMNL, ConditionalLogit, OrderedLogit
from composhed.common.hazard import AFTModel
from composhed.common.lognormal import LogNormalDuration
from composhed.common.periods import PeriodGrid
from composhed.common.schedule import (
    InvalidSchedule,
    check_rows,
    finalise,
    from_tours,
)
from composhed.common.timewindow import TimeWindow
from composhed.common.tod import TourTOD, layout_rescale
from composhed.common.tours import ExtractionCounters, extract_day_structure
from composhed.data import extract_tour_memberships


def day(*acts):
    """acts: (act, duration) pairs → sorted rows with start/end/duration."""
    rows, t = [], 0
    for a, d in acts:
        rows.append({"act": a, "start": t, "end": t + d, "duration": d})
        t += d
    assert t == 1440
    return rows


# ---- TimeWindow ------------------------------------------------------------------------
def test_timewindow_reserve_and_queries():
    tw = TimeWindow()
    assert tw.remaining() == 1440 and tw.is_free(0, 1440)
    tw.reserve(480, 1020)
    assert not tw.is_free(500, 600) and tw.is_free(0, 480) and tw.is_free(1020, 1440)
    assert tw.remaining() == 1440 - 540
    assert tw.free_windows() == [(0, 480), (1020, 1440)]
    with pytest.raises(ValueError):
        tw.reserve(1000, 1100)
    # usable windows keep the buffer next to tours and at the day edges; with none, tours may touch
    assert tw.usable_windows(10) == [(10, 470), (1030, 1430)]
    assert tw.largest_window(10) == (10, 470)
    assert tw.usable_windows(0) == [(0, 480), (1020, 1440)]
    tw.reserve(1020, 1020)  # a zero-length tour reserves nothing
    assert tw.remaining() == 1440 - 540


def test_point_mask_lets_tours_touch():
    from composhed.common.tod import point_mask

    tw = TimeWindow()
    tw.reserve(510, 990)
    s, e = np.array([450, 990, 510, 750, 990, 450]), np.array([510, 1050, 510, 750, 990, 570])
    # ends where the tour starts, starts where it ends, zero-length at either edge: available;
    # zero-length inside it, or overlapping it: not
    assert point_mask(tw, s, e).tolist() == [True, True, True, False, True, False]
    tw.reserve(300, 300)  # a zero-length tour: nothing may span it
    assert point_mask(tw, np.array([240, 300, 240]), np.array([360, 360, 300])).tolist() == [False, True, True]


def test_timewindow_masks():
    tw = TimeWindow()
    tw.reserve(600, 700)
    m = tw.availability_mask(np.array([10, 590, 710, 800]), np.array([100, 650, 800, 1440]), buffer=10)
    assert m.tolist() == [True, False, True, False]
    grid = PeriodGrid.uniform(60)
    ps, pe = grid.pairs()
    mask = tw.period_availability_mask(grid.lo(ps), grid.hi(ps), grid.lo(pe), grid.hi(pe), buffer=10)
    j = {(a, b): i for i, (a, b) in enumerate(zip(ps, pe))}
    assert mask[j[(8, 9)]]  # 08:00-10:00 fits before 10:00 tour with buffer
    assert not mask[j[(9, 12)]]  # spans the reserved tour
    assert mask[j[(11, 13)]]  # starts after 11:50


# ---- tours -------------------------------------------------------------------------------
def test_simple_work_day():
    d = extract_day_structure(day(("home", 480), ("work", 540), ("home", 420)))
    assert len(d.tours) == 1 and d.tours[0].purpose == "work"
    assert d.home_stays == [480, 420] and d.total() == 1440


def test_stops_both_sides_and_parity_with_compositional_labels():
    rows = day(("home", 400), ("escort", 20), ("work", 500), ("shop", 30), ("home", 490))
    d = extract_day_structure(rows)
    t = d.tours[0]
    assert [s.act for s in t.outbound] == ["escort"] and [s.act for s in t.inbound] == ["shop"]
    assert extract_tour_memberships(rows) == ["mandatory_outbound", "mandatory_inbound"]


def test_interrupted_work_is_not_merged():
    rows = day(("home", 420), ("work", 200), ("other", 30), ("work", 280), ("home", 510))
    c = ExtractionCounters()
    # DaySim / ActivitySim: the longest work episode is primary, the other is a work stop.
    t = extract_day_structure(rows, counters=c).tours[0]
    assert t.purpose == "work" and t.primary_start == 650 and t.primary_duration == 280
    assert [(s.act, s.duration) for s in t.outbound] == [("work", 200), ("other", 30)] and not t.inbound
    assert c.fragmented_tours == 1
    # CEMDAP: one work-based period with a work-based tour inside it.
    d = extract_day_structure(rows, mandatory="work_based")
    t = d.tours[0]
    assert t.primary_start == 420 and t.primary_duration == 510 and t.work_stays == [200, 280]
    assert [[s.act for s in run] for run in t.work_based] == [["other"]] and d.total() == 1440


def test_two_mandatory_tours_collapse_vs_keep():
    rows = day(("home", 420), ("work", 240), ("home", 60), ("shop", 20), ("work", 200), ("home", 500))
    keep = extract_day_structure(rows)
    assert [t.purpose for t in keep.tours] == ["work", "work"]
    c = ExtractionCounters()
    col = extract_day_structure(rows, collapse_mandatory_tours=True, counters=c, mandatory="work_based")
    assert [t.purpose for t in col.tours] == ["work", "shop"]
    assert col.tours[0].primary_duration == 440 and col.tours[0].work_stays == [440]
    assert col.total() == 1440 and c.collapsed_tours == 1
    rows2 = day(("home", 420), ("work", 240), ("home", 60), ("work", 200), ("home", 520))
    col2 = extract_day_structure(rows2, collapse_mandatory_tours=True)
    assert len(col2.tours) == 1 and col2.home_stays == [420, 580] and col2.total() == 1440


def test_discretionary_day_three_tours_and_primary_rule():
    rows = day(("home", 200), ("escort", 15), ("home", 100), ("shop", 30), ("visit", 90), ("home", 300),
               ("other", 60), ("home", 645))
    d = extract_day_structure(rows)
    assert [t.purpose for t in d.tours] == ["escort", "visit", "other"]
    assert [s.act for s in d.tours[1].outbound] == ["shop"]
    assert d.home_stays == [200, 100, 300, 645]


def test_early_departure():
    d = extract_day_structure(day(("home", 120), ("work", 600), ("home", 720)))
    assert d.tours[0].start == 120 and d.home_stays[0] == 120


# ---- periods ---------------------------------------------------------------------------
def test_period_grid():
    g = PeriodGrid.with_early_bin(60)
    assert g.n == 20 and g.period_of(0) == 0 and g.period_of(299) == 0 and g.period_of(300) == 1
    assert g.end_period_of(360) == 1 and g.end_period_of(361) == 2


# ---- choice models -------------------------------------------------------------------------
def test_mnl_recovers_parameters_and_respects_availability():
    rng = np.random.default_rng(0)
    N = 20000
    X = rng.normal(size=(N, 2))
    W = np.array([[0.0, 0.5, -0.5], [0.0, 1.0, -1.0], [0.0, -0.5, 0.8]])
    V = np.hstack([np.ones((N, 1)), X]) @ W
    P = np.exp(V) / np.exp(V).sum(1, keepdims=True)
    y = [["a", "b", "c"][(rng.random() < P[i].cumsum()).argmax()] for i in range(N)]
    m = MNL(l2=0).fit(X, y, classes=["a", "b", "c"])
    assert np.allclose(m.W_, W, atol=0.08)
    avail = np.ones((5, 3), bool)
    avail[:, 1] = False
    assert all(v != "b" for v in m.sample(X[:5], rng, avail))
    v = m.validate(X, y)
    assert v["n"] == N and 0 < v["prsquared"] < 1


def test_ordered_logit_recovery_and_caps():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, 2))
    y = ((X @ np.array([1.0, -0.5]) + rng.logistic(size=20000))[:, None] > np.array([-1.0, 1.0])).sum(1)
    m = OrderedLogit().fit(X, y)
    assert np.allclose(m.beta_, [1.0, -0.5], atol=0.06) and np.allclose(m.thresholds_, [-1, 1], atol=0.08)
    s = m.sample(X[:200], rng, max_allowed=1)
    assert s.max() <= 1
    s = m.sample(X[:200], rng, min_allowed=1)
    assert s.min() >= 1


def test_conditional_logit_recovers_shape_terms():
    rng = np.random.default_rng(2)
    J, N = 12, 15000
    F = np.eye(J)[:, 1:]
    G = np.linspace(0, 1, J)[:, None]
    Z = np.column_stack([np.ones(N), rng.normal(size=N)])
    B = np.array([[0.0], [2.0]])
    gamma = rng.normal(0, 0.3, J - 1)
    V = (F @ gamma)[None, :] + (Z @ B) @ G.T
    P = np.exp(V) / np.exp(V).sum(1, keepdims=True)
    y = (rng.random(N)[:, None] > P.cumsum(1)).sum(1)
    m = ConditionalLogit(F, G, l2=1e-6, maxiter=500).fit(Z, y)
    assert abs(m.B_[1, 0] - 2.0) < 0.2
    avail = np.zeros((3, J), bool)
    avail[:, 4] = True
    assert (m.sample(Z[:3], rng, avail) == 4).all()
    assert (m.sample(Z[:1], rng, np.zeros((1, J), bool)) == -1).all()



def test_conditional_logit_recovers_chooser_specific_attributes():
    """The dyn path (ActivitySim timetable terms): δ recovered with chunked attributes."""
    rng = np.random.default_rng(3)
    J, N = 8, 12000
    F = np.eye(J)[:, 1:]
    G = np.zeros((J, 1))
    A = rng.normal(size=(N, J, 2))
    delta = np.array([1.5, -0.8])
    V = A @ delta
    P = np.exp(V) / np.exp(V).sum(1, keepdims=True)
    y = (rng.random(N)[:, None] > P.cumsum(1)).sum(1)
    m = ConditionalLogit(F, G, l2=1e-6, maxiter=500, chunk=1000, n_dyn=2).fit(np.ones((N, 1)), y, dyn=lambda s, e: A[s:e])
    assert np.allclose(m.delta_, delta, atol=0.1)
    assert np.allclose(m.predict_proba(np.ones((5, 1)), dyn=lambda s, e: A[s:e]), P[:5], atol=0.02)


def test_alt_attr_mnl_recovers_interaction():
    rng = np.random.default_rng(4)
    attrs = {f"a{k}": [float(k), float(k >= 2)] for k in range(4)}
    N = 15000
    x = rng.normal(size=(N, 1))
    a = np.array(list(attrs.values()))
    V = 0.5 * a[:, 0][None, :] + (x * 1.2) @ a[:, :1].T
    P = np.exp(V) / np.exp(V).sum(1, keepdims=True)
    y = [f"a{k}" for k in (rng.random(N)[:, None] > P.cumsum(1)).sum(1)]
    m = AltAttrMNL(attrs, ["count", "2plus"], l2=1e-6).fit(x, y, list(attrs))
    assert abs(m.coef_table()["x0"]["count"] - 1.2) < 0.1
    m.offset[0] += 5.0  # offsets are alternative-specific
    assert m.predict_proba(x[:1])[0, 0] > P[0, 0]


def test_tod_timetable_terms():
    """ActivitySim's timetable rows on prototype_mtc's 19 hourly periods (5-23)."""
    from composhed.activitysim.tod_spec import SPECS

    grid = PeriodGrid.with_early_bin(60, early_end=360, early_point=330)
    assert grid.n == 19 and grid.points()[0] == 330 and grid.points()[1] == 390
    tod = TourTOD(grid, SPECS["nonmandatory"], dynamic="asim", points=True)
    tw = TimeWindow()
    tw.reserve(510, 990)  # periods 8 (START) .. 16 (END), 9-15 in the middle
    w = tod.state(tw)[0]
    assert w[3] == 2 and (w[4:11] == 7).all() and w[11] == 4
    A = tod.dyn_fn([tod.state(tw, first=True, numerator=2.0)])(0, 1)[0]
    alt = lambda a, b: tod.alt_index[(a - 5, b - 5)]  # clock hours -> period indices
    assert A[alt(16, 18), 0] == 1 and A[alt(17, 18), 0] == 0  # departs in the period the earlier tour ended
    assert A[alt(5, 8), 1] == 1 and A[alt(5, 7), 1] == 0  # arrives in the period the earlier tour began
    assert A[alt(18, 20), 2] == 1  # free periods before departure (16 is an END, not middle): a 0/1 flag
    assert A[alt(17, 18), 2] == 1 and A[alt(16, 18), 2] == 0  # period 15 is in the middle of the tour
    assert A[alt(5, 8), 3] == 0  # period 9 is in the middle of the tour
    assert A[alt(18, 20), 4] == 0 and A[alt(18, 20), 5] == 0  # not a later tour
    assert np.isclose(A[alt(16, 18), 6], 2.0 / (12 - 1))  # 12 free periods, 1 more taken by the tour
    assert (tod.dyn_fn([tod.state(TimeWindow())])(0, 1)[0] == 0).all()
    # "Subsequent tour must start after previous tour ends": nothing may start before period 16.
    m = tod.mask(tw, min_start=990)
    assert m[alt(16, 18)] and m[alt(22, 23)] and not m[alt(5, 7)] and not m[alt(15, 15)]
    assert tod.mask(tw)[alt(5, 7)]


# ---- duration models ------------------------------------------------------------------------
@pytest.mark.parametrize("distr", ["loglogistic", "weibull"])
def test_aft_recovery_and_truncation(distr):
    rng = np.random.default_rng(3)
    N = 20000
    X = rng.normal(size=(N, 1))
    eps = rng.logistic(size=N) if distr == "loglogistic" else np.log(rng.exponential(size=N))
    t = np.exp(4.0 + 0.5 * X[:, 0] + 0.4 * eps)
    m = AFTModel(distr).fit(X, t)
    assert np.allclose(m.beta_, [4.0, 0.5], atol=0.03) and abs(m.sigma_ - 0.4) < 0.02
    best = AFTModel.fit_best(X, t)
    assert best.distr == distr
    s = m.sample(X[:500], rng, lo=10.0, hi=30.0)
    assert s.min() >= 10 and s.max() <= 30


def test_lognormal_truncated_sampling():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5000, 1))
    t = np.exp(3 + X[:, 0] + rng.normal(0, 0.3, 5000))
    m = LogNormalDuration().fit(X, t)
    assert abs(m.residual_std_ - 0.3) < 0.02
    s = m.sample(X[:300], rng, lo=5, hi=8)
    assert s.min() >= 5 and s.max() <= 8


# ---- spec logits -----------------------------------------------------------------------------------
def test_spec_expressions_use_pandas_precedence():
    from composhed.common.spec import compile_expr

    env = {"a": np.array([[1], [2]]), "b": np.array([[True], [False]]), "t": np.array([[0, 1, 2]])}
    # pandas eval: & binds looser than ==, so this is (a == 1) and b, not a == (1 & b).
    assert compile_expr("a == 1 & b")(env).ravel().tolist() == [1, 0]
    assert compile_expr("~b | (a > 1)")(env).ravel().tolist() == [0, 1]
    assert compile_expr("b * (t == 1)")(env).tolist() == [[0, 1, 0], [0, 0, 0]]
    assert compile_expr("a * b").names == {"a", "b"}


def test_spec_logit_recovers_shared_coefficients_and_availability():
    from composhed.common.spec import Spec, SpecLogit

    rng = np.random.default_rng(3)
    n = 20000
    x = rng.normal(size=(n, 1))
    V = np.column_stack([np.zeros(n), 0.5 + 1.0 * x[:, 0], -0.5 + 1.0 * x[:, 0]])
    blocked = rng.random(n) < 0.3  # alternative c unavailable for 30%
    V[blocked, 2] = -np.inf
    P = np.exp(V) / np.exp(V).sum(1, keepdims=True)
    y = [["a", "b", "c"][i] for i in (rng.random(n)[:, None] > P.cumsum(1)).sum(1)]
    spec = (Spec(["a", "b", "c"], [])
            .add("asc_b", "1", {"b": "asc_b"}).add("asc_c", "1", {"c": "asc_c"})
            .add("x", "x", {"b": "beta_x", "c": "beta_x"})      # one coefficient on two alternatives
            .add("blocked", "blocked", {"c": -999.0})           # a fixed -999 makes c unavailable
            .add("never", "x * 0", {"b": "beta_none"})          # no variation: fixed at 0, listed
            .add("missing", "z", {"b": "beta_z"}).restrict({"x", "blocked"}))
    env = {"x": x, "blocked": blocked.reshape(n, 1)}
    m = SpecLogit(spec, l2=1e-6).fit(env, n, y)
    b = m.coef_table()
    assert abs(b["beta_x"] - 1.0) < 0.05 and abs(b["asc_b"] - 0.5) < 0.1 and abs(b["asc_c"] + 0.5) < 0.1
    assert m.unidentified_ == ["beta_none"] and spec.dropped_rows[0]["missing"] == ["z"]
    assert m.n_params == 3
    P_hat = m.predict_proba(env, n)
    assert np.all(P_hat[blocked, 2] == 0)


def test_read_spec_reads_activitysim_files():
    from pathlib import Path

    from composhed.common.spec import read_spec

    d = Path(__file__).parents[1] / "composhed" / "activitysim" / "specs"
    spec = read_spec(str(d / "mandatory_tour_frequency.csv"), coefficients=str(d / "mandatory_tour_frequency_coefficients.csv"))
    assert spec.alternatives == ["work1", "work2", "school1", "school2", "work_and_school"]
    # Availability rows become fixed -999 cells; literal 0 cells add nothing.
    avail_row = next(r for r in spec.rows if r.label == "util_availability_driving_age_child")
    assert avail_row.cells == {"work1": -999.0, "work2": -999.0}
    ft = next(r for r in spec.rows if r.label == "util_ft_worker")
    assert ft.cells == {"work2": "coef_ft_worker_work2_asc"}
    nm = read_spec(str(d / "non_mandatory_tour_frequency.csv"), columns=["PTYPE_FULL"],
                   coefficients=str(d / "non_mandatory_tour_frequency_coefficients_PTYPE_FULL.csv"))
    assert nm.alternatives == [] and all(set(r.cells) == {"*"} for r in nm.rows)


# ---- finalisation --------------------------------------------------------------------------------
def test_finalise_keeps_neighbours_and_rounds():
    rows = finalise([("home", 400.4), ("shop", 30.3), ("shop", 20.0), ("home", 989.3)])
    assert [r["act"] for r in rows] == ["home", "shop", "shop", "home"]
    assert sum(r["duration"] for r in rows) == 1440
    with pytest.raises(InvalidSchedule):
        finalise([("home", 100), ("work", 100)])
    with pytest.raises(InvalidSchedule):
        check_rows([{"act": "work", "start": 0, "end": 1440, "duration": 1440}])
    rows = from_tours([[("work", 480, 1000)], [("shop", 1000, 1030), ("shop", 1030, 1030)]])
    assert [(r["act"], r["duration"]) for r in rows] == [("home", 480), ("work", 520), ("home", 0), ("shop", 30),
                                                         ("shop", 0), ("home", 410)]


def test_tod_realise_respects_window_and_rescale_layout():
    grid = PeriodGrid.uniform(30)
    from composhed.daysim.tod_spec import SPECS

    tod = TourTOD(grid, SPECS["nonmandatory"], buffer=10)
    tw = TimeWindow()
    tw.reserve(500, 700)
    rng = np.random.default_rng(6)
    j = tod.alt_index[(15, 16)]  # 07:30–08:30, overlaps nothing if it ends before 08:10
    draws = [tod.realise(j, tw, rng) for _ in range(300)]
    assert all(450 <= s < e <= 490 and tw.is_free(s, e) for s, e in draws)
    starts = np.array([s for s, _ in draws])
    assert starts.min() < 455 and starts.max() > 475  # uniform over the period, not concentrated
    out, changed = layout_rescale([(100, 900), (500, 1400)], 10)
    assert changed and out[0][1] + 10 <= out[1][0] + 1e-9 and out[-1][1] <= 1439
