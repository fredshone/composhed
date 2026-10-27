"""Unit tests for composhed.common (A2b/c): time windows, tours, choice and duration models, calibration."""

import numpy as np
import pytest

from composhed.common.calibration import Target, calibrate, offset_accessors, share_dict
from composhed.common.choice import MNL, ConditionalLogit, OrderedLogit
from composhed.common.hazard import AFTModel
from composhed.common.lognormal import LogNormalDuration
from composhed.common.periods import IntraPeriodSampler, PeriodGrid
from composhed.common.schedule import InvalidSchedule, check_rows, finalise, from_episodes
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
    # usable windows keep a 10-minute buffer next to tours and 1 minute at day edges
    assert tw.usable_windows(10) == [(1, 470), (1030, 1439)]
    assert tw.largest_window(10) == (1, 470)


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


def test_fragmented_work_merged():
    rows = day(("home", 420), ("work", 240), ("other", 30), ("work", 240), ("home", 510))
    c = ExtractionCounters()
    d = extract_day_structure(rows, counters=c)
    t = d.tours[0]
    assert t.purpose == "work" and t.primary_duration == 480 and t.primary_start == 420
    assert [s.act for s in t.inbound] == ["other"]  # matches CompoShed mandatory_inbound
    assert extract_tour_memberships(rows) == ["mandatory_inbound"]
    assert c.fragmented_tours == 1 and d.total() == 1440


def test_two_mandatory_tours_collapse_vs_keep():
    rows = day(("home", 420), ("work", 240), ("home", 60), ("shop", 20), ("work", 200), ("home", 500))
    keep = extract_day_structure(rows)
    assert [t.purpose for t in keep.tours] == ["work", "work"]
    c = ExtractionCounters()
    col = extract_day_structure(rows, collapse_mandatory_tours=True, counters=c)
    assert [t.purpose for t in col.tours] == ["work", "shop"]
    assert col.tours[0].primary_duration == 440 and col.total() == 1440 and c.collapsed_tours == 1
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
def test_period_grid_and_intra_sampler():
    g = PeriodGrid.with_early_bin(60)
    assert g.n == 20 and g.period_of(0) == 0 and g.period_of(299) == 0 and g.period_of(300) == 1
    assert g.end_period_of(360) == 1 and g.end_period_of(361) == 2
    s = IntraPeriodSampler(PeriodGrid.uniform(30), min_count=5).fit({"k": [485] * 50})
    rng = np.random.default_rng(0)
    assert s.sample("k", 16, 480, 510, rng) == 485
    assert 490 <= s.sample("k", 16, 490, 510, rng) < 510  # infeasible mode → restricted range


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


# ---- calibration --------------------------------------------------------------------------------
def test_calibration_converges_on_misspecified_mnl():
    """A model without covariates calibrated to shares it was not estimated on."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, 1))
    m = MNL().fit(X, rng.choice(["a", "b", "c"], size=3000, p=[0.6, 0.3, 0.1]).tolist(), classes=["a", "b", "c"])
    target = {"a": 0.2, "b": 0.3, "c": 0.5}

    def simulate():
        return {"choice": share_dict(m.sample(X, np.random.default_rng(0)), ["a", "b", "c"])}

    rep = calibrate([Target("choice", target, "share", *offset_accessors(m))], simulate, max_iter=15)
    r = rep.targets["choice"]
    assert rep.converged and r.max_error <= r.tolerance <= 0.02
    assert rep.targets["choice"].history[0] > 0.3


# ---- finalisation --------------------------------------------------------------------------------
def test_finalise_counts_merges_and_rounds():
    from composhed.common.report import GenerationReport

    rep = GenerationReport()
    rows = finalise([("home", 400.4), ("shop", 30.3), ("shop", 20.0), ("home", 989.3)], rep)
    assert [r["act"] for r in rows] == ["home", "shop", "home"]
    assert sum(r["duration"] for r in rows) == 1440 and rep.merges == 1
    with pytest.raises(InvalidSchedule):
        finalise([("home", 100), ("work", 100)])
    with pytest.raises(InvalidSchedule):
        check_rows([{"act": "work", "start": 0, "end": 1440, "duration": 1440}])
    rows = from_episodes([("work", 480, 1000), ("shop", 1100, 1130)])
    assert [r["act"] for r in rows] == ["home", "work", "home", "shop", "home"]


def test_tod_realise_respects_window_and_rescale_layout():
    grid = PeriodGrid.uniform(30)
    tod = TourTOD(grid, buffer=10)
    tod.key = "k"
    tw = TimeWindow()
    tw.reserve(500, 700)
    rng = np.random.default_rng(6)
    j = tod.alt_index[(15, 16)]  # 07:30–08:30, overlaps nothing if it ends before 08:10
    s, e = tod.realise(j, tw, rng)
    assert 450 <= s < e <= 490 and tw.is_free(s, e)
    out, changed = layout_rescale([(100, 900), (500, 1400)], 10)
    assert changed and out[0][1] + 10 <= out[1][0] + 1e-9 and out[-1][1] <= 1439
