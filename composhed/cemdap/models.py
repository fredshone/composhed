"""CEMDAP-aligned component models (SPECIFICATION §5.2) and their design matrices.

| Component | Spec | Model |
|---|---|---|
| work participation | CEM-1 | binary logit (MNL, 2 classes) |
| education participation given no work | CEM-1 | binary logit |
| mandatory start (minutes since 03:00) | CEM-2 | AFT (log-logistic / Weibull by AIC) |
| mandatory duration given start | CEM-2 | AFT with start covariates |
| stops by position: outbound, inbound, pre-tours, post-tours | CEM-3 | 4 ordered logits (0..3) |
| non-mandatory day: number of tours | CEM-7 | ordered logit (0..4) |
| stops per non-mandatory tour | CEM-7 | ordered logit (1..4) |
| stop purpose | CEM-4 | MNL over 5 discretionary types |
| stop duration by purpose | CEM-5 | AFT per purpose |
| home stay between tours / first departure | CEM-8 | AFT |
| AB-2 ablation timing | — | KDE start per employment + log-normal duration |
"""

import numpy as np
from scipy.stats import gaussian_kde

from composhed.cemdap.data import (
    DAY,
    HOME_SIDES,
    MAND_START_ORIGIN,
    MAX_NM_TOURS,
    MAX_POSITION_COUNT,
    MAX_TOUR_STOPS,
    POSITIONS,
    CemdapRecord,
)
from composhed.common.choice import MNL, OrderedLogit
from composhed.common.hazard import AFTModel
from composhed.common.lognormal import LogNormalDuration
from composhed.data import DISC_ACTS

PURPOSES = sorted(DISC_ACTS)
COUNT_POSITIONS = ["n_out", "n_in", "n_pre", "n_post"]
TOUR_POSITIONS = ["pre_tour", "post_tour", "nm_tour"]


def _onehot(value, levels: list) -> list[float]:
    """Dummies for levels[1:] (first level is the reference)."""
    return [1.0 if value == lv else 0.0 for lv in levels[1:]]


# ---- design rows (shared by training and generation) ------------------------------
def x_start(xl, is_edu):
    return np.concatenate([xl, [float(is_edu)]])


def x_duration(xl, is_edu, start):
    s = start / DAY
    return np.concatenate([xl, [float(is_edu), s, s * s]])


def x_counts(xl, is_edu, start, duration):
    return np.concatenate([xl, [float(is_edu), duration / DAY, start / DAY, (DAY - start - duration) / DAY]])


def x_tour_size(xl, position, tour_index):
    return np.concatenate([xl, _onehot(position, TOUR_POSITIONS), [min(tour_index, 3) / 3.0]])


def x_stop(xl, position, index, residual):
    return np.concatenate([xl, _onehot(position, POSITIONS), [min(index, 3) / 3.0, residual / DAY]])


def x_home(xl, side, residual):
    return np.concatenate([xl, _onehot(side, HOME_SIDES), [residual / DAY]])


def x_first_departure(xl, n_tours):
    return np.concatenate([xl, [n_tours / MAX_NM_TOURS]])


class CemdapModels:
    def __init__(self, independent_timing: bool = False) -> None:
        self.independent_timing = independent_timing

    # ---- design matrices ---------------------------------------------------------
    @staticmethod
    def designs(records: list[CemdapRecord], X: np.ndarray) -> dict:
        d: dict = {k: ([], []) for k in (
            "work", "edu", "start", "duration", "counts", "nm_tours", "tour_size", "stop_purpose",
            "home", "first_departure",
        )}
        d["stop_duration"] = {p: ([], []) for p in PURPOSES}
        d["counts_y"] = {k: [] for k in COUNT_POSITIONS}
        d["employment"] = []
        for r, xl in zip(records, X):
            is_work = r.mand_type == "work"
            d["work"][0].append(xl)
            d["work"][1].append(int(is_work))
            if not is_work:
                d["edu"][0].append(xl)
                d["edu"][1].append(int(r.mand_type == "education"))
            if r.mand_type is not None:
                is_edu = r.mand_type == "education"
                d["start"][0].append(x_start(xl, is_edu))
                d["start"][1].append(max(r.mand_start - MAND_START_ORIGIN, 1))
                d["duration"][0].append(x_duration(xl, is_edu, r.mand_start))
                d["duration"][1].append(max(r.mand_duration, 1))
                d["counts"][0].append(x_counts(xl, is_edu, r.mand_start, r.mand_duration))
                for k in COUNT_POSITIONS:
                    d["counts_y"][k].append(min(getattr(r, k), MAX_POSITION_COUNT))
                d["employment"].append((r.labels["employment"], r.mand_start))
            else:
                d["nm_tours"][0].append(xl)
                d["nm_tours"][1].append(min(r.n_nm_tours, MAX_NM_TOURS))
                if r.first_departure is not None:
                    d["first_departure"][0].append(x_first_departure(xl, min(r.n_nm_tours, MAX_NM_TOURS)))
                    d["first_departure"][1].append(max(r.first_departure, 1))
            for pos, ti, n in r.tour_sizes:
                d["tour_size"][0].append(x_tour_size(xl, pos, ti))
                d["tour_size"][1].append(min(n, MAX_TOUR_STOPS))
            for s in r.stops:
                xs = x_stop(xl, s["position"], s["index"], s["residual"])
                d["stop_purpose"][0].append(xs)
                d["stop_purpose"][1].append(s["purpose"])
                d["stop_duration"][s["purpose"]][0].append(xs)
                d["stop_duration"][s["purpose"]][1].append(max(s["duration"], 1))
            for h in r.home_stays:
                d["home"][0].append(x_home(xl, h["side"], h["residual"]))
                d["home"][1].append(max(h["duration"], 1))
        return d

    @staticmethod
    def _arr(rows, width: int) -> np.ndarray:
        return np.array(rows, dtype=float) if rows else np.zeros((0, width))

    # ---- fit ---------------------------------------------------------------------
    def fit(self, records: list[CemdapRecord], X: np.ndarray, feature_names: list[str]) -> "CemdapModels":
        L = X.shape[1]
        d = self.designs(records, X)
        A = self._arr
        self.work = MNL().fit(A(d["work"][0], L), d["work"][1], classes=[0, 1], feature_names=feature_names)
        self.edu = MNL().fit(A(d["edu"][0], L), d["edu"][1], classes=[0, 1], feature_names=feature_names)
        if d["start"][1]:
            self.start = AFTModel.fit_best(A(d["start"][0], L + 1), np.array(d["start"][1]))
            self.duration = AFTModel.fit_best(A(d["duration"][0], L + 3), np.array(d["duration"][1]))
            Xc = A(d["counts"][0], L + 4)
            self.counts = {
                k: OrderedLogit(MAX_POSITION_COUNT).fit(Xc, d["counts_y"][k]) for k in COUNT_POSITIONS
            }
            # AB-2: CompoShed-style independent timing.
            self.kde_start = {}
            by_emp: dict[str, list[float]] = {}
            for emp, s in d["employment"]:
                by_emp.setdefault(emp, []).append(float(s))
            all_starts = [s for _, s in d["employment"]]
            for emp, vals in list(by_emp.items()) + [("__all__", all_starts)]:
                vals = np.asarray(vals, dtype=float)
                if len(vals) >= 5 and np.std(vals) > 0:
                    self.kde_start[emp] = gaussian_kde(vals)
            self.const_start = float(np.mean(all_starts))
            Xd = A(d["start"][0], L + 1)
            self.indep_duration = LogNormalDuration().fit(Xd, np.array(d["duration"][1]))
        else:
            self.start = self.duration = None
            self.counts = {}
        self.nm_tours = OrderedLogit(MAX_NM_TOURS).fit(A(d["nm_tours"][0], L), d["nm_tours"][1])
        self.tour_size = OrderedLogit(MAX_TOUR_STOPS).fit(A(d["tour_size"][0], L + 3), d["tour_size"][1])
        ns = L + len(POSITIONS) - 1 + 2
        self.stop_purpose = MNL().fit(A(d["stop_purpose"][0], ns), d["stop_purpose"][1], classes=PURPOSES)
        all_stop_X = A(d["stop_purpose"][0], ns)
        all_stop_t = np.array([v for p in PURPOSES for v in d["stop_duration"][p][1]], dtype=float)
        self.stop_duration = {}
        for p in PURPOSES:
            Xp, tp = d["stop_duration"][p]
            # Purposes with too few stops borrow the pooled model.
            if len(tp) >= 10:
                self.stop_duration[p] = AFTModel.fit_best(A(Xp, ns), np.array(tp))
            elif len(all_stop_t):
                self.stop_duration[p] = AFTModel.fit_best(all_stop_X, all_stop_t)
        self.home = AFTModel.fit_best(A(d["home"][0], L + 3), np.array(d["home"][1])) if d["home"][1] else None
        self.first_departure = (
            AFTModel.fit_best(A(d["first_departure"][0], L + 1), np.array(d["first_departure"][1]))
            if d["first_departure"][1] else None
        )
        return self

    # ---- validate ------------------------------------------------------------------
    def validate(self, records: list[CemdapRecord], X: np.ndarray) -> dict[str, dict]:
        L = X.shape[1]
        d = self.designs(records, X)
        A = self._arr
        out = {
            "CEM-1 work participation": self.work.validate(A(d["work"][0], L), d["work"][1]),
            "CEM-1 education participation": self.edu.validate(A(d["edu"][0], L), d["edu"][1]),
        }
        if self.start is not None and d["start"][1]:
            out["CEM-2 mandatory start (AFT)"] = self.start.validate(A(d["start"][0], L + 1), np.array(d["start"][1]))
            out["CEM-2 mandatory duration | start (AFT)"] = self.duration.validate(
                A(d["duration"][0], L + 3), np.array(d["duration"][1]))
            Xc = A(d["counts"][0], L + 4)
            for k in COUNT_POSITIONS:
                out[f"CEM-3 stops {k[2:]}"] = self.counts[k].validate(Xc, d["counts_y"][k])
        out["CEM-7 non-mandatory tours"] = self.nm_tours.validate(A(d["nm_tours"][0], L), d["nm_tours"][1])
        out["CEM-7 stops per tour"] = self.tour_size.validate(A(d["tour_size"][0], L + 3), d["tour_size"][1])
        ns = L + len(POSITIONS) - 1 + 2
        out["CEM-4 stop purpose"] = self.stop_purpose.validate(A(d["stop_purpose"][0], ns), d["stop_purpose"][1])
        groups = {}
        for p in PURPOSES:
            Xp, tp = d["stop_duration"][p]
            if tp and p in self.stop_duration:
                groups[p] = self.stop_duration[p].validate(A(Xp, ns), np.array(tp))
        out["CEM-5 stop duration (AFT)"] = {"n": sum(m["n"] for m in groups.values()), "groups": groups}
        if self.home is not None and d["home"][1]:
            out["CEM-8 home stay (AFT)"] = self.home.validate(A(d["home"][0], L + 3), np.array(d["home"][1]))
        if self.first_departure is not None and d["first_departure"][1]:
            out["CEM-8 first departure (AFT)"] = self.first_departure.validate(
                A(d["first_departure"][0], L + 1), np.array(d["first_departure"][1]))
        return out

    def coef_tables(self) -> dict:
        out = {"work": self.work.coef_table(), "education": self.edu.coef_table(),
               "stop_purpose": self.stop_purpose.coef_table(), "nm_tours": self.nm_tours.coef_table(),
               "tour_size": self.tour_size.coef_table()}
        if self.start is not None:
            out["mand_start"] = self.start.coef_table()
            out["mand_duration"] = self.duration.coef_table()
            out.update({f"count_{k}": m.coef_table() for k, m in self.counts.items()})
        out.update({f"stop_duration_{p}": m.coef_table() for p, m in self.stop_duration.items()})
        return out
