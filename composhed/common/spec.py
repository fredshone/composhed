"""Spec-driven logits: utility terms written as expressions, as in ActivitySim spec files.

A spec is a list of rows. Each row has an expression over named variables and
the coefficient it uses for each alternative (``cells``: alternative →
coefficient name or number). As in ActivitySim:

* an expression is evaluated for every chooser (person variables), or for
  every chooser × alternative when alternative attributes appear in it
  (``interaction_simulate``);
* the same coefficient name may appear in several rows and alternatives, and
  then has one value;
* a number in a cell is a fixed utility contribution; a coefficient whose
  value is at most -999 makes the alternative unavailable.

Expressions use pandas ``eval`` syntax: ``&``, ``|`` and ``~`` bind like
``and`` / ``or`` / ``not`` (looser than comparisons). They are compiled once to
numpy code.

Rows that use a variable the caller cannot provide are **dropped** and listed
(``dropped_rows``). Coefficients whose design column has no variation within
the choice sets of the estimation data are fixed at 0 and listed
(``unidentified``). Every other coefficient is re-estimated.
"""

import ast
import csv
import io
import tokenize
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

UNAVAILABLE = -999.0


# ---- expressions --------------------------------------------------------------------------------
class _ToNumpy(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        fn = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
        out = node.values[0]
        for v in node.values[1:]:
            out = ast.Call(ast.Attribute(ast.Name("_np", ast.Load()), fn, ast.Load()), [out, v], [])
        return out

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.Call(ast.Attribute(ast.Name("_np", ast.Load()), "logical_not", ast.Load()), [node.operand], [])
        return node


def _pandas_to_python(expr: str) -> str:
    """Rewrite ``&``, ``|``, ``~`` as ``and``, ``or``, ``not`` so Python's precedence matches pandas eval."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(expr.strip()).readline):
        if tok.type == tokenize.OP and tok.string in ("&", "|", "~"):
            out.append((tokenize.NAME, {"&": "and", "|": "or", "~": "not"}[tok.string]))
        else:
            out.append((tok.type, tok.string))
    return tokenize.untokenize(out)


@dataclass
class Expr:
    source: str
    names: set
    code: object

    def __call__(self, env: dict) -> np.ndarray:
        return np.asarray(eval(self.code, {"_np": np, "__builtins__": {}}, env), dtype=float)

    def __reduce__(self):  # code objects do not pickle: rebuild from the source
        return compile_expr, (self.source,)


def compile_expr(source: str) -> Expr:
    tree = ast.parse(_pandas_to_python(source), mode="eval")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    tree = ast.fix_missing_locations(_ToNumpy().visit(tree))
    return Expr(source, names, compile(tree, "<spec>", "eval"))


# ---- specs ----------------------------------------------------------------------------------------
@dataclass
class Row:
    label: str
    expr: Expr
    cells: dict  # alternative -> coefficient name (str) or fixed value (float)


@dataclass
class Spec:
    alternatives: list
    rows: list
    fixed: dict = field(default_factory=dict)  # coefficient name -> fixed value (constrained in the source)
    dropped_rows: list = field(default_factory=list)

    def add(self, label: str, expression: str, cells: dict) -> "Spec":
        self.rows.append(Row(label, compile_expr(expression), dict(cells)))
        return self

    def restrict(self, available: set) -> "Spec":
        """Drop rows that use a variable not in ``available`` (recorded in ``dropped_rows``)."""
        keep = []
        for r in self.rows:
            missing = sorted(r.expr.names - available)
            if missing:
                self.dropped_rows.append({"label": r.label, "expression": r.expr.source, "missing": missing})
            else:
                keep.append(r)
        self.rows = keep
        return self

    @property
    def coefficient_names(self) -> list[str]:
        names = []
        for r in self.rows:
            for c in r.cells.values():
                if isinstance(c, str) and c not in self.fixed and c not in names:
                    names.append(c)
        return names


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def read_coefficients(path: str) -> tuple[dict, dict]:
    """(values, constrained) from an ActivitySim coefficients file; a missing ``constrain`` column means free."""
    values, constrained = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            name = r["coefficient_name"]
            if not name or name.startswith("#") or not str(r["value"]).strip():
                continue
            values[name] = float(r["value"])
            constrained[name] = str(r.get("constrain", "F")).strip().upper() == "T"
    return values, constrained


def read_spec(path: str, alternatives: list | None = None, columns: list | None = None,
              coefficients: str | None = None) -> Spec:
    """An ActivitySim ``simple_simulate`` / ``interaction_simulate`` spec file.

    ``columns``: the coefficient columns to read (default: every column after
    ``Expression``); for ``interaction_simulate`` specs with one coefficient
    column per segment, pass that one column and ``alternatives=None``: its
    coefficient is applied to the expression value of every alternative (cell key
    ``"*"``). Constrained coefficients keep their file value; others are
    re-estimated. Blank cells contribute nothing.
    """
    values, constrained = read_coefficients(coefficients) if coefficients else ({}, {})
    with open(path) as f:
        rows = [r for r in csv.reader(f)]
    header = rows[0]
    ei = header.index("Expression")
    cols = columns or header[ei + 1:]
    ci = [header.index(c) for c in cols]
    interaction = alternatives is None and columns is not None
    spec = Spec(alternatives=[] if interaction else list(alternatives or cols), rows=[])
    for k, r in enumerate(rows[1:]):
        if not r or r[0].startswith("#") or len(r) <= ei or not r[ei].strip():
            continue
        label = r[0] if r[0] and r[0] != r[ei] else f"row{k + 1}"
        cells = {}
        for c, i in zip(cols, ci):
            cell = r[i].strip() if i < len(r) else ""
            if not cell:
                continue
            key = "*" if interaction else c
            fixed = _num(cell)
            if fixed is not None:
                if fixed != 0:
                    cells[key] = fixed
            elif constrained.get(cell, False) or values.get(cell, 0.0) <= UNAVAILABLE:
                if values.get(cell, 0.0) != 0:
                    cells[key] = values[cell]
            else:
                cells[key] = cell
        if cells:
            spec.rows.append(Row(label, compile_expr(r[ei]), cells))
    return spec


# ---- design ---------------------------------------------------------------------------------------
def design(spec: Spec, env: dict, n: int, coef_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(D (n, J, C) float32, offset (n, J)): utility = D @ beta + offset.

    ``env`` holds person variables as (n, 1) arrays and alternative attributes as
    (1, J) arrays, so an expression broadcasts to (n, 1) or (n, J).
    """
    J = len(spec.alternatives)
    cidx = {c: i for i, c in enumerate(coef_names)}
    D = np.zeros((n, J, len(coef_names)), dtype=np.float32)
    off = np.zeros((n, J))
    for r in spec.rows:
        x = np.atleast_2d(r.expr(env))
        x = np.broadcast_to(x, (n, x.shape[1]))
        for alt, c in r.cells.items():
            for j in (range(J) if alt == "*" else [spec.alternatives.index(alt)]):
                v = x[:, j] if x.shape[1] > 1 else x[:, 0]
                if isinstance(c, str):
                    D[:, j, cidx[c]] += v
                else:
                    off[:, j] += v * c
    return D, off


class SpecLogit:
    """Multinomial logit whose utilities come from a ``Spec`` (see module docstring)."""

    def __init__(self, spec: Spec, l2: float = 1e-4, maxiter: int = 500) -> None:
        self.spec = spec
        self.l2 = l2
        self.maxiter = maxiter
        self.coef_names_ = spec.coefficient_names
        self.beta_ = np.zeros(len(self.coef_names_))

    @property
    def classes_(self) -> list:
        return self.spec.alternatives

    def _design(self, env: dict, n: int):
        D, off = design(self.spec, env, n, self.coef_names_)
        avail = off > UNAVAILABLE / 2
        return D, np.where(avail, off, 0.0), avail

    def fit(self, env: dict, n: int, y: list, avail: np.ndarray | None = None) -> "SpecLogit":
        D, off, av = self._design(env, n)
        if avail is not None:
            av = av & avail
        yi = np.array([self.classes_.index(v) for v in y], dtype=int)
        ok = av[np.arange(n), yi]
        self.n_unavailable_chosen_ = int((~ok).sum())
        D, off, av, yi = D[ok], off[ok], av[ok], yi[ok]
        N = len(yi)
        # Unidentified: no variation of the column within any chooser's available alternatives.
        Dm = np.where(av[:, :, None], D, np.nan)
        spread = np.nanmax(Dm, axis=1) - np.nanmin(Dm, axis=1)
        free = np.nansum(spread, axis=0) > 0
        self.unidentified_ = [c for c, f in zip(self.coef_names_, free) if not f]
        Df = D[:, :, free]
        Y = np.zeros(av.shape)
        Y[np.arange(N), yi] = 1.0

        def obj(b):
            V = np.einsum("njc,c->nj", Df, b) + off
            V = np.where(av, V, -np.inf)
            logP = V - logsumexp(V, axis=1, keepdims=True)
            P = np.exp(logP)
            ll = logP[np.arange(N), yi].sum()
            g = np.einsum("njc,nj->c", Df, Y - P)
            return -ll / N + 0.5 * self.l2 * b @ b, -g / N + self.l2 * b

        res = minimize(obj, np.zeros(int(free.sum())), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.maxiter})
        self.beta_ = np.zeros(len(self.coef_names_))
        self.beta_[free] = res.x
        self.converged_ = bool(res.success)
        self.n_obs_ = N
        self.llf_ = float(-res.fun * N + 0.5 * self.l2 * N * res.x @ res.x)
        nalt = av.sum(axis=1)
        self.ll_equal_ = float(-np.log(nalt).sum())
        counts = np.bincount(yi, minlength=len(self.classes_)).astype(float)
        self.train_shares_ = counts / max(N, 1)
        return self

    def predict_proba(self, env: dict, n: int, avail: np.ndarray | None = None) -> np.ndarray:
        D, off, av = self._design(env, n)
        if avail is not None:
            av = av & avail
        V = np.where(av, np.einsum("njc,c->nj", D, self.beta_) + off, -np.inf)
        with np.errstate(invalid="ignore"):
            P = np.exp(V - logsumexp(V, axis=1, keepdims=True))
        return np.nan_to_num(P)

    def sample(self, env: dict, n: int, rng: np.random.Generator, avail: np.ndarray | None = None) -> list:
        P = self.predict_proba(env, n, avail)
        c = np.cumsum(P, axis=1)
        u = rng.random(n) * c[:, -1]
        idx = np.minimum((c <= u[:, None]).sum(axis=1), P.shape[1] - 1)
        return [self.classes_[i] for i in idx]

    @property
    def n_params(self) -> int:
        return len(self.coef_names_) - len(getattr(self, "unidentified_", []))

    def validate(self, env: dict, n: int, y: list, avail: np.ndarray | None = None) -> dict:
        P = self.predict_proba(env, n, avail)
        yi = np.array([self.classes_.index(v) for v in y], dtype=int)
        p = P[np.arange(n), yi]
        ok = p > 0
        ll = float(np.log(p[ok]).sum())
        return {"n": int(ok.sum()), "n_unavailable_chosen": int((~ok).sum()), "n_params": self.n_params,
                "log_likelihood": ll, "rho2_equal_shares": 1.0 - ll / self.ll_equal_ if self.ll_equal_ < 0 else None,
                "accuracy": float((P.argmax(axis=1) == yi)[ok].mean()) if ok.any() else float("nan"),
                "shares": {str(c): {"obs": float((yi == j).mean()), "pred": float(P[:, j].mean())}
                           for j, c in enumerate(self.classes_)},
                "dropped_rows": [d["label"] for d in self.spec.dropped_rows],
                "unidentified": list(getattr(self, "unidentified_", []))}

    def coef_table(self) -> dict:
        return {c: round(float(b), 4) for c, b in zip(self.coef_names_, self.beta_)}
