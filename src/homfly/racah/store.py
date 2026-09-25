"""Storage and loading of precomputed Racah matrices (user tables).

Layout (see docs/RACAH_FORMAT.md):

    data/racah/
        inclusive/<Y>_<R>_<Z>.json     e.g.  21_1_32.json  for U[[2,1],[1],[1]->[3,2]]
        exclusive/<R>.json             S and S̄ for R ⊗ R ⊗ R̄ -> R
        eigenvalues/<R>.json           signs eps_X and multiplicity labels in R ⊗ R

Each JSON file stores a RacahMatrix (racah.model) with entries as lists of
terms {"coef": "<num>/<den>", "sqrt": "<num>/<den>"} whose strings are Laurent
polynomials in A, q (parse_laurent syntax).  Adapters convert other formats
(Mathematica .m / Maple / text dumps) into this canonical JSON once; the
engines only ever read the canonical form.

Status: loader for the canonical format implemented; the adapter for the
user's existing tables is to be written once a sample file is available
(`adapters.py`, ROADMAP M2).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from ..algebra.laurent import parse_laurent
from ..reps.partitions import P
from .model import Entry, RacahMatrix, RationalFunction, Term
from .provider import RacahProvider

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "racah")


def rep_key(R):
    return "".join(map(str, R)) if all(x < 10 for x in R) else "-".join(map(str, R))


def _rf(s):
    if s is None:
        return None
    if "/" in s and s.count("/") == 1 and not s.strip().startswith("("):
        n, d = s.split("/")
    elif s.rstrip().endswith(")") and ")/(" in s:
        n, d = s.split(")/(")
        n, d = n + ")", "(" + d
    else:
        n, d = s, "1"
    return RationalFunction(parse_laurent(n), parse_laurent(d))


def entry_from_json(obj):
    if isinstance(obj, (int, str)):
        obj = [{"coef": str(obj)}]
    return Entry([Term(_rf(t["coef"]), _rf(t.get("sqrt"))) for t in obj])


def matrix_from_json(d):
    return RacahMatrix(
        kind=d["kind"],
        reps=tuple(tuple(x) if isinstance(x, list) else x for x in d["reps"]),
        rows=[tuple(tuple(x) if isinstance(x, list) else x for x in r) for r in d["rows"]],
        cols=[tuple(tuple(x) if isinstance(x, list) else x for x in c) for c in d["cols"]],
        entries=[[entry_from_json(e) for e in row] for row in d["entries"]],
        convention=d.get("convention", "MMM"),
    )


class RacahStore(RacahProvider):
    """RacahProvider backed by canonical JSON files."""

    def __init__(self, root=DEFAULT_DIR):
        self.root = os.path.abspath(root)

    @lru_cache(maxsize=None)
    def _load(self, sub, name):
        path = os.path.join(self.root, sub, name + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            return json.load(f)

    def available(self):
        out = {}
        for sub in ("inclusive", "exclusive", "eigenvalues"):
            d = os.path.join(self.root, sub)
            out[sub] = sorted(x[:-5] for x in os.listdir(d)) if os.path.isdir(d) else []
        return out

    def decompose(self, Y, R):
        from ..reps.characters import lr_product
        out = []
        for Z, mult in sorted(lr_product(P(Y), P(R)).items()):
            out.extend((Z, i) for i in range(mult))
        return out

    def eigenvalues(self, R, F, A, q):
        from ..reps.partitions import kappa
        d = self._load("eigenvalues", rep_key(R))
        out = {}
        for item in d["eigenvalues"]:
            X = tuple(item["X"])
            out[(X, item.get("mult", 0))] = item["sign"] * q ** (kappa(X) - 2 * kappa(R))
        return out

    def inclusive(self, Y, R, Z, F, A, q):
        name = "%s_%s_%s" % (rep_key(Y), rep_key(R), rep_key(Z))
        M = matrix_from_json(self._load("inclusive", name))
        U = M.evaluate(F, A, q)
        return M.rows, M.cols, U, None

    def exclusive(self, R, F, A, q):
        d = self._load("exclusive", rep_key(R))
        S = matrix_from_json(d["S"])
        Sb = matrix_from_json(d["Sbar"])
        return S, Sb
