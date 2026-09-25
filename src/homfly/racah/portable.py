"""Family-P ("portable", racah_homfly v0.7) inclusive 3-strand blocks.

For every Q ∈ R⊗R⊗R the data give rational matrices R1, R2 (braid generators
sigma_1, sigma_2 on the multiplicity space of Q, *including* the topological
framing factor) and their inverses.  Entries are num/den polynomials in (A, q)
with integer coefficients (converted once by scripts/import_racah_portable.py).

    H_R(beta) = sum_Q dim_q(Q) Tr_Q(rho(beta)) / dim_q(R)

evaluated at the point given by ``POINT_MAP`` (the family's convention relative
to our STANDARD one; pinned in tests/test_portable.py).
"""
from __future__ import annotations

import gzip
import json
import os
from functools import lru_cache

from ..algebra.fields import BadPoint

DEFAULT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                            "data", "racah", "portable"))


def rep_key(R):
    return "".join(map(str, R))


class _Poly:
    """Integer polynomial in (A, q) with a small power cache per evaluation."""
    __slots__ = ("terms",)

    def __init__(self, terms):
        self.terms = [(i, j, c) for i, j, c in terms]

    def eval(self, Ap, qp, zero):
        s = zero
        for i, j, c in self.terms:
            s = s + Ap[i] * qp[j] * c
        return s


class PortableChannel:
    def __init__(self, d):
        self.Q = tuple(d["Q"])
        self.dim = d["dim"]
        self.labels = d["labels"]
        self.mats = {k: [[(_Poly(n), _Poly(dd)) for n, dd in row] for row in d[k]]
                     for k in ("R1", "R2", "R1_inv", "R2_inv")}
        self.maxA = max(max((i for i, _, _ in p.terms), default=0)
                        for M in self.mats.values() for row in M for e in row for p in e)
        self.maxq = max(max((j for _, j, _ in p.terms), default=0)
                        for M in self.mats.values() for row in M for e in row for p in e)

    def evaluate(self, F, A, q):
        zero, one = F.zero, F.one
        Ap = [one]
        for _ in range(self.maxA):
            Ap.append(Ap[-1] * A)
        qp = [one]
        for _ in range(self.maxq):
            qp.append(qp[-1] * q)
        out = {}
        for k, M in self.mats.items():
            rows = []
            for row in M:
                r = []
                for n, d in row:
                    dv = d.eval(Ap, qp, zero)
                    if dv == 0:
                        raise BadPoint("pole of a Racah entry")
                    r.append(n.eval(Ap, qp, zero) / dv)
                rows.append(r)
            out[k] = rows
        return out


    def evaluate_modp(self, p, A, q, keys=("R1", "R2", "R1_inv", "R2_inv")):
        """Fast path: plain ints mod p (A, q ints).  Returns {key: int matrix}."""
        Ap = [1]
        for _ in range(self.maxA):
            Ap.append(Ap[-1] * A % p)
        qp = [1]
        for _ in range(self.maxq):
            qp.append(qp[-1] * q % p)
        out = {}
        for k in keys:
            rows = []
            for row in self.mats[k]:
                r = []
                for n, d in row:
                    dv = sum(Ap[i] * qp[j] * c for i, j, c in d.terms) % p
                    if dv == 0:
                        raise BadPoint("pole of a Racah entry")
                    nv = sum(Ap[i] * qp[j] * c for i, j, c in n.terms)
                    r.append(nv * pow(dv, -1, p) % p)
                rows.append(r)
            out[k] = rows
        return out


def matmul_mod(X, Y, p):
    Yt = list(zip(*Y))
    return [[sum(a * b for a, b in zip(row, col)) % p for col in Yt] for row in X]


class PortableInclusive:
    def __init__(self, R, root=DEFAULT_DIR):
        self.R = tuple(R)
        path = os.path.join(root, "inclusive_%s.json.gz" % rep_key(self.R))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        self.channels = [PortableChannel(c) for c in d["channels"]]


@lru_cache(maxsize=None)
def load_inclusive(R, root=DEFAULT_DIR):
    return PortableInclusive(tuple(R), root)


def available(root=DEFAULT_DIR):
    if not os.path.isdir(root):
        return []
    out = []
    for f in sorted(os.listdir(root)):
        if f.startswith("inclusive_") and f.endswith(".json.gz"):
            out.append(tuple(int(c) for c in f[len("inclusive_"):-len(".json.gz")]))
    return out
