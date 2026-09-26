"""Two-bridge knots from exclusive Racah data of the release v1.0.0.

Family P (|R| <= 5, [6], [1^6]) -- C and D̄²; families H (other 6-box reps) --
S̄ and T̄² via value_sbar.  Family P formula:

For the all-even 4-plat continued fraction cf = (a_1, ..., a_k)
(TwoBridge.even_cf):

    M = prod_i ( C · D̄²^{a_i/2 - 1} ) · C,        H_R = dim_q(R) · M[vac, vac],

where C = S T^{-1} V and D̄² = T̄² (diagonal) are rational in (A, q)
(racah_homfly v0.7 conventions).  Covers all 362 two-bridge knots <= 12
crossings in every R with exclusive data (|R| <= 5, [6], [1^6]).

Chirality: KnotInfo's [p, q] does not always match the package's sign
convention, so the orientation is fixed per knot by comparing the fundamental
result with the KnotInfo HOMFLY (``chirality``), and then used for every R.
"""
from __future__ import annotations

import gzip
import json
import os
import random
from functools import lru_cache

from ..algebra.fields import GF, BadPoint
from ..algebra.linalg import matmul
from ..knots.families import DoubleBraid, TwoBridge
from ..racah.portable import DEFAULT_DIR, _Poly, matmul_mod, rep_key
from ..reps.partitions import P
from ..reps.qdim import qdim
from .base import Method

# family-P point for our STANDARD (A, q); pinned by tests/test_two_bridge.py
def _point(A, q):
    return A ** -1, q ** -1


class Exclusive:
    def __init__(self, R, root=DEFAULT_DIR):
        path = os.path.join(root, "exclusive_%s.json.gz" % rep_key(R))
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        self.R = tuple(R)
        self.vac = d["vacuum_index"]
        self.C = [[(_Poly(n), _Poly(dd)) for n, dd in row] for row in d["C"]]
        self.D2 = [(_Poly(n), _Poly(dd)) for n, dd in d["Dbar2"]]
        polys = [p for row in self.C for e in row for p in e] + [p for e in self.D2 for p in e]
        self.maxA = max(max((i for i, _, _ in p.terms), default=0) for p in polys)
        self.maxq = max(max((j for _, j, _ in p.terms), default=0) for p in polys)

    def evaluate(self, F, A, q):
        one, zero = F.one, F.zero
        Ap, qp = [one], [one]
        for _ in range(self.maxA):
            Ap.append(Ap[-1] * A)
        for _ in range(self.maxq):
            qp.append(qp[-1] * q)

        def ev(e):
            d = e[1].eval(Ap, qp, zero)
            if d == 0:
                raise BadPoint("pole in exclusive data")
            return e[0].eval(Ap, qp, zero) / d
        return [[ev(e) for e in row] for row in self.C], [ev(e) for e in self.D2]


    def evaluate_modp(self, p, A, q):
        Ap, qp = [1], [1]
        for _ in range(self.maxA):
            Ap.append(Ap[-1] * A % p)
        for _ in range(self.maxq):
            qp.append(qp[-1] * q % p)

        def ev(e):
            d = sum(Ap[i] * qp[j] * c for i, j, c in e[1].terms) % p
            if d == 0:
                raise BadPoint("pole in exclusive data")
            return sum(Ap[i] * qp[j] * c for i, j, c in e[0].terms) * pow(d, -1, p) % p
        return [[ev(e) for e in row] for row in self.C], [ev(e) for e in self.D2]


class SbarData:
    """Families H/G of the release: rational S̄ (S̄² = 1) and diagonal T̄²
    (including the framing t0²), stored by scripts/import_racah_sbar.py."""

    def __init__(self, R, root=DEFAULT_DIR):
        path = os.path.join(root, "sbar_%s.json.gz" % rep_key(R))
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        self.R = tuple(R)
        self.family = d["family"]
        self.vac = d["vacuum_index"]
        self.t0sq = tuple(d["t0_squared"])
        self.S = [[(_Poly(n), _Poly(dd)) for n, dd in row] for row in d["Sbar"]]
        self.T2 = [tuple(x) for x in d["Tbar2"]]      # (A-exp, q-exp) monomials
        polys = [p for row in self.S for e in row for p in e]
        self.minA = min(min((i for i, _, _ in p.terms), default=0) for p in polys)
        self.minq = min(min((j for _, j, _ in p.terms), default=0) for p in polys)
        self.maxA = max(max((i for i, _, _ in p.terms), default=0) for p in polys)
        self.maxq = max(max((j for _, j, _ in p.terms), default=0) for p in polys)

    def _flat(self):
        """Flat numpy term arrays for vectorised evaluation (built lazily)."""
        if getattr(self, "_np", None) is None:
            import numpy as np
            parts = {}
            for which in (0, 1):
                coef, ai, qj, ptr = [], [], [], [0]
                for row in self.S:
                    for e in row:
                        terms = e[which].terms or [(0, 0, 0)]
                        for i, j, c in terms:
                            coef.append(c)
                            ai.append(i - self.minA)
                            qj.append(j - self.minq)
                        ptr.append(len(coef))
                parts[which] = (np.array(coef, dtype=object), np.array(ai), np.array(qj),
                                np.array(ptr[:-1]))
            self._np = parts
        return self._np

    def evaluate_np(self, p, A, q):
        """(S, T2) as int64 numpy arrays mod p (p < 2^31)."""
        import numpy as np
        flat = self._flat()
        Ai, qi = pow(A, -1, p), pow(q, -1, p)
        nA, nq = self.maxA - self.minA + 1, self.maxq - self.minq + 1
        Ap = np.empty(nA, dtype=np.int64)
        Ap[0] = pow(Ai, -self.minA, p) if self.minA < 0 else pow(A, self.minA, p)
        for k in range(1, nA):
            Ap[k] = Ap[k - 1] * A % p
        qp = np.empty(nq, dtype=np.int64)
        qp[0] = pow(qi, -self.minq, p) if self.minq < 0 else pow(q, self.minq, p)
        for k in range(1, nq):
            qp[k] = qp[k - 1] * q % p
        vals = []
        for which in (0, 1):
            coef, ai, qj, ptr = flat[which]
            cm = self._coef_mod.get((which, p)) if hasattr(self, "_coef_mod") else None
            if cm is None:
                cm = np.array([int(c) % p for c in coef], dtype=np.int64)
                if not hasattr(self, "_coef_mod"):
                    self._coef_mod = {}
                self._coef_mod[(which, p)] = cm
            t = cm * Ap[ai] % p * qp[qj] % p
            vals.append(np.add.reduceat(t, ptr) % p)
        num, den = vals
        if (den == 0).any():
            raise BadPoint("pole in S̄ data")
        n = len(self.S)
        inv = np.array([pow(int(x), -1, p) for x in den], dtype=np.int64)
        S = (num * inv % p).reshape(n, n)
        T2 = [pow(A, a, p) if a >= 0 else pow(Ai, -a, p) for a, _ in self.T2]
        T2 = [t * (pow(q, b, p) if b >= 0 else pow(qi, -b, p)) % p for t, (_, b) in zip(T2, self.T2)]
        return S, np.array(T2, dtype=np.int64)

    def evaluate_modp(self, p, A, q):
        Ai, qi = pow(A, -1, p), pow(q, -1, p)
        Ap = {0: 1}
        for k in range(1, self.maxA + 1):
            Ap[k] = Ap[k - 1] * A % p
        for k in range(1, -self.minA + 1):
            Ap[-k] = Ap[-k + 1] * Ai % p
        qp = {0: 1}
        for k in range(1, self.maxq + 1):
            qp[k] = qp[k - 1] * q % p
        for k in range(1, -self.minq + 1):
            qp[-k] = qp[-k + 1] * qi % p

        def ev(e):
            d = sum(Ap[i] * qp[j] * c for i, j, c in e[1].terms) % p
            if d == 0:
                raise BadPoint("pole in S̄ data")
            return sum(Ap[i] * qp[j] * c for i, j, c in e[0].terms) * pow(d, -1, p) % p
        S = [[ev(e) for e in row] for row in self.S]
        T2 = [pow(A, a, p) * pow(q, b, p) % p for a, b in self.T2]
        return S, T2


@lru_cache(maxsize=None)
def load_exclusive(R, root=DEFAULT_DIR):
    return Exclusive(tuple(R), root)


@lru_cache(maxsize=None)
def load_sbar(R, root=DEFAULT_DIR):
    return SbarData(tuple(R), root)


def available_sbar(root=DEFAULT_DIR):
    if not os.path.isdir(root):
        return []
    return [tuple(int(c) for c in f[len("sbar_"):-len(".json.gz")])
            for f in sorted(os.listdir(root))
            if f.startswith("sbar_") and f.endswith(".json.gz")]


def _inv_mod(M, p):
    n = len(M)
    X = [list(r) + [int(i == j) for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        piv = next((r for r in range(c, n) if X[r][c] % p), None)
        if piv is None:
            raise BadPoint("singular S̄")
        X[c], X[piv] = X[piv], X[c]
        iv = pow(X[c][c], -1, p)
        X[c] = [x * iv % p for x in X[c]]
        for r in range(n):
            if r != c and X[r][c]:
                f = X[r][c]
                X[r] = [(x - f * y) % p for x, y in zip(X[r], X[c])]
    return [r[n:] for r in X]


NP_BOUND = 2 ** 31


def _inv_mod_np(M, p):
    """Modular Gauss--Jordan inverse with numpy int64 (p < 2^31)."""
    import numpy as np
    n = M.shape[0]
    X = np.concatenate([M % p, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        nz = np.nonzero(X[c:, c])[0]
        if len(nz) == 0:
            raise BadPoint("singular S̄")
        r = c + nz[0]
        if r != c:
            X[[c, r]] = X[[r, c]]
        X[c] = X[c] * pow(int(X[c, c]), -1, p) % p
        f = X[:, c].copy()
        f[c] = 0
        X = (X - f[:, None] * X[c][None, :]) % p
    return X[:, n:]


def _vecmat_np(v, M, p):
    return (v[:, None] * M % p).sum(axis=0) % p


def _chain_np(S, Si, T2, cf, v0, p, alternating):
    import numpy as np
    v = (Si if alternating else S)[v0].copy()
    for idx, a in enumerate(cf):
        e = a // 2
        t = np.array([pow(int(x), e, p) if e >= 0 else pow(pow(int(x), -1, p), -e, p) for x in T2],
                     dtype=np.int64)
        v = v * t % p
        M = (Si, S)[(idx + 1) % 2] if alternating else S
        v = _vecmat_np(v, M, p)
    den = (Si if alternating else S)[v0, v0]
    return int(v[v0]) * pow(int(den), -1, p) % p


def value_sbar_alternating(R, cf, F, A, q, root=DEFAULT_DIR):
    """Family G (rational 'vacuum-dual' gauge, S̄² != 1):
    <0| S̄⁻¹ T̄^{a1} S̄ T̄^{a2} S̄⁻¹ ... |0> / <0|S̄⁻¹|0>, framing-free T̄,
    evaluated at the NATURAL point (pinned against the U_Q 3-strand data)."""
    p = F.p
    d = load_sbar(tuple(R), root)
    if p < NP_BOUND:
        S, T2 = d.evaluate_np(p, int(A), int(q))
        return F(_chain_np(S, _inv_mod_np(S, p), T2, cf, d.vac, p, True))
    S, T2 = d.evaluate_modp(p, int(A), int(q))
    Si = _inv_mod(S, p)
    n, v0 = len(S), d.vac
    mats = (Si, S)
    v = list(Si[v0])
    for idx, a in enumerate(cf):
        e = a // 2
        t = [pow(x, e, p) if e >= 0 else pow(pow(x, -1, p), -e, p) for x in T2]
        v = [v[j] * t[j] % p for j in range(n)]
        M = mats[(idx + 1) % 2]
        v = [sum(v[k] * M[k][j] for k in range(n)) % p for j in range(n)]
    return F(v[v0]) / F(Si[v0][v0])


def value_sbar(R, cf, F, A, q, root=DEFAULT_DIR):
    """[S̄ T̄^{a1} S̄ T̄^{a2} ... T̄^{an} S̄]_{00} / S̄_{00}   (S̄_{00} = 1/dim_q R),
    framing-free T̄ (t0 = 1).  With the same even cf as family P this is directly
    the STANDARD H_R(A, q) (pinned against family P on R = [6] in the tests)."""
    p = F.p
    d = load_sbar(tuple(R), root)
    if p < NP_BOUND:
        S, T2 = d.evaluate_np(p, int(A), int(q))
        # S̄² = 1 here: <0| S̄ D S̄ ... S̄ |0> / S̄_00 (row form of the same chain)
        return F(_chain_np(S, None, T2, cf, d.vac, p, False))
    S, T2 = d.evaluate_modp(p, int(A), int(q))
    n = len(S)
    v = [S[i][d.vac] for i in range(n)]              # S̄ |0>
    for a in reversed(cf):
        e = a // 2
        t = [pow(x, e, p) if e >= 0 else pow(pow(x, -1, p), -e, p) for x in T2]
        v = [t[i] * v[i] % p for i in range(n)]
        v = [sum(S[i][j] * v[j] for j in range(n)) % p for i in range(n)]
    val = v[d.vac] * pow(S[d.vac][d.vac], -1, p) % p
    return F(val)


def available(root=DEFAULT_DIR):
    if not os.path.isdir(root):
        return []
    return [tuple(int(c) for c in f[len("exclusive_"):-len(".json.gz")])
            for f in sorted(os.listdir(root))
            if f.startswith("exclusive_") and f.endswith(".json.gz")]


def value(R, cf, F, A, q, root=DEFAULT_DIR):
    """Family-P value (their convention) at (A, q)."""
    ex = load_exclusive(tuple(R), root)
    if getattr(F, "p", None) is not None:
        return _value_modp(ex, R, cf, F, A, q)
    C, D2 = ex.evaluate(F, A, q)
    n = len(C)
    M = None
    for a in cf:
        e = a // 2 - 1
        scaled = [[C[i][j] * D2[j] ** e for j in range(n)] for i in range(n)]
        M = scaled if M is None else matmul(M, scaled)
    M = C if M is None else matmul(M, C)
    return qdim(R, A, q) * M[ex.vac][ex.vac]


def _value_modp(ex, R, cf, F, A, q):
    p = F.p
    C, D2 = ex.evaluate_modp(p, int(A), int(q))
    n = len(C)
    M = None
    for a in cf:
        e = a // 2 - 1
        d = [pow(x, e, p) if e >= 0 else pow(pow(x, -1, p), -e, p) for x in D2]
        scaled = [[C[i][j] * d[j] % p for j in range(n)] for i in range(n)]
        M = scaled if M is None else matmul_mod(M, scaled, p)
    M = C if M is None else matmul_mod(M, C, p)
    return qdim(R, A, q) * M[ex.vac][ex.vac]


@lru_cache(maxsize=None)
def chirality(p, q, reference=None):
    """True if the knot [p, q] needs the mirrored cf to match ``reference``
    (a Laurent polynomial, the fundamental HOMFLY in STANDARD convention)."""
    tb = TwoBridge(p, q)
    F = GF(2 ** 61 - 1)
    rng = random.Random(p * 1000003 + q)
    A, qq = F.random_element(rng), F.random_element(rng)
    ref = reference.evaluate([A, qq], one=F.one)
    Ap, qp = _point(A, qq)
    for mirror in (False, True):
        if value((1,), tb.even_cf(mirror), F, Ap, qp) == ref:
            return mirror
    raise ValueError("two-bridge [%d,%d]: neither chirality matches the reference" % (p, q))


def value_std(R, cf, F, A, q, root=DEFAULT_DIR):
    """STANDARD-convention H_R of the 4-plat with all-even cf (family-P
    orientation convention), from whichever exclusive data exist for R."""
    R = P(R)
    if R in available(root):
        Ap, qp = _point(A, q)
        return value(R, cf, F, Ap, qp, root)
    if load_sbar(R, root).family == "G":
        Ap, qp = _point(A, q)
        return value_sbar_alternating(R, cf, F, Ap, qp, root)
    return value_sbar(R, cf, F, A, q, root)


class TwoBridgeMethod(Method):
    name = "two-bridge"
    prime_bound = NP_BOUND        # numpy fast path for the S̄ families

    def __init__(self, root=DEFAULT_DIR):
        self.root = root

    def supports(self, knot, R):
        return (isinstance(knot, (TwoBridge, DoubleBraid)) and knot.is_knot()
                and (P(R) in available(self.root) or P(R) in available_sbar(self.root)))

    def evaluate(self, knot, R, F, A, q):
        return value_std(P(R), knot.even_cf(), F, A, q, self.root)
