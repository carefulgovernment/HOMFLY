"""Strengthened Formula II: closed point evaluation of antiparallel double braids, twist knots and Borromean rings.

Formula II writes every invariant through one symmetric cubic tensor T on the labels X (channels of R x R-bar):

    c(R) = T(., ., chi(R)),   H_R(m, n) = T(Lambda^m, Lambda^n, chi(R)),   b(R1, R2, R3) = T(chi(R1), chi(R2), chi(R3)).

The naive tensor (one rank-one term g_c e_c^3 per channel, methods/interpolation.py) is exact for |R| <= 7.  The
strengthened engine adds the zero and consistency rules for the rows, channel splitting, the kernel cubes of
[4,3,1], [5,3,1], [5,4,1], [6,3,1], [7,3,1], [5,3,2], [5,3,1,1] (and transposes), the deferred terms of [4,3,2], [4,3,1,1],
[4,3,1,1,1], [5,3,2], [5,3,1,1] (and transposes), the support-principle rows of [4,3,2,1] and the closed 15 x 15 block of
[4,3,2,1] (Gram matrix data/block4321_closed.json).  Everything is closed: no Racah data, no calibration.

Range: every R with |R| <= 10, for double braids and for Borromean rings with any three such colours (checked against the stored polynomials: twist knots, double braids and Borromean rings
for |R| <= 9, twist knots for all 42 diagrams with |R| = 10).  The 11-box diagrams over [4,3,2,1] need a calibration
from ring legs and are not available here.

The engine is mod-p integer arithmetic with numpy int64, so the prime must be below 2^31 (PRIME_BOUND).

    >>> from homfly import formula2
    >>> formula2.double_braid((4, 3, 1), 2, -1, A, q, p)        # H_[4,3,1](DoubleBraid(2, -1)) mod p
    >>> formula2.borromean((2, 1), (2, 1), (2, 1), A, q, p)     # reduced H of the Borromean rings, colours R1, R2, R3
"""
from __future__ import annotations

from contextlib import contextmanager

from ..algebra.fields import GF, BadPoint
from ..methods import interpolation as _I
from . import _linalg, _core, _strong, _block4321

PRIME_BOUND = 2 ** 31
MAX_BOXES = 10
U4321 = (4, 3, 2, 1)

__all__ = ["Formula2Point", "double_braid", "twist_knot", "borromean", "supported", "PRIME_BOUND", "MAX_BOXES"]

_CLOSED_BLOCK = None


def _closed_block():
    global _CLOSED_BLOCK
    if _CLOSED_BLOCK is None:
        _CLOSED_BLOCK = _block4321.ClosedBlock()
    return _CLOSED_BLOCK


def _set_prime(p):
    """rebind the working prime of the engine modules (they keep it as a module global)"""
    if _core.p == p: return
    F = GF(p)
    for mod in (_linalg, _core, _strong, _block4321):
        mod.p = p
    _core.F = F; _strong.F = F


@contextmanager
def _engine(p, nodes=None):
    """prime p for the engine.  The engine evaluates the rows with a fixed node bound (Core sets methods.interpolation.nodes);
    that setting is installed while the engine runs (nodes) and the original function restored afterwards."""
    if not (2 < p < PRIME_BOUND):
        raise ValueError("formula2 works modulo primes below 2^31 (got %d)" % p)
    orig = _I.nodes
    _set_prime(p)
    if nodes is not None: _I.nodes = nodes
    try:
        yield
    except (ZeroDivisionError, ValueError) as e:            # pow(0, -1, p), q^{2k} = 1, vanishing dimension, ...
        if isinstance(e, ValueError) and "invertible" not in str(e) and "inverse" not in str(e):
            raise
        raise BadPoint(str(e))
    finally:
        _I.nodes = orig


def _norm(R):
    return tuple(int(r) for r in R if r)


def supported(R):
    """True when the closed engine covers the representation R"""
    return 0 < sum(_norm(R)) <= MAX_BOXES


class Formula2Point:
    """The strengthened Formula II at one point (A, q) modulo a prime p < 2^31 (all values are ints mod p).
    The rows, kernels and blocks are cached per instance, so evaluate many knots / colours at one point with one instance."""

    def __init__(self, A, q, p):
        self.p = p
        with _engine(p):
            self.st = _strong.Strong(A, q)
            self._nodes = _I.nodes                         # node bound fixed by the engine for this point
        self.A, self.q = self.st.A, self.st.q

    def _prepare(self, R):
        R = _norm(R)
        if not supported(R):
            raise NotImplementedError("formula2 covers 1 <= |R| <= %d (got %s)" % (MAX_BOXES, list(R)))
        if _core.contains(R, U4321) and U4321 not in self.st._open:
            V = _block4321.block_basis(self.st)
            self.st._open[U4321] = (V, _block4321.closed_M(_closed_block(), self.st, V))
        return R

    def double_braids(self, R, mn):
        """{(m, n): H_R(DoubleBraid(m, n))} for the pairs in mn (reduced, STANDARD convention)"""
        with _engine(self.p, self._nodes):
            R = self._prepare(R)
            return self.st.double_braid(R, [tuple(x) for x in mn])

    def double_braid(self, R, m, n):
        return self.double_braids(R, [(m, n)])[(m, n)]

    def twist_knot(self, R, m):
        """H_R(DoubleBraid(m, 1)): m = 1 trefoil 3_1, m = -1 figure-eight 4_1, m = 2 5_2, m = -2 6_1, ..."""
        return self.double_braid(R, m, 1)

    def borromean_b(self, R1, R2, R3):
        """b = T(chi(R1), chi(R2), chi(R3)): the Borromean invariant normalized by all three quantum dimensions,
        b = H(R1, R2, R3) / (d_R2 d_R3)  with H reduced by d_R1.  T is symmetric; it is contracted as
        1 + chi(R2)^T c(R1) chi(R3) with R1 the smallest colour."""
        cols = sorted((_norm(R) for R in (R1, R2, R3)), key=lambda R: (sum(R), R))
        return self.contract(*cols)

    def contract(self, R1, R2, R3):
        """1 + chi(R2)^T c(R1) chi(R3) (any order of the colours gives the same value)"""
        with _engine(self.p, self._nodes):
            R1 = self._prepare(R1); R2 = self._prepare(R2); R3 = self._prepare(R3)
            st = self.st; core = st.core; p = self.p
            tl, rd = st.terms(R1)
            b = 1
            for W, e in tl:
                b = (b + W * core.Phi(e, R2) % p * core.Phi(e, R3)) % p
            for coef, u, v in st.deferred_terms(R1):
                b = (b + coef * st.hval(u, R2) % p * st.hval(v, R3)) % p
            for B, M in st.blocks(R1):            # [4,3,2,1] block: M = T_U(., ., f(U)) at R1 = U
                f2 = [st.hval(x, R2) for x in B]; f3 = [st.hval(x, R3) for x in B]
                b = (b + sum(M[a][c] * f2[a] % p * f3[c] for a in range(len(B)) for c in range(len(B)))) % p
            return b

    def qdim(self, R):
        """quantum dimension d_R(A, q) (STANDARD convention: unknot = d_R before reduction)"""
        p = self.p; A, q = self.A, self.q
        d = 1
        R = _norm(R); Rc = [sum(1 for x in R if x > j) for j in range(R[0])]
        for i, r in enumerate(R):
            for j in range(r):
                h = R[i] - j + Rc[j] - i - 1
                num = (A * pow(q, (j - i) % (p - 1), p) - pow(A * pow(q, (j - i) % (p - 1), p) % p, p - 2, p)) % p
                den = (pow(q, h, p) - pow(q, (p - 1 - h) % (p - 1), p)) % p
                if den == 0: raise BadPoint("q^{2h} = 1")
                d = d * num % p * pow(den, p - 2, p) % p
        return d

    def borromean(self, R1, R2, R3):
        """reduced colored HOMFLY of the Borromean rings (unnormalized / d_R1): d_R2 d_R3 b(R1, R2, R3)"""
        b = self.borromean_b(R1, R2, R3)
        return b * self.qdim(R2) % self.p * self.qdim(R3) % self.p


_LAST = {}


def _point(A, q, p):
    key = (A % p, q % p, p)
    pt = _LAST.get("pt")
    if pt is None or (pt.A, pt.q, pt.p) != key:
        pt = Formula2Point(A, q, p); _LAST["pt"] = pt
    return pt


def double_braid(R, m, n, A, q, p):
    """H_R(DoubleBraid(m, n)) at (A, q) mod p"""
    return _point(A, q, p).double_braid(R, m, n)


def twist_knot(R, m, A, q, p):
    return _point(A, q, p).twist_knot(R, m)


def borromean(R1, R2, R3, A, q, p):
    """reduced colored HOMFLY of the Borromean rings with colours R1, R2, R3 at (A, q) mod p"""
    return _point(A, q, p).borromean(R1, R2, R3)
