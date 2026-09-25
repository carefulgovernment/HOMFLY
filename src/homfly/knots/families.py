"""Parametric knot families handled by specialised methods."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

from .braid import Braid, torus_braid


@dataclass(frozen=True)
class TorusKnot:
    """T[m, n]: closure of (sigma_1 ... sigma_{m-1})^n.  Knot iff gcd = 1."""
    m: int
    n: int

    def braid(self):
        return torus_braid(self.m, self.n)

    def is_knot(self):
        return gcd(self.m, self.n) == 1


@dataclass(frozen=True)
class TwoBridge:
    """Rational (2-bridge) knot b(p, q), 0 < q < p, p odd for knots.

    KnotInfo's ``two_bridge_notation`` is ``[p, q]``.  The continued fraction
    p/q = [a_1, ..., a_k] gives the twist regions of a 4-plat diagram, which
    is the input of the arborescent (fingers) evaluation.
    """
    p: int
    q: int
    mirror: bool = False

    def continued_fraction(self, even=False):
        return continued_fraction(Fraction(self.p, self.q), even=even)

    def is_knot(self):
        return self.p % 2 == 1

    def even_cf(self, mirror=None):
        """All-even negative continued fraction of the 4-plat used by the
        family-P exclusive data (racah_homfly ``schubert_to_even_cf``):
        p/q_even = a1 - 1/(a2 - 1/(...)), q_even = q or p - q (the even one),
        entries negated.  ``mirror=True`` gives the mirror knot."""
        p, q = self.p, self.q
        q_even = q if q % 2 == 0 else p - q
        cf = tuple(-a for a in even_minus_cf(Fraction(p, q_even)))
        m = self.mirror if mirror is None else mirror
        return tuple(-a for a in cf) if m else cf


def _nearest_even(x):
    k = x.numerator // (2 * x.denominator)
    cands = [2 * (k + j) for j in (-2, -1, 0, 1, 2)]
    return min(cands, key=lambda a: (abs(x - a), abs(a)))


def even_minus_cf(x: Fraction):
    """x = a1 - 1/(a2 - 1/(... - 1/an)) with all a_i nonzero even."""
    out = []
    for _ in range(256):
        if x.denominator == 1:
            a = x.numerator
            if a == 0 or a % 2:
                raise ValueError("no all-even expansion for %s" % x)
            out.append(a)
            return tuple(out)
        a = _nearest_even(x)
        if a == 0:
            raise ValueError("zero coefficient for %s" % x)
        out.append(a)
        x = 1 / (Fraction(a) - x)
    raise RuntimeError("even continued fraction did not terminate")


@dataclass(frozen=True)
class DoubleBraid:
    """Double braid knot: two twist regions with m and n (full) twists.

    ``antiparallel=(True, True)`` is the genus-1 family C(2m, 2n) of
    A. Morozov, "Factorization of differential expansion for antiparallel
    double-braid knots"; twist knots are n = +-1.  Parallel regions give
    the remaining double-braid families.  The rectangular-representation
    method (methods.double_braid) uses the evolution in m, n with exclusive
    Racah matrices S̄ for R ⊗ R̄.
    """
    m: int
    n: int
    antiparallel: tuple = (True, True)

    def two_bridge(self):
        if self.antiparallel == (True, True):
            f = Fraction(2 * self.m) + Fraction(1, 2 * self.n)
            return TwoBridge(abs(f.numerator), abs(f.denominator) % abs(f.numerator))
        raise NotImplementedError


def continued_fraction(x: Fraction, even=False):
    """Regular continued fraction; with ``even=True`` the (unique when it
    exists) expansion with all even partial quotients (+-), which for 2-bridge
    knots gives a diagram with all twist regions antiparallel."""
    if not even:
        out = []
        while True:
            a = x.numerator // x.denominator
            out.append(a)
            x -= a
            if x == 0:
                return out
            x = 1 / x
    out = []
    while True:
        a = x.numerator // x.denominator
        if a % 2:
            a += 1
        out.append(a)
        x = x - a
        if x == 0:
            return out
        x = 1 / x
        if len(out) > 64:
            raise ValueError("no even continued fraction")
