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

    def continued_fraction(self, even=False):
        return continued_fraction(Fraction(self.p, self.q), even=even)

    def is_knot(self):
        return self.p % 2 == 1


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
