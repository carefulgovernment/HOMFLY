"""Rosso--Jones formula for torus knots T[m, n] (gcd(m, n) = 1), any R:

    H^{unred}_{R, nat}(T[m,n]) = theta_R^{-mn} sum_{Q |- m|R|} c^Q_{R,m} theta_Q^{n/m} dim_q(Q),

where psi_m(s_R) = s_R[p_m] = sum_Q c^Q_{R,m} s_Q is the Adams operation
(reps.characters.adams).  theta_Q^{n/m} = A^{n|R|} q^{2 n kappa_Q / m}.

References: M. Rosso, V. Jones, "On the invariants of torus knots derived from
quantum groups" (1993); X.-S. Lin, H. Zheng, "On the Hecke algebras and the
colored HOMFLY polynomial" (2010); Mironov--Morozov--Morozov, "Character
expansion for HOMFLY polynomials" I-III.

Extensions (TODO): torus links (products of Adams operations over components),
cabled torus knots / iterated torus knots, closed-form symbolic output.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..conventions import natural_point
from ..reps.characters import adams
from ..reps.partitions import P, kappa
from ..reps.qdim import qdim
from .base import Method, TorusKnot


def rosso_jones_terms(R, m, n):
    """[(c_Q, A-exponent, q-exponent, Q)] of the natural *unreduced* sum."""
    R = P(R)
    r = sum(R)
    out = []
    for Q, c in adams(R, m).items():
        eq = Fraction(2 * n * kappa(Q), m) - 2 * m * n * kappa(R)
        if eq.denominator != 1:
            raise ArithmeticError("non-integral q exponent in Rosso-Jones (check convention)")
        out.append((c, n * r - m * n * r, int(eq), Q))
    return out


def natural_unreduced(R, m, n, A, q):
    tot = None
    for c, ea, eq, Q in rosso_jones_terms(R, m, n):
        v = c * A ** ea * q ** eq * qdim(Q, A, q)
        tot = v if tot is None else tot + v
    return tot


class RossoJones(Method):
    name = "rosso-jones"

    def supports(self, knot, R):
        return isinstance(knot, TorusKnot) and gcd(knot.m, knot.n) == 1 and knot.m > 0

    def evaluate(self, knot, R, F, A, q):
        A, q = natural_point(A, q)
        return natural_unreduced(R, knot.m, knot.n, A, q) / qdim(R, A, q)
