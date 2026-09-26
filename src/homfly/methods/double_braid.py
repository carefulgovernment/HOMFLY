"""Double braid knots in rectangular representations R = [r^s].

SUPERSEDED for antiparallel double braids by methods.interpolation
(DoubleBraidInterpolation), which covers every R.  Kept as the specification of
the rectangular closed form (factorised differential expansion) for later
symbolic-in-(m, n) output.

For rectangular R the representation R ⊗ R̄ is multiplicity free, its
components are labelled by sub-diagrams of R, the exclusive matrix S̄ is known
in closed form (conjecturally, via the "eigenvalue"/"hidden integrability"
structure), and the colored HOMFLY of the double-braid family has the evolution
form

    H_R(m, n) = sum_{X, Y ⊆ R}  C_{XY}(A, q)  Λ_X^{m} Λ_Y^{n},

with Λ_X = (normalised) eigenvalues of the double twist and C_{XY} built from
S̄.  The differential expansion factorises (A. Morozov, "Factorization of
differential expansion for antiparallel double-braid knots"; Ya. Kononov,
A. Morozov, "On rectangular HOMFLY for twist knots").

Planned (ROADMAP M6):
  1. closed-form S̄ for [r] (q-6j symbols of U_q(sl_2) with A-deformation)
     and for [r^s] (from the rectangular factorisation / from the store);
  2. evolution coefficients C_{XY} evaluated in GF(p);
  3. symbolic-in-(m, n) output: coefficients as Laurent polynomials, powers
     Λ^m kept symbolic, so a single computation gives the whole family;
  4. validation against cabling (small r, s) and against arborescent method.
"""
from __future__ import annotations

from ..knots.families import DoubleBraid
from ..reps.partitions import is_rectangular
from .base import Method


class DoubleBraidRectangular(Method):
    name = "double-braid"

    def supports(self, knot, R):
        return isinstance(knot, DoubleBraid) and is_rectangular(tuple(R))

    def evaluate(self, knot, R, F, A, q):
        raise NotImplementedError("ROADMAP M6: double-braid evolution formula")
