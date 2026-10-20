"""Antiparallel double braids by the strengthened Formula II (homfly.formula2): exact for every R with |R| <= 10.

The naive interpolation formula (methods/interpolation.py, DoubleBraidInterpolation) is exact only up to 7 boxes;
from 8 boxes on, the rows, kernel cubes, deferred terms and the [4,3,2,1] block of the strengthened formula are needed.
"""
from __future__ import annotations

from .base import Method, DoubleBraid
from .. import formula2 as F2


class DoubleBraidStrong(Method):
    """DoubleBraid(m, n) in any R with |R| <= 10 (closed, no Racah data)."""

    name = "double-braid-strong"
    prime_bound = F2.PRIME_BOUND          # numpy int64 arithmetic

    def supports(self, knot, R):
        return isinstance(knot, DoubleBraid) and knot.antiparallel == (True, True) and F2.supported(R)

    def evaluate(self, knot, R, F, A, q):
        p = F.p
        return F(F2.double_braid(tuple(R), knot.m, knot.n, int(A.v), int(q.v), p))

    def cost_estimate(self, knot, R):
        return 2.0 ** max(0, sum(R) - 6)
