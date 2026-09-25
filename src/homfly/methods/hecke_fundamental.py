"""Fundamental HOMFLY of an arbitrary braid closure via the Ocneanu trace
realised in Young's seminormal representations of the Hecke algebra:

    H^{unred}_nat = A^{-w} sum_{Q |- m} dim_q(Q) Tr_{S^Q}(rho(beta)).

Always applicable (knots and links, any number of strands); cost grows like
sum_Q (f^Q)^2 * len(beta).  Main validation reference for all other methods.
"""
from __future__ import annotations

from ..conventions import natural_point
from ..knots.braid import Braid
from ..hecke.seminormal import SeminormalModule
from ..reps.partitions import partitions
from ..reps.qdim import qdim
from .base import Method


def natural_unreduced(braid, A, q):
    tot = None
    for Q in partitions(braid.strands):
        M = SeminormalModule(Q, q)
        t = qdim(Q, A, q) * M.trace_word(braid.word)
        tot = t if tot is None else tot + t
    return tot * A ** (-braid.writhe)


class HeckeFundamental(Method):
    name = "hecke-fundamental"

    def supports(self, knot, R):
        return isinstance(knot, Braid) and tuple(R) == (1,)

    def evaluate(self, knot, R, F, A, q):
        A, q = natural_point(A, q)
        return natural_unreduced(knot, A, q) / qdim((1,), A, q)
