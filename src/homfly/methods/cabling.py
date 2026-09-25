"""Colored HOMFLY by cabling.

H_R(K) is the fundamental HOMFLY of the r-parallel cable (r = |R|) of a braid
for K with a minimal Hecke idempotent E_R inserted on one bundle:

    H^{unred}_{R, nat} = theta_R^{-w} sum_{Q |- m r} dim_q(Q) Tr_{S^Q}(E_R rho(cable(beta))).

In the seminormal basis E_R (for the standard tableau t of shape R on letters
1..r) is the diagonal projector onto tableaux whose restriction to 1..r is t,
so no idempotent has to be constructed explicitly.

Independent of Racah matrices: the reference for validating colored methods.
Cost explodes with m*r (dimension of S^Q), so in practice: m*r <= ~12.
"""
from __future__ import annotations

from ..conventions import natural_point
from ..knots.braid import Braid
from ..hecke.seminormal import SeminormalModule
from ..reps.partitions import P, partitions, standard_tableaux
from ..reps.qdim import qdim, theta
from .base import Method


def natural_unreduced_colored(braid, R, A, q):
    R = P(R)
    r = sum(R)
    t = standard_tableaux(R)[0]
    cab = braid.cable(r)
    tot = None
    for Q in partitions(cab.strands):
        M = SeminormalModule(Q, q)
        support = [i for i, T in enumerate(M.basis) if T[:r] == t]
        if not support:
            continue
        v = qdim(Q, A, q) * M.trace_word(cab.word, support)
        tot = v if tot is None else tot + v
    return tot * theta(R, A, q) ** (-braid.writhe)


class Cabling(Method):
    name = "cabling"

    def __init__(self, max_strands=12):
        self.max_strands = max_strands

    def supports(self, knot, R):
        return isinstance(knot, Braid) and knot.is_knot() and knot.strands * sum(R) <= self.max_strands

    def evaluate(self, knot, R, F, A, q):
        A, q = natural_point(A, q)
        return natural_unreduced_colored(knot, R, A, q) / qdim(R, A, q)

    def cost_estimate(self, knot, R):
        from math import factorial
        return factorial(knot.strands * sum(R)) ** 0.5 * len(knot) * sum(R) ** 2
