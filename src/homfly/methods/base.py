"""Common interface of all computational methods.

A *method* is a black box: given a knot description, a representation R and a
point (A0, q0) in a field F it returns the value of the REDUCED colored HOMFLY
polynomial H_R(K; A0, q0) in the STANDARD convention (homfly.conventions).
The reconstruction layer turns black boxes into Laurent polynomials.

Knot descriptions understood by the methods:

    Braid                      (knots.braid.Braid)      -- braid-based methods
    TorusKnot(m, n)                                     -- Rosso--Jones
    TwoBridge(p, q) / continued fraction                -- arborescent / 2-bridge
    DoubleBraid(...)                                    -- double braid family
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..knots.families import TorusKnot, TwoBridge, DoubleBraid  # noqa: F401 (re-export)


class NotApplicable(Exception):
    """The method cannot handle this (knot, representation) pair."""


class Method(ABC):
    name = "abstract"
    #: exponent steps of the result (A^2, q^2 for knots)
    steps = (2, 2)

    @abstractmethod
    def supports(self, knot, R) -> bool:
        ...

    @abstractmethod
    def evaluate(self, knot, R, F, A, q):
        """Reduced H_R(knot) at (A, q) in F (STANDARD convention)."""

    def black_box(self, knot, R):
        if not self.supports(knot, R):
            raise NotApplicable("%s cannot compute %s in %s" % (self.name, knot, R))
        return lambda F, A, q: self.evaluate(knot, R, F, A, q)

    def cost_estimate(self, knot, R) -> float:
        """Rough relative cost used by the automatic method selector."""
        return 1.0
