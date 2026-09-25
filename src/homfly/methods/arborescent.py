"""Arborescent calculus (Mironov--Morozov--Morozov--Sleptsov, "Tabulating
knot polynomials for arborescent knots"; "Colored HOMFLY polynomials for the
pretzel knots and links").

A 2-bridge (rational) knot with continued fraction [a_1, ..., a_k] -- and more
generally any arborescent knot -- is glued from *fingers* (twist regions) and
*propagators*.  In representation R:

    finger (parallel twist region, a crossings):   F(a)  = S T^a S^†
    finger (antiparallel region):                  F̄(a) = S̄ T̄^a S̄^†
    H_R ∝ < fingers glued along the tree >_{0}

with T, T̄ the diagonal eigenvalue matrices on R⊗R and R⊗R̄ and S, S̄ the
exclusive Racah matrices.  2-bridge knots are chains (a path-shaped tree), so
H_R is a single matrix product.

Planned (ROADMAP M3):
  * TwoBridge(p, q) -> chain of fingers with orientation bookkeeping (which
    twist regions are parallel/antiparallel follows from the parity pattern
    of the continued fraction / the orientation of the 4-plat);
  * evaluation in GF(p) with S, S̄ from RacahStore.exclusive(R);
  * validation: fundamental rep against Hecke; [2], [1,1], [2,1] against
    cabling; tables of 2-bridge knots up to 12 crossings (362 knots, see
    knots.table.knots(two_bridge=True)).
  * later: pretzel, Montesinos (KnotInfo montesinos_notation), and general
    arborescent trees (Conway notation parser).
"""
from __future__ import annotations

from ..knots.families import TwoBridge
from .base import Method


class Arborescent(Method):
    name = "arborescent"

    def __init__(self, store=None):
        self.store = store

    def supports(self, knot, R):
        return isinstance(knot, TwoBridge) and self.store is not None

    def evaluate(self, knot, R, F, A, q):
        raise NotImplementedError("ROADMAP M3: arborescent evaluation")
