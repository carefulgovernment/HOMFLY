"""Eigenvalue hypothesis (Itoyama--Mironov--Morozov--Morozov, "Eigenvalue
hypothesis for Racah matrices and HOMFLY polynomials for 3-strand knots in any
symmetric and antisymmetric representations", 2012):

a Racah matrix of a multiplicity-free block is determined (up to signs) by the
normalised eigenvalues of the adjacent R-matrix.  Explicit formulas exist for
sizes 2..5 (and conjecturally beyond); for size >= 6 degenerate eigenvalues
break the hypothesis.

Implemented: size 2.   TODO: sizes 3, 4, 5 (closed formulas from the
literature, or by solving the Yang--Baxter equation for U numerically in
GF(p) at each point -- the latter is size-agnostic and should be the default).
"""
from __future__ import annotations


def racah_2x2(l1, l2, F):
    """U = [[c, s], [s, -c]] with c = 1/[2]_t, s = sqrt([3]_t)/[2]_t,
    t^2 = -l1/l2, so that D = diag(l1, l2) and U D U satisfy the braid
    relation.  Written without t: c = sqrt(-l1 l2)/(l1 - l2)."""
    c = F.sqrt(-l1 * l2) / (l1 - l2)
    s = F.sqrt(1 - c * c)
    return [[c, s], [s, -c]]


def racah_from_eigenvalues(eigs, F):
    n = len(eigs)
    if n == 1:
        return [[F.one]]
    if n == 2:
        return racah_2x2(eigs[0], eigs[1], F)
    raise NotImplementedError("eigenvalue hypothesis for %dx%d blocks: see ROADMAP M4" % (n, n))
