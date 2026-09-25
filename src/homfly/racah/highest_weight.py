"""Highest-weight method for Racah matrices (planned, ROADMAP M5).

Goal: compute inclusive U[Y, R, R -> Z] and exclusive S, S̄ matrices for
representations beyond the supplied tables (|R| > 6 boxes, 4+ strands with
large intermediate Y) by

1. building highest-weight vectors of Z inside (Y ⊗ R) ⊗ R and Y ⊗ (R ⊗ R)
   in U_q(sl_N) for a generic but *fixed small* N (N >= number of rows
   involved), using the coproduct and the kernel of the raising operators
   E_i -- linear algebra over Q(q), done modularly: GF(p) at many q-points;
2. U = Gram-overlap matrix of the two bases (no normalisation needed: only
   U and U^{-1} enter the trace, so the non-unitary rational form avoids all
   square roots);
3. A-dependence is restored by the standard argument that the Racah matrices
   depend on N only through A = q^N once they are "N-independent" (true for
   inclusive matrices of Young-diagram representations; exclusive ones need
   several N and rational reconstruction in A).

References: Mironov--Morozov--Morozov--Sleptsov, "Colored knot polynomials:
HOMFLY in representation [2,1]" and follow-ups; Bishler--Morozov--Sleptsov--
Shakirov on [3,1]; Shakirov--Sleptsov "Quantum Racah matrices and 3-strand
braids in irreps R with |R| = 4".
"""
from __future__ import annotations

from .provider import RacahProvider


class HighestWeightRacah(RacahProvider):
    def __init__(self, N=None):
        self.N = N

    def decompose(self, Y, R):
        from ..reps.characters import lr_product
        out = []
        for Z, mult in sorted(lr_product(Y, R).items()):
            out.extend((Z, i) for i in range(mult))
        return out

    def eigenvalues(self, R, F, A, q):
        raise NotImplementedError("ROADMAP M5")

    def inclusive(self, Y, R, Z, F, A, q):
        raise NotImplementedError("ROADMAP M5")
