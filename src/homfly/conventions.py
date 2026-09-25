"""Normalisation conventions -- the single source of truth.

STANDARD convention (the output of every public function):

* variables A, q;  z = q - q^{-1};
* fundamental skein relation   A^{-1} H(L+) - A H(L-) = (q - q^{-1}) H(L0),
  i.e. KnotInfo's P(v, z) with v = A, z = q - 1/q;
* reduced:  H_R(unknot) = 1;  unreduced = reduced * dim_q R;
* topological framing;
* positive trefoil (closure of sigma_1^3):  H_[1] = A^2 q^2 + A^2 q^{-2} - A^4;
* H_R(A = q^N, q) is the U_q(sl_N) invariant with R a Young diagram
  ([r] = symmetric), e.g. H_[1,1](A = q^2) = 1.

NATURAL convention (internal to the Hecke / cabling / Rosso--Jones engines):
R-matrix eigenvalues q (symmetric) and -1/q (antisymmetric) on [1]⊗[1],
framing factor theta_R = A^{|R|} q^{2 kappa_R}, dim_q as in reps.qdim.
It is related to STANDARD by the mirror map (A, q) -> (1/A, 1/q):

    H_standard(K; A, q) = H_natural(K; 1/A, 1/q).

Engines built from external Racah matrices (Mironov--Morozov--Morozov and
collaborators) must declare their own convention and convert here; the tests
in tests/test_conventions.py pin everything against KnotInfo.
"""
from __future__ import annotations


def natural_point(A, q):
    """Point at which a NATURAL-convention engine must be evaluated to produce
    the STANDARD value at (A, q)."""
    return A ** -1, q ** -1


def mirror(H):
    """Mirror image of a Laurent polynomial in (A, q)."""
    from .algebra.laurent import Laurent
    return Laurent({(-a, -b): c for (a, b), c in H.terms.items()}, H.vars)


def transpose_rep(H):
    """H_{R^T}(A, q) = H_R(A, -1/q)."""
    from .algebra.laurent import Laurent
    return Laurent({(a, -b): c * (-1) ** (b % 2) for (a, b), c in H.terms.items()}, H.vars)
