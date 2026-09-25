"""Composite representations (R, P) of GL(N) -- the irreps in R ⊗ P̄ -- needed
for exclusive Racah matrices S̄ and for antiparallel twist regions
(2-bridge, double-braid, arborescent methods).

Planned (ROADMAP M3):
  * decomposition R ⊗ R̄ = sum_{Y} (Y, Y') via Koike's formula
    [R ⊗ P̄] = sum_{X,Y,Z} c^R_{XZ} c^P_{YZ} (X, Y);
  * quantum dimension of (X, Y): dim_q (X,Y) via the Koike determinant or
    via  dim_q = s_{(X,Y)}(q^rho) with A = q^N kept symbolic;
  * eigenvalue of the antiparallel R-matrix on (Y, Y'):
    lambda_bar = eps q^{kappa_Y - kappa_{Y'}} A^{|Y|}  (convention to be pinned
    against the fundamental case, where R ⊗ R̄ = (∅,∅) + ([1],[1])).
"""
from __future__ import annotations

from .characters import lr_product  # noqa: F401


def composite_decomposition(R, P):
    raise NotImplementedError("ROADMAP M3")


def composite_qdim(X, Y, A, q):
    raise NotImplementedError("ROADMAP M3")
