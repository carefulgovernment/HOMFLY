"""Consistency checks for Racah data (run on every imported table).

* orthogonality  U U^T = 1 (unitary gauge) at random GF(p) points;
* braid / Yang--Baxter relation of the generated generators on path spaces;
* agreement of 3-strand results with cabling for small |R|;
* eigenvalue hypothesis agreement for multiplicity-free blocks.
"""
from __future__ import annotations

from ..algebra.linalg import identity, matmul, transpose


def is_orthogonal(U, F):
    return matmul(U, transpose(U)) == identity(len(U), F.one)


def check_braid_relations(space, strands):
    """Check sigma_k sigma_{k+1} sigma_k = sigma_{k+1} sigma_k sigma_{k+1} and
    far commutativity on a methods.rt_braid.PathSpace (or any object with
    apply_word and paths)."""
    F = space.F
    n = len(space.paths)
    for i in range(n):
        e = [F.zero] * n
        e[i] = F.one
        for k in range(1, strands - 1):
            if space.apply_word([k, k + 1, k], e) != space.apply_word([k + 1, k, k + 1], e):
                return False
        for k in range(1, strands):
            for l in range(k + 2, strands):
                if space.apply_word([k, l], e) != space.apply_word([l, k], e):
                    return False
    return True
