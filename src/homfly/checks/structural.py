"""Structural properties every colored HOMFLY must satisfy -- used as cheap
consistency checks on computed tables (independent of the method used).

* special polynomial:      H_R(A, q=1) = H_[1](A, q=1)^{|R|}
* sl_N vanishing/triviality: H_R(A = q^N, q) = 1 when R has more than N rows
  ... more precisely the invariant of a rep of sl_N; for l(R) = N+1 it is the
  trivial invariant only after normalisation, so we test only the clean case
  H_[1^k](A = q^{k-1}... ) -- see tests.
* transposition:           H_{R^T}(A, q) = H_R(A, -1/q)
* mirror:                  H_R(K*; A, q) = H_R(K; 1/A, 1/q)
* differential expansion (TODO, ROADMAP M7): for symmetric R = [r],
  H_[r] - 1 is divisible by {Aq^{r}}{A/q}; for rectangular R the
  Z-factors of Kononov--Morozov.
"""
from __future__ import annotations

from ..algebra.laurent import Laurent


def specialise_q1(H):
    t = {}
    for (a, b), c in H.terms.items():
        t[(a, 0)] = t.get((a, 0), 0) + c
    return Laurent(t, H.vars)


def special_polynomial_ok(H_R, H_1, size):
    return specialise_q1(H_R) == specialise_q1(H_1) ** size


def differential_expansion_symmetric_ok(H, r, F=None):
    """H_[r](A,q) - 1 vanishes at A = q^{-r} and at A = q (zeros of {Aq^r}{A/q})."""
    from ..algebra.fields import GF
    import random
    F = F or GF(1000000007)
    rng = random.Random(0)
    for _ in range(3):
        q = F.random_element(rng)
        for A in (q ** (-r), q):
            if H.evaluate([A, q], one=F.one) != 1:
                return False
    return True
