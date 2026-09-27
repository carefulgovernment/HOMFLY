"""Colored HOMFLY of any braid closure by cabling in the multiplicity spaces.

Wraps the engine of the "colored-homfly-cabling" skill
(``_cabling_paths_engine``): the r-cable of the braid is traced only over the
multiplicity spaces W_Q of Q in R^{(x)m}, in a path basis
0 < R < ... < Q; projectors and band crossings live in small skew-shape
Hecke representations, so the big irreps of H_{mr} are never built.  The
A-dependence is exact at every point q0 (S*_Q are polynomials in A^2), the
q-dependence is Newton-interpolated per prime, a second prime verifies (CRT if
needed).  No Racah matrices are used, so it is the method of last resort for
knots without two-bridge / Montesinos / 3-strand descriptions (e.g. the
polyhedral knots), limited by the size of the 2|R|-box band spaces.

Conventions: the engine returns the NATURAL-convention polynomial (Knot Atlas,
a = A); STANDARD is H_std(A, q) = H_nat(1/A, 1/q) (checked against the stored
two-bridge / Montesinos / 3-strand results).
"""
from __future__ import annotations

from functools import lru_cache

from ..algebra.laurent import Laurent
from ..knots.braid import Braid
from ..reps.partitions import P
from .base import Method


def _engine():
    from . import _cabling_paths_engine as E
    return E


@lru_cache(maxsize=None)
def _count_skew_syt(lam, mu):
    """number of standard fillings of mu/lam (counted, not listed)"""
    E = _engine()
    if lam == mu:
        return 1
    tot = 0
    for i in range(len(mu)):
        # remove a removable corner of mu that is outside lam
        if mu[i] > (lam[i] if i < len(lam) else 0) and (i + 1 >= len(mu) or mu[i + 1] < mu[i]):
            nu = list(mu)
            nu[i] -= 1
            while nu and nu[-1] == 0:
                nu.pop()
            if E.contains(tuple(nu), lam):
                tot += _count_skew_syt(lam, tuple(nu))
    return tot


@lru_cache(maxsize=None)
def band_space_size(m, R, letters):
    """total number of 2|R|-box skew tableaux the engine will build (memory estimate)"""
    E = _engine()
    r = sum(R)
    levels, c = [[()]], {}
    for _ in range(m):
        nxt = set()
        for lam in levels[-1]:
            for mu in E.shapes_adding(lam, r):
                cc = E.lr_coeff(lam, R, mu)
                if cc:
                    c[(lam, mu)] = cc
                    nxt.add(mu)
        levels.append(sorted(nxt))
    tot = 0
    for j in letters:
        pairs = {(lam, nu) for lam in levels[j - 1] for nu in levels[j + 1]
                 if any((lam, ka) in c and (ka, nu) in c for ka in levels[j])}
        tot += sum(_count_skew_syt(lam, nu) for lam, nu in pairs)
    return tot


class CablingPaths(Method):
    name = "cabling-paths"
    direct = True                  # returns the polynomial itself (own reconstruction)

    def __init__(self, max_tableaux=4_000_000):
        self.max_tableaux = max_tableaux

    def supports(self, knot, R):
        if not isinstance(knot, Braid) or knot.strands < 2 or not knot.word:
            return False
        if knot.components() != 1:
            return False
        R = P(R)
        letters = tuple(sorted({abs(g) for g in knot.word}))
        return band_space_size(knot.strands, R, letters) <= self.max_tableaux

    def polynomial(self, knot, R, report=None):
        E = _engine()
        H = E.ColoredHOMFLY(list(knot.word), knot.strands, P(R))
        poly = H.compute()
        if report is not None:
            report.evaluations = H.nevals
            report.verified = True          # checked at a further prime inside compute()
        return Laurent({(-a, -b): c for (a, b), c in poly.items()})

    def evaluate(self, knot, R, F, A, q):
        """black-box form: engine at q' = 1/q, A-polynomial evaluated at 1/A"""
        E = _engine()
        eng = _cached_engine(tuple(knot.word), knot.strands, P(R))
        p = F.p
        v = eng.evaluate(int(1 / q), p)
        Ai = 1 / A
        return sum((F(int(c)) * Ai ** a for a, c in zip(eng.a_exponents(), v)), F.zero)


@lru_cache(maxsize=8)
def _cached_engine(word, strands, R):
    return _engine().CablingEngine(list(word), strands, R)
