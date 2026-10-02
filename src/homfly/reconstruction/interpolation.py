"""Polynomial interpolation over a field.

Implemented:
  * Newton interpolation (incremental, O(n^2)), with early termination
  * univariate Laurent reconstruction with automatic degree detection
  * dense bivariate reconstruction on a tensor grid with known exponent box

Planned (see docs/ROADMAP.md): Zippel / Ben-Or--Tiwari sparse interpolation,
Thiele rational-function interpolation (for quantities that are genuinely
rational in A, q such as unreduced polynomials or Racah matrix entries),
and fast (subproduct-tree) interpolation.
"""
from __future__ import annotations

import random

from ..algebra.fields import BadPoint


class NewtonInterpolator:
    """Incremental Newton interpolation p(x) through points (x_i, y_i).
    Over a prime field the arithmetic runs on plain ints mod p."""

    def __init__(self, F):
        self.F = F
        self.p = getattr(F, "p", None)
        self.xs = []
        self.c = []  # Newton coefficients

    def __len__(self):
        return len(self.xs)

    def _value_int(self, x):
        p, xs, c = self.p, self.xs, self.c
        r = c[-1]
        for k in range(len(c) - 2, -1, -1):
            r = (r * (x - xs[k]) + c[k]) % p
        return r

    def value(self, x):
        if not self.c:
            return self.F.zero
        if self.p is not None:
            return self.F(self._value_int(int(x)))
        r = self.c[-1]
        for k in range(len(self.c) - 2, -1, -1):
            r = r * (x - self.xs[k]) + self.c[k]
        return r

    def add(self, x, y):
        # divided difference update
        if self.p is not None:
            p = self.p
            x, y = int(x) % p, int(y) % p
            if not self.xs:
                self.xs.append(x)
                self.c.append(y)
                return
            num = (y - self._value_int(x)) % p
            den = 1
            for xi in self.xs:
                den = den * (x - xi) % p
            if den == 0:
                raise BadPoint("division by zero mod %d" % p)
            self.xs.append(x)
            self.c.append(num * pow(den, -1, p) % p)
            return
        n = len(self.xs)
        if n == 0:
            self.xs.append(x)
            self.c.append(y)
            return
        num = y - self.value(x)
        den = self.F.one
        for xi in self.xs:
            den = den * (x - xi)
        self.xs.append(x)
        self.c.append(num / den)

    def monomial_coefficients(self):
        """Coefficients a_0..a_d of the interpolant in the monomial basis."""
        F = self.F
        if self.p is not None:
            p = self.p
            coeffs = [0]
            for k in range(len(self.c) - 1, -1, -1):
                xk = self.xs[k]
                new = [0] * (len(coeffs) + 1)
                for i, a in enumerate(coeffs):
                    new[i + 1] += a
                    new[i] -= a * xk
                new[0] += self.c[k]
                coeffs = [v % p for v in new]
            while len(coeffs) > 1 and coeffs[-1] == 0:
                coeffs.pop()
            return [F(v) for v in coeffs]
        coeffs = [F.zero]
        for k in range(len(self.c) - 1, -1, -1):
            # coeffs <- coeffs * (x - xs[k]) + c[k]
            new = [F.zero] * (len(coeffs) + 1)
            for i, a in enumerate(coeffs):
                new[i + 1] = new[i + 1] + a
                new[i] = new[i] - a * self.xs[k]
            new[0] = new[0] + self.c[k]
            coeffs = new
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs


def interpolate(xs, ys, F):
    ni = NewtonInterpolator(F)
    for x, y in zip(xs, ys):
        ni.add(x, y)
    return ni.monomial_coefficients()


def _fresh_point(F, used, rng, step=1):
    """A random point not used before; for an even exponent step also -x is
    marked used (x^step = (-x)^step would break the interpolation)."""
    while True:
        x = F.random_element(rng)
        if x not in used:
            used.add(x)
            if step % 2 == 0:
                used.add(-x)
            return x


def univariate_laurent(f, F, lower=-16, step=1, extra=3, budget=64, max_points=20000,
                       rng=random):
    """Reconstruct a univariate Laurent polynomial ``f`` (a black box F -> F)
    whose exponents are multiples of ``step``, with early termination.

    Returns ``{exponent: coefficient}``.  ``lower`` is a guess for the minimal
    exponent.  If the guess is too high, f * x^{-lower} is not a polynomial and
    Newton interpolation does not terminate within ``budget`` points; the guess
    and the budget are then doubled.  A result whose lowest coefficient sits
    exactly at the guess is also retried with a lower guess.
    """
    lower -= lower % step
    pts, used = [], set()          # black-box values are kept across retries

    def point(k):
        while len(pts) <= k:
            b = _fresh_point(F, used, rng, step)
            try:
                pts.append((b, f(b)))
            except BadPoint:
                continue
        return pts[k]
    while True:
        if budget > max_points:
            raise RuntimeError("univariate reconstruction did not terminate")
        ni = NewtonInterpolator(F)
        agree, k = 0, 0
        while agree < extra and len(ni) <= budget:
            b, fb = point(k)
            k += 1
            y = fb * b ** (-lower)
            X = b ** step
            if len(ni) and ni.value(X) == y:
                agree += 1
            else:
                agree = 0
            ni.add(X, y)
        if agree < extra:
            # the span exceeds the budget: assume it is roughly balanced
            lower = min(2 * lower - step, -step * (budget // 2))
            lower -= lower % step
            budget = 2 * budget
            continue
        coeffs = ni.monomial_coefficients()
        nz = [i for i, a in enumerate(coeffs) if a != 0]
        if not nz:
            return {}
        if nz[0] == 0:
            lower = 2 * lower - step
            continue
        return {lower + step * i: coeffs[i] for i in nz}


def detect_exponent_box(f2, F, steps=(1, 1), lines=2, lower=-16, rng=random):
    """Find the exponent ranges of a bivariate Laurent black box f2(A, q) by
    univariate reconstruction along random lines (generic w.h.p.)."""
    sA, sq = steps
    lower_A = lower - lower % sA
    lower_q = lower - lower % sq
    boxA, boxq = [None, None], [None, None]
    for _ in range(lines):
        q0 = F.random_element(rng)
        dA = univariate_laurent(lambda a: f2(a, q0), F, lower=lower_A, step=sA, rng=rng)
        A0 = F.random_element(rng)
        dq = univariate_laurent(lambda q: f2(A0, q), F, lower=lower_q, step=sq, rng=rng)
        for box, d in ((boxA, dA), (boxq, dq)):
            if d:
                lo, hi = min(d), max(d)
                box[0] = lo if box[0] is None else min(box[0], lo)
                box[1] = hi if box[1] is None else max(box[1], hi)
    return tuple(boxA), tuple(boxq)


def dense_bivariate(f2, F, A_range, q_range, steps=(1, 1), rng=random, qsym=False):
    """Interpolate f2(A, q) = sum c_{ij} A^i q^j with i in A_range, j in q_range
    (exponents multiples of the respective step).  Returns {(i, j): c}.
    ``qsym``: f2(A, -1/q) = f2(A, q) (self-conjugate R); every evaluated q-row
    is reused at -1/q, halving the evaluations."""
    sA, sq = steps
    (a0, a1), (q0e, q1e) = A_range, q_range
    if a0 is None or q0e is None:
        return {}
    nA = (a1 - a0) // sA + 1
    nq = (q1e - q0e) // sq + 1
    usedA, usedq = set(), set()
    Avals = [_fresh_point(F, usedA, rng, sA) for _ in range(nA)]
    # rows[j] = coefficients in A (length nA) at the j-th q-point
    qpts, rows = [], []
    while len(rows) < nq:
        qv = _fresh_point(F, usedq, rng, sq)
        try:
            ys = [f2(a, qv) * a ** (-a0) for a in Avals]
        except BadPoint:
            continue
        ni = NewtonInterpolator(F)
        for a, y in zip(Avals, ys):
            ni.add(a ** sA, y)
        c = ni.monomial_coefficients()
        c += [F.zero] * (nA - len(c))
        qpts.append(qv)
        rows.append(c)
        if qsym and len(rows) < nq:
            qm = -1 / qv
            if qm not in usedq and qm ** sq != qv ** sq:
                usedq.update((qm, -qm))
                qpts.append(qm)
                rows.append(list(c))
    out = {}
    for i in range(nA):
        ni = NewtonInterpolator(F)
        for qv, row in zip(qpts, rows):
            ni.add(qv ** sq, row[i] * qv ** (-q0e))
        c = ni.monomial_coefficients()
        for j, v in enumerate(c):
            if v != 0:
                out[(a0 + sA * i, q0e + sq * j)] = v
    return out
