"""M_R = R (x) Rbar: symbolic checks of the channel factors of formula II
(methods/interpolation.py) in exact rational functions of (A, q).

    Z_R^c = F_c(R) = d_c * Phi_c(R),   Phi_c(R) = sum_X E_cX (chi^_X(R) - 1)

1. symmetric R = [r], channel c = [[j],[j]]:
       F_c([r]) = binom(r, j) prod_{i<j} D_{r+i} D_{i-1}             (j <= 4, r <= 6)
       d_c      = D_{-1} (D_0..D_{j-2})^2 D_{2j-1} / ([j]! {q}^j)^2
       Phi_c    = binom(r, j) ([j]!{q}^j)^2 prod_{i<j} D_{r+i} / (D_{2j-1} prod_{i<j-1} D_i)
2. adjoint channel, any R (|R| <= 6):  d_Adj (chi^_Adj(R) - 1) = sum_{b in R} D_{c_b+h_b} D_{c_b-h_b}
3. factorized F_c(R) for a few non-rectangular R (printed in D_n notation).

Needs sympy.  PYTHONPATH=src python scripts/mr_composite_check.py
"""
import sys
from fractions import Fraction

import sympy as sp
from sympy import QQ

from homfly.methods import interpolation as I
from homfly.reps.partitions import boxes, hook, partitions

A, q = sp.symbols("A q")
K = QQ.frac_field(A, q)


class SymField:
    characteristic = 0
    zero, one = K.zero, K.one

    def __call__(self, x):
        if isinstance(x, Fraction):
            return K(QQ(x.numerator, x.denominator))
        return K(x)


pt = I.Point(SymField(), K(A), K(q))
S = I.single
Dn = lambda k: K(A) * K(q) ** k - 1 / (K(A) * K(q) ** k)          # noqa: E731
br = lambda n: (K(q) ** n - K(q) ** -n) / (K(q) - 1 / K(q))       # noqa: E731
eps = K(q) - 1 / K(q)


def qfac(n):
    v = K.one
    for i in range(1, n + 1):
        v *= br(i)
    return v


def binom(r, j):
    return qfac(r) / qfac(j) / qfac(r - j)


def dform(x):
    """Factor x as prod D_n^{e_n} * (rest)."""
    num, den = sp.fraction(sp.factor(K.to_sympy(x)))
    pw, rest = {}, sp.Integer(1)
    for f, m in list(sp.factor_list(num)[1]) + [(f, -m) for f, m in sp.factor_list(den)[1]]:
        p = sp.Poly(f, A, q)
        t = p.terms()
        if p.degree(A) == 1 and len(t) == 2 and t[0][0][0] == 1 and t[1][0][0] == 0:
            n = t[0][0][1] - t[1][0][1]
            pw[n] = pw.get(n, 0) + m
        else:
            rest *= f ** m
    s = " ".join("D%d" % n + ("^%d" % (v // 2) if v // 2 != 1 else "") for n, v in sorted(pw.items()))
    return s, sp.factor(rest)


def main():
    for j in range(1, 5):
        c = S((j,))
        d = K.one * Dn(-1) * Dn(2 * j - 1) / (qfac(j) * eps ** j) ** 2
        for i in range(j - 1):
            d *= Dn(i) ** 2
        assert pt.d(c) == d, j
        for r in range(j, 7):
            Z = binom(r, j)
            for i in range(j):
                Z *= Dn(r + i) * Dn(i - 1)
            assert pt.F_c(c, (r,)) == Z, (j, r)
            Phi = binom(r, j) * (qfac(j) * eps ** j) ** 2 / Dn(2 * j - 1)
            for i in range(j):
                Phi *= Dn(r + i)
            for i in range(j - 1):
                Phi /= Dn(i)
            assert pt.F_c(c, (r,)) / pt.d(c) == Phi, (j, r)
    print("1. symmetric: F = binom prod D_{r+i}D_{i-1}, d_c and Phi closed forms: OK (j<=4, r<=6)")

    adj, n = S((1,)), 0
    for m in range(1, 7):
        for R in partitions(m):
            s = sum((Dn(b[1] - b[0] + hook(R, b)) * Dn(b[1] - b[0] - hook(R, b)) for b in boxes(R)), K.zero)
            assert pt.d(adj) * pt.psi(adj, R) == s, R
            n += 1
    print("2. adjoint: d_Adj (chi^_Adj(R)-1) = sum_b D_{c+h} D_{c-h}: OK on %d diagrams" % n)

    print("3. F_c(R) for non-rectangular R:")
    for R in [(2, 1), (3, 1), (2, 1, 1)] + ([(3, 2), (4, 1)] if "--more" in sys.argv else []):
        print("  R =", R)
        for c in I.channels_of(R):
            s, rest = dform(pt.F_c(c, R))
            print("    %-28s %-36s | %s" % (c, s, rest))


if __name__ == "__main__":
    main()
