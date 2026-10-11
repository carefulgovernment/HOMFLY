"""Exact (two-variable) check of the [2,1] channel DE with end-factor poles.

Same set-up as research/MR/mr_21_channels.py: G_1, G_2, G_11 from the symmetric
DE (G_2, G_11 may carry the pole End = D_0), remainder
    Rem = H_21 - 1 - sum_{Adj,[2],[11]} Z G  =  D_2 D_-2 (alpha G_p + beta G_21),
alpha = {q}^4 [3]^2,  beta = D_1 D_-1 D_3 D_-3,  n = Rem / (D_2 D_-2).
Ideal membership over QQ[A^+-1, q^+-1] (Groebner basis) of
    n in (alpha, beta)                      Laurent G_p, G_21
    n in (alpha, D_3 D_-3)                  G_21 with the pole End([2,1]) = D_1 D_-1
    n D_1 D_-1 in (alpha, beta)             both with the pole D_1 D_-1
    n D_2 D_-2 in (alpha, beta)             both with the pole D_2 D_-2 (MR thread)

    PYTHONPATH=src python scripts/de_21_exact.py 5_1 7_1 8_19 ...
"""
import os
import sys
from fractions import Fraction

import sympy as sp
from sympy import QQ

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from homfly.methods import interpolation as I  # noqa: E402

A, q, t = sp.symbols("A q t")
K = QQ.frac_field(A, q)
ROOT = os.path.join(os.path.dirname(__file__), "..")


class SymField:
    characteristic = 0
    zero, one = K.zero, K.one

    def __call__(self, x):
        if isinstance(x, Fraction):
            return K(QQ(x.numerator, x.denominator))
        return K(x)


pt = I.Point(SymField(), K(A), K(q))
REPS = [(1,), (2,), (1, 1), (2, 1)]
Fc = {(c, R): pt.F_c(c, R) for R in REPS for c in I.channels_of(R)}
c1, c2, c11 = I.single((1,)), I.single((2,)), I.single((1, 1))
Dn = lambda n: K(A) * K(q) ** n - 1 / (K(A) * K(q) ** n)     # noqa: E731
Dp = {k: (A ** 2 * q ** (2 * k) - 1) for k in range(1, 5)}
Dm = {k: (A ** 2 - q ** (2 * k)) for k in range(1, 5)}
alpha = sp.expand((q ** 2 - 1) ** 4 * (q ** 4 + q ** 2 + 1) ** 2)
beta = sp.expand(Dp[1] * Dm[1] * Dp[3] * Dm[3])
GB = sp.groebner([alpha, beta, t * A * q - 1], t, A, q, order="lex", domain="QQ")
GB3 = sp.groebner([alpha, sp.expand(Dp[3] * Dm[3]), t * A * q - 1], t, A, q, order="lex", domain="QQ")


def load(name):
    out, R = {}, None
    for line in open(os.path.join(ROOT, "data", "homfly", name + ".txt")):
        if line.startswith("R = "):
            R = tuple(int(x) for x in line[4:].strip().strip("[]").split(","))
        elif line.startswith("H = ") and R:
            out[R] = K.from_sympy(sp.sympify(line[4:].strip().replace("^", "**"), locals={"A": A, "q": q}))
    return out


def work(name):
    H = load(name)
    G1 = (H[(1,)] - 1) / Fc[(c1, (1,))]
    G2 = (H[(2,)] - 1 - Fc[(c1, (2,))] * G1) / Fc[(c2, (2,))]
    G11 = (H[(1, 1)] - 1 - Fc[(c1, (1, 1))] * G1) / Fc[(c11, (1, 1))]
    Rem = H[(2, 1)] - 1 - Fc[(c1, (2, 1))] * G1 - Fc[(c2, (2, 1))] * G2 - Fc[(c11, (2, 1))] * G11
    n, den = sp.fraction(sp.together(K.to_sympy(Rem / (Dn(2) * Dn(-2)))))
    n = sp.expand(n)
    assert len(sp.Poly(den, A, q).terms()) == 1, "D2D-2 does not divide Rem"
    inn = lambda G, f: G.reduce(sp.expand(f))[1] == 0              # noqa: E731
    return {
        "knot": name,
        "laurent": inn(GB, n),
        "pole_End21_in_G21": inn(GB3, n),
        "pole_D1D-1": inn(GB, n * Dp[1] * Dm[1]),
        "pole_D2D-2": inn(GB, n * Dp[2] * Dm[2]),
    }


if __name__ == "__main__":
    import json
    for nm in sys.argv[1:]:
        print(json.dumps(work(nm)), flush=True)
