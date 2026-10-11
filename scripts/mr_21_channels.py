"""Do the symmetric knot functions survive as channel functions of [2,1]?

Formula II (methods/interpolation.py) writes H_[2,1] over the channels of
[2,1] (x) [2,1]bar:  Adj, [2,2bar], [11,11bar], pair {[2],[11]}, [21,21bar].
For every knot with H_[1], H_[2], H_[1,1], H_[2,1] in data/homfly we take the
knot functions of Adj, [2], [11] from the symmetric (antisymmetric) DE,
G_1 = (H_1-1)/Z, G_2, G_11, and test whether the remainder

    Rem = H_21 - 1 - Z^Adj G_1 - Z^[2] G_2 - Z^[11] G_11 = Z^pair G_p + Z^[21] G_21

with Z^pair = alpha D_2 D_-2, Z^[21] = beta D_2 D_-2, alpha = {q}^4[3]^2 (unit), beta = D_{+-1}D_{+-3},
admits Laurent G_p, G_21, or G's with poles D_0 and/or D_{+-2}
(ideal membership via a Groebner basis of (alpha, beta) over QQ[t, A, q], t = 1/(Aq)).

Output (tsv): knot, Rem divisible by D2D-2, and membership of
n, n D0, n D2D-2, n D0D2D-2, n (D0D2D-2)^2 in (alpha, beta),  n = Rem/(D2D-2).

    PYTHONPATH=src python scripts/mr_21_channels.py [-j 4] > data/mr_21_channels.tsv
"""
import glob
import multiprocessing as mp
import os
import sys
from fractions import Fraction

import sympy as sp
from sympy import QQ

from homfly.methods import interpolation as I

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

a_n = sp.expand((q ** 2 - 1) ** 4 * (q ** 4 + q ** 2 + 1) ** 2)
b_n = sp.expand((A ** 2 - q ** 2) * (A ** 2 * q ** 2 - 1) * (A ** 2 - q ** 6) * (A ** 2 * q ** 6 - 1))
GB = sp.groebner([a_n, b_n, t * A * q - 1], t, A, q, order="lex", domain="QQ")
d0, dp2, dm2 = A ** 2 - 1, A ** 2 * q ** 4 - 1, A ** 2 - q ** 4


def parse(s):
    return K.from_sympy(sp.sympify(s.replace("^", "**"), locals={"A": A, "q": q}))


def load(fn):
    out, R = {}, None
    for line in open(fn):
        if line.startswith("R = "):
            R = tuple(int(x) for x in line[4:].strip().strip("[]").split(","))
        elif line.startswith("H = ") and R:
            out[R] = line[4:].strip()
    return out


def work(fn):
    d = load(fn)
    if not all(R in d for R in REPS):
        return None
    H = {R: parse(d[R]) for R in REPS}
    G1 = (H[(1,)] - 1) / Fc[(c1, (1,))]
    G2 = (H[(2,)] - 1 - Fc[(c1, (2,))] * G1) / Fc[(c2, (2,))]
    G11 = (H[(1, 1)] - 1 - Fc[(c1, (1, 1))] * G1) / Fc[(c11, (1, 1))]
    Rem = H[(2, 1)] - 1 - Fc[(c1, (2, 1))] * G1 - Fc[(c2, (2, 1))] * G2 - Fc[(c11, (2, 1))] * G11
    n, den = sp.fraction(sp.together(K.to_sympy(Rem / (Dn(2) * Dn(-2)))))
    n = sp.expand(n)
    mono = len(sp.Poly(den, A, q).terms()) == 1
    inn = lambda f: GB.reduce(sp.expand(f))[1] == 0              # noqa: E731
    g = d0 * dp2 * dm2
    return (os.path.basename(fn)[:-4], mono, inn(n), inn(n * d0), inn(n * dp2 * dm2), inn(n * g), inn(n * g * g))


def main():
    j = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 1
    files = sorted(glob.glob(os.path.join(ROOT, "data", "homfly", "*.txt")))
    print("knot\tD2D-2|Rem\tlaurent\tpole_D0\tpole_D2D-2\tpole_D0D2D-2\tpole_(D0D2D-2)^2")
    with mp.Pool(j) as p:
        for r in p.imap(work, files, chunksize=8):
            if r:
                print("\t".join(str(int(x)) if isinstance(x, bool) else x for x in r), flush=True)


if __name__ == "__main__":
    main()
