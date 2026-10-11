"""Box reading of the single-channel factors Z_R^[mu,mu] = F_c(R) of formula II.

For every R (|R| <= N) and every single channel c = [mu, mu] of R (x) Rbar the
A-dependent part of F_c(R) is compared with the contents of two placements of mu:

  T: mu glued to the right of R row by row, box (i, j) -> (i, R_i + j),  D_{R_i + j - i}
  U: mu glued under R column by column,  box (i, j) -> (R'_j + i, j),  D_{j - i - R'_j}

(for R = [r^s]: T = D_{r+c}, U = D_{c-s}, the rectangular formula).
Columns: R, mu, multiplicity m_c(R), factorized (A-dependence is a product of D_n),
D-multiset of F_c, T+U multiset, match, T and U share a content.

    PYTHONPATH=src python scripts/mr_channel_factors.py [N=5]
"""
import sys
from collections import Counter

import sympy as sp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mr_composite_check import A, K, q, pt, I  # noqa: E402
from homfly.reps.partitions import conjugate, partitions  # noqa: E402


def dmult(x):
    num, den = sp.fraction(sp.factor(K.to_sympy(x)))
    pw, fact = Counter(), True
    for f, m in list(sp.factor_list(num)[1]) + [(f, -m) for f, m in sp.factor_list(den)[1]]:
        p = sp.Poly(f, A, q)
        t = p.terms()
        if p.degree(A) == 0 or (len(t) == 1):
            continue                                  # q-number or monomial
        if p.degree(A) == 1 and len(t) == 2 and t[0][0][0] == 1 and t[1][0][0] == 0:
            pw[t[0][0][1] - t[1][0][1]] += m
        else:
            fact = False
    return Counter({n: v // 2 for n, v in pw.items() if v}), fact


def glue(R, mu):
    RT = conjugate(tuple(R))
    cells = [(i, j) for i, l in enumerate(mu) for j in range(l)]
    T = Counter((R[i] if i < len(R) else 0) + j - i for i, j in cells)
    U = Counter(j - i - (RT[j] if j < len(RT) else 0) for i, j in cells)
    return T, U


def fmt(c):
    return " ".join("D%d" % n + ("^%d" % v if v != 1 else "") for n, v in sorted(c.items()))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print("R\tmu\tmult\tfactorized\tF_c\tT+U\tmatch\tT&U")
    for m in range(1, N + 1):
        for R in partitions(m):
            for c in I.channels_of(R):
                if c[0] != "s":
                    continue
                pw, fact = dmult(pt.F_c(c, R))
                T, U = glue(R, c[1])
                print("\t".join(map(str, [list(R), list(c[1]), I.presence(c, R), int(fact), fmt(pw),
                                          fmt(T + U), int(fact and pw == T + U), int(bool(T & U))])), flush=True)


if __name__ == "__main__":
    main()
