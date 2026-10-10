"""The tower members P_m(q, lam) = H_[r](A = q^-m)|_{q^r = lam} and their Links-Gould checks.

P_m is assembled from the r-independent G_k of the differential expansion (see
scripts/defect_scan.py):

    P_m = 1 + sum_k [r k]_q D_{r+k-1} ... D_r G_k(q^-m, q),   q^r = lam,

the sum terminating at k = e_m (D_m | G_k for k > e_m).  It is "complete" when the
last available G_k vanishes at A = q^-m; otherwise only a lower bound for e_m is seen.

For m = 1 the candidate identification is P_1(q, lam) = LG^{2,1}(t0, t1) with
t0 = lam^-2, t1 = q^2 lam^2 (t0 t1 = q^2), up to mirror.  Checked per knot:

  sym   P_1(q, lam) = P_1(q, q/lam)               (LG symmetric in t0 <-> t1)
  mmr   P_1(1, lam) = Delta(lam^2)^2              (LG(t0, 1/t0) = Delta(t0)^2)
  qi    P_1(i, lam) = +- Delta(lam^4)             (LG(t0, -1/t0) = Delta(t0^2))
  e1    = (lam-span of P_1) / 4                   (Kohli-Tahar: span LG <= 4g  <=>  e1 <= 2g)

and, with --roots M, the root-of-unity factorization for m <= M:

  q a primitive 2k-th root of unity, 2 <= k <= m+1:
      P_m = +- Delta(lam^2)^a Delta(lam^{2k})^b,   a + k b = m + 1,  b = floor((m+1)/k)

(a consequence of H_[r+k] = H_[r] H_[k] at q^{2k} = 1, Kononov-Morozov 1505.06170).

usage: python3 scripts/links_gould_check.py [--roots M] [knot ...] > report.tsv
       (default: every knot in data/homfly; needs sympy and python-flint;
        genus/tau/signature columns need database_knotinfo, else '-')
"""
import glob
import os
import sys
from multiprocessing import Pool

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import defect_scan as ds  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "homfly")
q, lam = sp.symbols("q lam")

try:
    from database_knotinfo import link_list
    KI = {r["name"]: r for r in link_list()[1:]}
except ImportError:
    KI = {}


def at_line(Lp, m):
    """L(A, q) at A = q^-m as a sympy expression in q"""
    return sum(int(c) * q ** (-m * (i + Lp.sa) + j + Lp.sq) for (i, j), c in zip(Lp.p.monoms(), Lp.p.coeffs()))


def symmetric_G(H):
    G = {}
    for k in sorted(H):
        if k != len(G) + 1:
            break
        N = H[k] - ds.ONE
        for j in range(1, k):
            t = ds.qbinom(k, j)
            for i in range(j):
                t = t * ds.D(k + i)
            N = N - t * G[j]
        den = ds.ONE
        for i in range(k):
            den = den * ds.D(k + i)
        G[k] = N.divexact(den)
    return G


def tower_member(G, m):
    """(P_m(q, lam), complete)"""
    tot, last = sp.Integer(1), 0
    for k, Gk in G.items():
        g = sp.expand(at_line(Gk, m))
        if g == 0:
            continue
        last = k
        qint = lambda n: (q ** n - q ** -n) / (q - 1 / q)  # noqa: E731
        binom = sp.prod([(lam * q ** -i - q ** i / lam) / (q - 1 / q) for i in range(k)]) / sp.prod(
            [qint(i + 1) for i in range(k)])
        Ds = sp.prod([lam * q ** (i - m) - q ** (m - i) / lam for i in range(k)])  # D_{r+i} at A = q^-m
        tot += binom * Ds * g
    return sp.expand(sp.cancel(sp.together(tot))), last < max(G)


def alexander(H1):
    """Delta(lam^2) = H_[1](A = 1, q = lam)"""
    return sp.expand(sum(int(c) * lam ** (j + H1.sq) for (i, j), c in zip(H1.p.monoms(), H1.p.coeffs())))


def lam_span(p):
    e = [mon[0] for mon in sp.Poly(sp.expand(p * lam ** 400), lam).monoms()]
    return max(e) - min(e)


def at_root(p, k):
    """numerator, denominator of p reduced mod the cyclotomic polynomial Phi_2k(q)"""
    num, den = sp.fraction(sp.together(p))
    phi = sp.cyclotomic_poly(2 * k, q)
    return sp.rem(sp.expand(num), phi, q), sp.rem(sp.expand(den), phi, q), phi


def root_factorization(p, D, m):
    ok = True
    for k in range(2, m + 2):
        rn, rd, phi = at_root(p, k)
        b = (m + 1) // k
        t = sp.expand(D ** (m + 1 - k * b) * sp.expand(D.subs(lam, lam ** k)) ** b)
        ok &= any(sp.rem(sp.expand(rn - s * rd * t), phi, q) == 0 for s in (1, -1))
    return ok


def check(args):
    knot, roots = args
    H = ds.load(os.path.join(DATA, knot + ".txt"))
    G = symmetric_G(H)
    D = alexander(H[1])
    p, complete = tower_member(G, 1)
    sym = sp.expand(p.subs(lam, q / lam) - p) == 0
    mmr = sp.expand(p.subs(q, 1) - D ** 2) == 0
    pi, D4 = sp.expand(p.subs(q, sp.I)), sp.expand(D.subs(lam, lam ** 2))
    qi = sp.expand(pi - D4) == 0 or sp.expand(pi + D4) == 0
    row = [knot, "complete" if complete else "lower_bound", sym, mmr, qi, lam_span(p) // 4,
           lam_span(D) // 4]
    r = KI.get(knot)
    row += [r["three_genus"], r["ozsvath_szabo_tau_invariant"], r["signature"]] if r else ["-"] * 3
    if roots:
        rr = []
        for m in range(1, roots + 1):
            pm, cm = tower_member(G, m)
            rr.append("%s%s" % (root_factorization(pm, D, m), "" if cm else "?"))
        row.append(",".join(rr))
    return row


def main():
    argv, roots = sys.argv[1:], 0
    if argv[:1] == ["--roots"]:
        roots, argv = int(argv[1]), argv[2:]
    knots = argv or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(DATA, "*.txt"))
                           if not os.path.basename(f).startswith("L"))  # knots only
    print("knot\tP1\tsym\tmmr\tqi\te1\tdegAlex\tgenus\ttau\tsignature" + ("\troots_m1..%d" % roots if roots else ""))
    with Pool(os.cpu_count()) as pool:
        for row in pool.imap(check, [(k, roots) for k in knots], chunksize=4):
            print("\t".join(map(str, row)), flush=True)


if __name__ == "__main__":
    main()
