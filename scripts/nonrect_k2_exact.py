"""Exact reconstruction of the k = 2 family P_d(q, lam) = H_{(r+d, r)}(A = q^-m, q)|_{q^r = lam} (gl(2|m+2)).

As nonrect_exact.py: at all q' = e^{2 pi i s/N} (N prime, s = 1..N-1) the lam-coefficients are fitted from
NR values of r in [2, RMAX] chosen so that lam = q'^r spreads over the circle; each coefficient is then fitted as a
Laurent polynomial in q' with exponents in [-W, W] and rounded.  Reports rounding error, exact agreement with
data/homfly, P(1, lam) vs Delta(lam^2)^4 (gl(2|2): 4 odd positive roots), lam-support and symmetries.

usage: python3 scripts/nonrect_k2_exact.py PAR KNOT d J W N [NR RMAX]   (env NPROC)
"""
import cmath
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_family as nf  # noqa: E402
import nonrect_k2_family as k2  # noqa: E402
import nonrect_supergroup as ns  # noqa: E402

q, lam = sp.symbols("q lam")


def sample(args):
    par, word, d, s_, N, NR, RMAX, J = args
    qd = cmath.exp(2j * cmath.pi * s_ / N)
    cand = list(range(2, RMAX + 1))
    ang = {r: (s_ * r) % N for r in cand}
    pick = []
    for t in range(NR):
        target = t * N / NR
        best = min((r for r in cand if r not in pick),
                   key=lambda r: (min(abs(ang[r] - target), N - abs(ang[r] - target)), r))
        pick.append(best)
    vals = np.array([k2.value(par, (r + d, r), word, qd)[0] for r in pick])
    c, res, cond = nf.fit(vals, pick, qd, J)
    return [c[j] for j in range(-J, J + 1)], res, cond


def main():
    par = [int(x) for x in sys.argv[1]]
    knot, d, J, W, N = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    NR = int(sys.argv[7]) if len(sys.argv) > 7 else 2 * J + 7
    RMAX = int(sys.argv[8]) if len(sys.argv) > 8 else 45
    m = par.count(1) - par.count(0)
    word = ns.braid_word(ns.knot_rows()[knot])
    with Pool(int(os.environ.get("NPROC", "4"))) as pool:
        out = pool.map(sample, [(par, word, d, s_, N, NR, RMAX, J) for s_ in range(1, N)], chunksize=1)
    C = np.array([o[0] for o in out])
    maxres = max(o[1] for o in out)
    maxcond = max(o[2] for o in out)
    Q = np.exp(2j * np.pi * np.arange(1, N) / N)
    ex = np.arange(-W, W + 1)
    V = Q[:, None] ** ex[None, :]
    alpha, *_ = np.linalg.lstsq(V, C, rcond=None)
    rnd = np.round(alpha.real).astype(int)
    round_err = np.abs(alpha - rnd).max()
    resid = np.abs(V @ rnd - C).max()
    edge = np.abs(rnd[[0, -1], :]).max()
    P = sp.expand(sum(int(rnd[a, j]) * q ** int(ex[a]) * lam ** (j - J)
                      for a in range(len(ex)) for j in range(2 * J + 1) if rnd[a, j]))
    print("%s d=%d m=%d N=%d J=%d W=%d: max fit resid %.1e cond %.1e round_err %.1e resid %.1e edge_coef %d" % (
        knot, d, m, N, J, W, maxres, maxcond, round_err, resid, edge))
    for r in range(2, 8):
        t = ns.data_poly(knot, (r + d, r))
        if t is None:
            continue
        H = sum(c * q ** (-m * i + j) for i, j, c in t)
        print("  data %s: exact match %s" % ([r + d, r], sp.expand(P.subs(lam, q ** r) - H) == 0))
    t1 = ns.data_poly(knot, (1,))
    Dl = sp.expand(sum(c * lam ** j for i, j, c in t1))
    P1 = sp.factor(P.subs(q, 1))
    print("  P(1,lam) = %s" % P1)
    for e in (2, 3, 4):
        print("  P(1,lam) / Delta(lam^2)^%d = %s" % (e, sp.factor(sp.cancel(P.subs(q, 1) / Dl ** e))))
    for a in range(-8, 9):
        if sp.expand(P.subs(lam, q ** a / lam) - P) == 0:
            print("  symmetry lam -> q^%d/lam" % a)
    if sp.expand(P.subs(q, 1 / q) - P) == 0:
        print("  symmetry q -> 1/q")
    pl = sp.Poly(sp.expand(P * lam ** (2 * J)), lam)
    degs = [mm[0] - 2 * J for mm in pl.monoms()]
    print("  lam-support %d..%d, e = %s" % (min(degs), max(degs), sp.Rational(max(degs) - min(degs), 4)))
    print("  top coefficient: %s" % sp.factor(pl.coeffs()[0]))
    print("  P = %s" % P)


if __name__ == "__main__":
    main()
