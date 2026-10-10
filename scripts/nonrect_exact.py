"""Exact reconstruction of P^mu_m(q, lam) = H_{(r,mu)}(q^-m, q)|_{q^r = lam} from the vertex model.

At many unit-modulus q' = e^{i theta} (theta away from 0 and pi, where the nodes lam = q'^r cluster) the
coefficients c_j(q') of P = sum_j c_j(q') lam^j are fitted from r = r0..r1 (nonrect_family.fit); each c_j is
then fitted as a Laurent polynomial in q' with exponents in the window [WLO, WHI] and rounded to integers.  Checks:
  * rounding error and the residual of the rounded P on all samples,
  * exact agreement with every H_{(r,mu)} in data/homfly (sympy, as Laurent polynomials in q),
  * the q = 1 limit compared with Delta(lam^2)^(m+1) (Delta from H_[1] at A = 1),
  * the symmetry lam -> q^a / lam.

With NDFT=N (prime) the samples are instead all N-th roots of unity except 1 (well conditioned).

usage: [NDFT=N] python3 scripts/nonrect_exact.py PARITIES MU KNOT r0 r1 J W|WLO:WHI [NSAMPLES]
       (q'-exponent window of the coefficients: [-W, W] or [WLO, WHI])
"""
import cmath
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_family as nf  # noqa: E402
import nonrect_supergroup as ns  # noqa: E402

q, lam = sp.symbols("q lam")


def main():
    par = [int(x) for x in sys.argv[1]]
    mu = () if sys.argv[2] == "-" else tuple(int(x) for x in sys.argv[2].split(","))
    knot = sys.argv[3]
    r0, r1, J = map(int, sys.argv[4:7])
    wlo, whi = (map(int, sys.argv[7].split(':')) if ':' in sys.argv[7]
                else (-int(sys.argv[7]), int(sys.argv[7])))
    ns_ = int(sys.argv[8]) if len(sys.argv) > 8 else 3 * (whi - wlo) + 40
    m = par.count(1) - par.count(0)
    word = ns.braid_word(ns.knot_rows()[knot])
    rs = list(range(r0, r1 + 1))
    nr = r1 - r0 + 1
    NDFT = int(os.environ.get("NDFT", "0"))
    C, thetas = [], []
    if NDFT:
        # q' = exp(2 pi i s / NDFT), s = 1..NDFT-1 (NDFT prime, > 2 r_max): exact roots of unity of large
        # order; for each s the r's (< NDFT - 4) are chosen so that lam = q'^r spreads over the circle.
        for s_ in range(1, NDFT):
            qd = cmath.exp(2j * cmath.pi * s_ / NDFT)
            cand = list(range(r0, NDFT - 4))
            ang = {r: (s_ * r) % NDFT for r in cand}
            pick = []
            for t in range(nr):
                target = (t * NDFT) / nr
                best = min((r for r in cand if r not in pick),
                           key=lambda r: (min(abs(ang[r] - target), NDFT - abs(ang[r] - target)), r))
                pick.append(best)
            vals, _ = nf.values(par, mu, word, qd, pick)
            c, res, cond = nf.fit(vals, pick, qd, J)
            C.append([c[j] for j in range(-J, J + 1)])
            thetas.append(2 * np.pi * s_ / NDFT)
    else:
        rng = np.random.default_rng(1)        # random angles: rational angles are roots of unity
        for t in rng.uniform(0, 2 * np.pi, ns_):
            if not 0.45 < t % np.pi < np.pi - 0.45:
                continue
            qd = cmath.exp(1j * t)
            vals, _ = nf.values(par, mu, word, qd, rs)
            c, res, cond = nf.fit(vals, rs, qd, J)
            C.append([c[j] for j in range(-J, J + 1)])
            thetas.append(t)
    C = np.array(C)
    Q = np.exp(1j * np.array(thetas))
    ex = np.arange(wlo, whi + 1)
    V = Q[:, None] ** ex[None, :]
    alpha, *_ = np.linalg.lstsq(V, C, rcond=None)
    rnd = np.round(alpha.real).astype(int)
    round_err = np.abs(alpha - rnd).max()
    resid = np.abs(V @ rnd - C).max()
    P = sum(int(rnd[a, j]) * q ** int(ex[a]) * lam ** (j - J)
            for a in range(len(ex)) for j in range(2 * J + 1) if rnd[a, j])
    P = sp.expand(P)
    print("%s m=%d mu=%s: samples=%d cond(V)=%.1e round_err=%.1e resid=%.1e" % (
        knot, m, list(mu), len(thetas), np.linalg.cond(V), round_err, resid))
    # exact comparison with the data
    for r in rs:
        R = (r,) + mu
        t = ns.data_poly(knot, R)
        if t is None:
            continue
        H = sum(c * q ** (-m * i + j) for i, j, c in t)
        print("  data %s: exact match %s" % (list(R), sp.expand(P.subs(lam, q ** r) - H) == 0))
    t1 = ns.data_poly(knot, (1,))
    Dl = sp.expand(sum(c * lam ** j for i, j, c in t1))    # Delta(lam^2) = H_[1](A=1, q=lam)
    P1 = sp.factor(P.subs(q, 1))
    print("  P(1,lam) = %s" % P1)
    print("  P(1,lam) / Delta(lam^2)^(m+1) = %s" % sp.factor(sp.cancel(P.subs(q, 1) / Dl ** (m + 1))))
    for a in range(-6, 7):
        if sp.expand(P.subs(lam, q ** a / lam) - P) == 0:
            print("  symmetry lam -> q^%d/lam" % a)
    pl = sp.Poly(sp.expand(P * lam ** (2 * J)), lam)
    degs = [mm[0] - 2 * J for mm in pl.monoms()]
    print("  lam-support %d..%d, e = %s" % (min(degs), max(degs), sp.Rational(max(degs) - min(degs), 4)))
    print("  top coefficient: %s" % sp.factor(pl.coeffs()[0]))
    print("  P = %s" % P)


if __name__ == "__main__":
    main()
