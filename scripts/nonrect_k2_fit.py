"""Analysis of k = 2 two-row families (values from nonrect_k2_family.py vals ...).

Input rows: knot r1 r2 dim Re Im scal health (q' = e^{i theta}, H_{(r1,r2)}(A = q'^-m, q')).
(1) per d = r1 - r2: least-squares fit H = sum_{|j|<=J} c_j(d) lam^j, lam = q'^{r2}, with residual, window test
    (J and J2), held-out validation (fit on even r2, predict odd r2), lam-support, symmetry lam -> q'^a/lam,
    data cross-check (data/homfly, |R| <= 6).
(2) two-variable test: is H(r1, r2) = sum_{a,b} C_{ab} lam1^a lam2^b (lam_i = q'^{r_i}) on the whole grid?
    On the line r1 = r2 + d this means c_j(d) = sum_a C_{a, j-a} q'^{a d}, i.e. every coefficient c_j(d) is a
    Laurent polynomial in q'^d.  Tested by fitting c_j(d) for d in D_fit with exponent window |a| <= A and
    predicting the held-out d (only meaningful if #D_fit > number of exponents).

usage: python3 scripts/nonrect_k2_fit.py FILE theta m J J2 [A]
"""
import cmath
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_family as nf  # noqa: E402
import nonrect_supergroup as ns  # noqa: E402


def main():
    fn, theta, m, J, J2 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    qd = cmath.exp(1j * theta)
    data = defaultdict(dict)
    meta = defaultdict(lambda: [0.0, 0.0, set()])
    for line in open(fn):
        if line.startswith("#") or not line.strip():
            continue
        kn, r1, r2, dim, re, im, sc, he = line.split("\t")
        r1, r2 = int(r1), int(r2)
        data[(kn, r1 - r2)][r2] = complex(float(re), float(im))
        mt = meta[(kn, r1 - r2)]
        if sc.strip() != "nan":
            mt[0] = max(mt[0], float(sc))
        mt[1] = max(mt[1], float(he))
        mt[2].add(int(dim))
    coefs = {}
    print("# per-d fits in lam = q'^r2, q' = e^{i %g}, m = %d" % (theta, m))
    print("knot\td\tdim\tr2_range\tJ\tlam_support\tspan\te=span/4\tparity\tsym_a\tresid\tJ2_support\tJ2_resid\t"
          "holdout_err\tdata_check(n)\tscalar_dev\thealth")
    for (kn, d), dd in sorted(data.items()):
        rs = sorted(dd)
        v = np.array([dd[r] for r in rs])
        c, res, cond = nf.fit(v, rs, qd, J)
        lo, hi, sup, syms = nf.analyse(c, qd)
        c2, res2, _ = nf.fit(v, rs, qd, J2)
        lo2, hi2, _, _ = nf.analyse(c2, qd)
        ev = [i for i, r in enumerate(rs) if r % 2 == 0]
        od = [i for i, r in enumerate(rs) if r % 2 == 1]
        if len(ev) > (hi - lo) + 2:
            # fit on even r2 with the found support only, predict odd r2
            lam = np.array([qd ** r for r in rs])
            js = np.arange(lo, hi + 1)
            V = lam[:, None] ** js[None, :]
            ce, *_ = np.linalg.lstsq(V[ev], v[ev], rcond=None)
            hold = np.abs(V[od] @ ce - v[od]).max() / max(1, np.abs(v).max())
        else:
            hold = float("nan")
        dc, nd = 0.0, 0
        for r in rs:
            h = ns.data_value(kn, (r + d, r), qd ** (-m), qd)
            if h is not None:
                nd += 1
                dc = max(dc, abs(dd[r] - h) / max(1, abs(h)))
        par = "even" if all(j % 2 == 0 for j in sup) else ("odd" if all(j % 2 for j in sup) else "mixed")
        sc, he, dims = meta[(kn, d)]
        print("%s\t%d\t%s\t%d..%d\t%d\t%d..%d\t%d\t%g\t%s\t%s\t%.1e\t%d..%d\t%.1e\t%.1e\t%.1e(%d)\t%.1e\t%.1e" % (
            kn, d, "/".join(map(str, sorted(dims))), rs[0], rs[-1], J, lo, hi, hi - lo, (hi - lo) / 4, par,
            ",".join(map(str, syms)) or "-", res, lo2, hi2, res2, hold, dc, nd, sc, he))
        coefs[(kn, d)] = (c, lo, hi)
    # two-variable test
    A = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    print("# two-variable test: c_j(d) = sum_{|a|<=%d} C_a q'^{a d}; fit on all d but one, predict the held-out d" % A)
    print("knot\tj\td_values\theldout_d\trel_pred_err")
    knots = sorted(set(k for k, _ in coefs))
    for kn in knots:
        ds = sorted(d for k, d in coefs if k == kn)
        if len(ds) < 2 * A + 2:
            print("%s\t-\t%s\t-\tnot enough d values for window |a|<=%d" % (kn, ds, A))
            continue
        allj = sorted(set(j for d in ds for j, x in coefs[(kn, d)][0].items()
                          if abs(x) > 1e-7 * max(1, max(abs(y) for y in coefs[(kn, d)][0].values()))))
        worst = 0.0
        for j in allj:
            cj = np.array([coefs[(kn, d)][0].get(j, 0) for d in ds])
            for t, dh in enumerate(ds):
                keep = [i for i in range(len(ds)) if i != t]
                a = np.arange(-A, A + 1)
                V = np.array([[qd ** (aa * ds[i]) for aa in a] for i in keep])
                C, *_ = np.linalg.lstsq(V, cj[keep], rcond=None)
                pred = sum(C[i] * qd ** (aa * dh) for i, aa in enumerate(a))
                err = abs(pred - cj[t]) / max(1, np.abs(cj).max())
                worst = max(worst, err)
                if t == len(ds) - 1:
                    print("%s\t%d\t%s\t%d\t%.1e" % (kn, j, ",".join(map(str, ds)), dh, err))
        print("%s\tall\t%s\tall\tworst %.1e" % (kn, ",".join(map(str, ds)), worst))


if __name__ == "__main__":
    main()
