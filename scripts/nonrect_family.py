"""H2: the family R(r) = (r, mu) on the line A = q^-m as a function of lambda = q^r ("colored Links-Gould").

Vertex model of U_q gl(1|m+1) on V_{(r,mu)} (scripts/nonrect_supergroup.py), independent of data/homfly.
In the data convention (q' = 1/q, A = q'^-m) we fit, at a fixed unit-modulus q',

    H_{(r,mu)}(q'^-m, q') = sum_{j=-J..J} c_j(q') lambda^j,   lambda = q'^r,   r = r0 .. r1   (least squares),

and report the support of c_j (|c_j| > TOL), the lambda-span s, e = s/4, the residual, and the
symmetry lambda -> q'^a / lambda (the a for which c_{-j+s0} = q'^{a j} c_j).  If data exist for some r
they are compared with the vertex value as a cross-check.

usage: python3 scripts/nonrect_family.py PARITIES MU KNOT [r0 r1 J theta rho]   (q' = rho e^{i theta})
       MU = '-' for the symmetric family, '1' for [r,1], '1,1' for [r,1,1], '2' for [r,2] ...
"""
import cmath
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_supergroup as ns  # noqa: E402

TOL = 1e-7
CHECK = os.environ.get("CHECK", "1") == "1"
SCAL = [0.0]


def values(par, mu, word, qd, rs, knot=None):
    """vertex values at data-q' = qd for R = (r,)+mu, r in rs; also max |vertex - data| where data exist"""
    q = 1 / qd
    m = par.count(1) - par.count(0)
    out, dd, sc = [], 0.0, 0.0
    for r in rs:
        R = (r,) + tuple(mu)
        mod = ns.Module(par, R, q)
        v, dv = mod.knot_value(word, check_scalar=CHECK)
        sc = max(sc, dv / max(1, abs(v)))
        out.append(v)
        if knot:
            h = ns.data_value(knot, R, qd ** (-m), qd)
            if h is not None:
                dd = max(dd, abs(v - h) / max(1, abs(h)))
    SCAL[0] = max(SCAL[0], sc)
    return np.array(out), dd


def fit(vals, rs, qd, J):
    lam = np.array([qd ** r for r in rs])
    js = np.arange(-J, J + 1)
    Vm = lam[:, None] ** js[None, :]
    c, *_ = np.linalg.lstsq(Vm, vals, rcond=None)
    res = np.abs(Vm @ c - vals).max() / max(1, np.abs(vals).max())
    return dict(zip(js.tolist(), c)), res, np.linalg.cond(Vm)


def analyse(c, qd):
    sup = [j for j, v in c.items() if abs(v) > TOL * max(1, max(abs(x) for x in c.values()))]
    lo, hi = min(sup), max(sup)
    syms = []
    for a in range(-4 * (hi - lo) - 10, 4 * (hi - lo) + 11):
        # P(q, q^a / lam) = sum c_j q^{aj} lam^{-j}; need = P(q, lam): c_{-j} = c_j q^{a j}; centre lo+hi = 0?
        if lo + hi != 0:
            continue
        if all(abs(c.get(-j, 0) - c[j] * qd ** (a * j)) < 1e-6 * max(1, abs(c[j])) for j in sup):
            syms.append(a)
    return lo, hi, sup, syms


def main():
    par = [int(x) for x in sys.argv[1]]
    mu = () if sys.argv[2] == "-" else tuple(int(x) for x in sys.argv[2].split(","))
    knot = sys.argv[3]
    m = par.count(1) - par.count(0)
    r0 = int(sys.argv[4]) if len(sys.argv) > 4 else max(m + 1, mu[0] if mu else 1)
    r1 = int(sys.argv[5]) if len(sys.argv) > 5 else r0 + 30
    J = int(sys.argv[6]) if len(sys.argv) > 6 else 12
    theta = float(sys.argv[7]) if len(sys.argv) > 7 else 0.7
    rho = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    qd = rho * cmath.exp(1j * theta)
    rows = ns.knot_rows()
    word = ns.braid_word(rows[knot])
    rs = list(range(r0, r1 + 1))
    vals, dd = values(par, mu, word, qd, rs, knot)
    c, res, cond = fit(vals, rs, qd, J)
    lo, hi, sup, syms = analyse(c, qd)
    span = hi - lo
    print("%s m=%d mu=%s r=%d..%d J=%d theta=%g: support lam^%d..lam^%d span=%d e=%g parity=%s sym_a=%s "
          "residual=%.1e cond=%.1e data_check=%.1e scalar_dev=%.1e |q|=%g" % (
              knot, m, list(mu), r0, r1, J, theta, lo, hi, span, span / 4,
              "even" if all(j % 2 == 0 for j in sup) else ("odd" if all(j % 2 for j in sup) else "mixed"),
              syms, res, cond, dd, SCAL[0], rho))


if __name__ == "__main__":
    main()
