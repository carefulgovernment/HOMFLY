"""H2 scan: lambda-span tower members e^mu_m of the families (r, mu) over many knots (vertex model only).

For each knot (KnotInfo braid word) and each (m, mu) in FAMILIES, runs the fit of nonrect_family.py
(lambda = q'^r, q' = e^{i theta}, theta = golden angle, r = r0..r0+NR-1, window |j| <= J) and prints
e = span/4, the symmetry exponent a (P(q, lam) = P(q, q^a/lam)), the fit residual, the scalar deviation
of the (1,1)-tangle and the agreement with the data where they exist.  Columns genus / deg Delta from
database_knotinfo / data.

usage: python3 scripts/nonrect_family_scan.py MAXB [knot ...] > data/nonrect_family_e.tsv
       FAMILIES env, e.g. "1:-;1:1;1:1,1"  (m:mu)
"""
import cmath
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_family as nf  # noqa: E402
import nonrect_supergroup as ns  # noqa: E402

try:
    from database_knotinfo import link_list
    KI = {r["name"]: r for r in link_list()[1:]}
except ImportError:
    KI = {}

THETA = 2.39996
FAM = [(int(a), () if b == "-" else tuple(int(x) for x in b.split(",")))
       for a, b in (s.split(":") for s in os.environ.get("FAMILIES", "1:-;1:1;1:1,1").split(";"))]
J = int(os.environ.get("J", "12"))
NR = int(os.environ.get("NR", "31"))


def degalex(knot):
    t = ns.data_poly(knot, (1,))
    if t is None:
        return "-"
    js = {}
    for i, j, c in t:
        js[j] = js.get(j, 0) + c
    e = [j for j, c in js.items() if c]
    return (max(e) - min(e)) // 4


def job(args):
    knot, b, word = args
    out = []
    for m, mu in FAM:
        par = [0] + [1] * (m + 1)
        r0 = max(m + 1, mu[0] if mu else 1)
        rs = list(range(r0, r0 + NR))
        qd = cmath.exp(1j * THETA)
        nf.SCAL[0] = 0.0
        vals, dd = nf.values(par, mu, word, qd, rs, knot)
        c, res, cond = nf.fit(vals, rs, qd, J)
        lo, hi, sup, syms = nf.analyse(c, qd)
        sat = "window" if (hi >= J or lo <= -J) else ""
        g = KI.get(knot, {}).get("three_genus", "-")
        out.append([knot, b, g, degalex(knot), m, "[" + ",".join(map(str, mu)) + "]", "%g" % ((hi - lo) / 4),
                    "%d..%d" % (lo, hi), ",".join(map(str, syms)) or "-", "%.1e" % res, "%.1e" % nf.SCAL[0],
                    "%.1e" % dd, sat or "-"])
    return out


def main():
    maxb = int(sys.argv[1])
    rows = ns.knot_rows()
    names = sys.argv[2:] or sorted(k for k, r in rows.items() if int(r["crossing_number"]) <= 10)
    tasks = [(k, int(rows[k]["braid_index"]), ns.braid_word(rows[k])) for k in names
             if k in rows and int(rows[k]["braid_index"]) <= maxb]
    print("# theta=%g r=r0..r0+%d J=%d; e = lam-span/4; data_check = max rel. diff vertex vs data (r with data)"
          % (THETA, NR - 1, J))
    print("knot\tbraid_index\tgenus\tdegDelta\tm\tmu\te\tlam_support\tsym_a\tfit_residual\tscalar_dev\t"
          "data_check\tflag")
    with Pool(int(os.environ.get("NPROC", "1"))) as pool:
        for rr in pool.imap(job, tasks):
            for row in rr:
                print("\t".join(map(str, row)), flush=True)


if __name__ == "__main__":
    main()
