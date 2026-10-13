"""Family-P exclusive Racah data for symmetric [r] from the root-free q-6j formulas
of racah_homfly (software/RACAH_HOMFLY_PORTABLE of the Racah release):

    python scripts/make_symmetric_racah.py <path to RACAH_HOMFLY_PORTABLE> 7 12

writes data/racah/montesinos/P_<key>.json.gz (key = rep_key((r,)), e.g. "7", "x10").
racah_homfly.rational_sym.rational_exclusive((r,), framing="four_plat") with q -> -q
reproduces the stored P_[r], r <= 6, exactly (S, V, Sbar, T, Tbar); for r = 7..9 the
resulting H_[r] agree with independent computations (pointwise mixed Racah engine,
cabling) for 17 Montesinos / arborescent knots.
"""
import gzip
import json
import os
import sys
import time

import sympy as sp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from homfly.racah.portable import rep_key  # noqa: E402


def main():
    sys.path.insert(0, sys.argv[1])
    from racah_homfly.rational_sym import rational_exclusive
    from racah_homfly.core import A, q

    def enc(e):
        n, d = sp.fraction(sp.cancel(sp.together(e)))

        def pl(x):
            P = sp.Poly(sp.expand(x), A, q)
            return [[int(m[0]), int(m[1]), int(c)] for m, c in zip(P.monoms(), P.coeffs())]
        return [pl(n), pl(d)]
    for r in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
        t = time.time()
        d = rational_exclusive((r,), framing="four_plat", cancel_entries=False)
        out = {"R": [r], "family": "P", "vacuum_index": d["vacuum_index"]}
        for k in ("Sbar", "S", "V"):
            M = d[k].subs(q, -q)
            out[k] = [[enc(M[i, j]) for j in range(r + 1)] for i in range(r + 1)]
        for k in ("T", "Tbar"):
            out[k] = [enc(d[k][i, i].subs(q, -q)) for i in range(r + 1)]
        path = os.path.join(ROOT, "data", "racah", "montesinos", "P_%s.json.gz" % rep_key((r,)))
        json.dump(out, gzip.open(path, "wt"))
        print("[%d] -> %s (%.0fs)" % (r, path, time.time() - t), flush=True)


if __name__ == "__main__":
    main()
