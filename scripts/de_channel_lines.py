"""How strong is the R (x) Rbar channel DE?  Compare, line by line, its conditions
with the linear identities that hold for ALL knots.

On the line A = q^{-k} (x = A^2 = q0^{-2k}, q = q0 fixed) the DE with end factors
removed,  H_R - 1 = sum_c F'_c(R) G_c,  F'_mu = F_mu / End(mu),  gives
    c_k = n - rank F'(x_k)
first-order linear conditions on the values (H_R - 1)(A = q^{-k}).  The universal
identities on the same line are measured on the knot table: u_k = n - rank of the
matrix of (H_R^K - 1)(A = q^{-k}) over many knots K.  Since every knot satisfies
the DE, the DE conditions are among the universal ones; c_k = u_k means that on this
line the DE says nothing beyond the universal (Berezinian) reductions.
The total length of the quotient module, sum_k ord_{D_k} det, is compared with sum_k c_k
(equality = no higher-order conditions).

    python scripts/de_channel_lines.py --N 5 --knots 120
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import flint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_channel_defect as Dd  # noqa: E402
import de_channel_test as T  # noqa: E402

P = T.P_MOD


def rank_mod(rows):
    if not rows or not rows[0]:
        return 0
    return flint.nmod_mat(rows, P).rank()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--knots", type=int, default=120)
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--no-end", action="store_true")
    a = ap.parse_args()
    base = T.Knot("db", ["1", "-1"], a.N)
    prep = T.prepare(base, "x")
    q0 = prep["fixed"]
    reps, chans, polys = base.reps, prep["chans"], prep["polys"]
    # F' = F / End (exact column division)
    Fp = {}
    for c in chans:
        e = T.ONE
        if c[0] == "s" and not a.no_end:
            for k in Dd.end_factor(c[1]):
                e = e * T.D_poly(k, "x", q0)
        for R in reps:
            if (c, R) in polys:
                qt, rm = divmod(polys[(c, R)], e)
                assert rm.is_zero(), (c, R)
                Fp[(c, R)] = qt
    M = [[Fp.get((c, R), T.ZERO) for c in chans] for R in reps]
    E, piv = T.echelon(M)
    det = T.ONE
    for i, pc in enumerate(piv):
        if pc is not None:
            det = det * E[i][pc]
    # knots with all reps
    names = []
    for f in sorted(os.listdir(T.ROOT / "data" / "homfly")):
        s = (T.ROOT / "data" / "homfly" / f).read_text()
        if all(("R = " + json.dumps(list(R)).replace(" ", "")) in s.replace(", ", ",") for R in reps):
            names.append(f[:-4])
    random.Random(0).shuffle(names)
    names = names[: a.knots]
    tables = {n: T.load_table_file(n) for n in names}
    W = flint.nmod_poly([P - 1, 1], P) ** 2
    out = []
    for k in range(-a.K, a.K + 1):
        if k == 0:
            continue
        xk = pow(int(q0), (-2 * k) % (P - 1), P)
        Ak = T.FLD(int(q0)) ** (-k)
        Fk = [[int(M[i][j](xk)) for j in range(len(chans))] for i in range(len(reps))]
        ck = len(reps) - rank_mod(Fk)
        V = [[int(tables[n][R].evaluate([Ak, T.FLD(int(q0))], one=T.FLD.one) - 1) for n in names] for R in reps]
        uk = len(reps) - rank_mod(V)
        # do the DE conditions annihilate all knots?  left null space of Fk
        Fk_mat = flint.nmod_mat(Fk, P)
        ns, nd = Fk_mat.transpose().nullspace()
        ok = True
        if nd:
            Nmat = flint.nmod_mat([[int(ns[i, j]) for i in range(len(reps))] for j in range(nd)], P)
            prod = Nmat * flint.nmod_mat(V, P)
            ok = all(int(prod[i, j]) == 0 for i in range(prod.nrows()) for j in range(prod.ncols()))
        # order of D_k in det (length of the quotient on this line)
        Dk = T.D_poly(k, "x", q0)
        ordk, d = 0, det
        while True:
            qt, rm = divmod(d, Dk)
            if not rm.is_zero():
                break
            d, ordk = qt, ordk + 1
        out.append({"k": k, "DE_first_order": ck, "DE_length": ordk, "universal": uk, "knots_satisfy_DE": ok})
        print(json.dumps(out[-1]), flush=True)
    print(json.dumps({"N": a.N, "knots": len(names), "end_removed": not a.no_end}))


if __name__ == "__main__":
    main()
