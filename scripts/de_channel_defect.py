"""Which channels of R (x) Rbar need 'defect' denominators for a given knot?

Formula II's factors F_c(R) are normalised for defect-zero knots: on the
symmetric channels they are the Z of the defect-zero DE, i.e. they contain the
'end column' prod_{i=1}^{j-1} {A q^{i-1}} which a knot with defect > 0 does not
supply.  Here we allow G_c the denominators
    [j]:   D_0 D_1 ... D_{j-2}          [1^j]:  D_0 D_-1 ... D_-(j-2)
(the symmetric DE in its G-normalisation, eq. (2) of the paper), and then look
for the denominators the NON-RECTANGULAR channels need (x-test, q = q0 fixed).

    python scripts/de_channel_defect.py torus 2 5 --N 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_channel_test as T  # noqa: E402


def is_rect(la):
    return len(set(la)) == 1


def sym_allow(chans):
    allow = {}
    for c in chans:
        if c[0] != "s":
            continue
        mu = c[1]
        if len(mu) == 1:
            allow[c] = list(range(0, mu[0] - 1))
        elif max(mu) == 1:
            allow[c] = [-i for i in range(0, len(mu) - 1)]
    return allow


# End(mu): the R-independent factor of F_mu(R), measured from F_mu(mu) of Formula II
# (prod of D_k, multiset of k), |mu| <= 6.  F_mu(R) = (q-factor) * prod_{b in row 1 of mu} D_{R_1 + c(b)}
#   * prod_{b in column 1 of mu} D_{c(b) - l(R)} * End(mu)   (checked for R = mu and mu + box, |mu| <= 4).
# End(mu^T) = -End(mu).
END_TABLE = {
    (1,): [], (2,): [0], (3,): [0, 1], (2, 1): [-1, 1], (4,): [0, 1, 2], (3, 1): [-1, 0, 2],
    (2, 2): [-2, -1, 1, 2], (5,): [0, 1, 2, 3], (4, 1): [-1, 0, 1, 3], (3, 2): [-2, -1, 0, 1, 3],
    (3, 1, 1): [-2, -1, 1, 2], (6,): [0, 1, 2, 3, 4], (5, 1): [-1, 0, 1, 2, 4],
    (4, 2): [-2, -1, 0, 1, 1, 4], (4, 1, 1): [-2, -1, 0, 2, 3], (3, 3): [-2, -1, -1, 0, 2, 3, 4],
    (3, 2, 1): [-3, -1, -1, 1, 1, 3],
}


def end_factor(mu):
    """R-independent factor End(mu) of F_mu(R) as a list of k in D_k (table for |mu| <= 6;
    otherwise the product rule: boxes outside the first row give D_{mu_1 + c}, outside the
    first column D_{c - l(mu)}, which is exact for hooks and rectangles)."""
    mu = tuple(mu)
    if mu in END_TABLE:
        return list(END_TABLE[mu])
    mt = tuple(sum(1 for r in mu if r > j) for j in range(mu[0]))
    if mt in END_TABLE:
        return sorted(-k for k in END_TABLE[mt])
    out = []
    for i, row in enumerate(mu):
        for j in range(row):
            c = j - i
            if i > 0:
                out.append(mu[0] + c)
            if j > 0:
                out.append(c - len(mu))
    return sorted(out)


def end_allow(chans, keep=lambda k, mu: False):
    """Allow every single channel mu the denominator End(mu) (minus factors with keep(k, mu))."""
    return {c: [k for k in end_factor(c[1]) if not keep(k, c[1])] for c in chans if c[0] == "s"}


def ks_of(obs):
    out = []
    for name, e in obs:
        if not name.startswith("D_"):
            return None
        out += [int(name[2:])] * e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("args", nargs="+")
    ap.add_argument("--N", type=int, default=5)
    a = ap.parse_args()
    knot = T.Knot(a.kind, a.args, a.N)
    prep = T.prepare(knot, "x")
    chans = prep["chans"]
    base = T.run(knot, "x", prep=prep)
    allow = sym_allow(chans)
    r1 = T.run(knot, "x", prep=prep, allow=allow)
    out = {"knot": knot.label, "N": knot.N, "plain": base["obstruction"], "after_sym": r1["obstruction"]}
    rest = ks_of(r1["obstruction"])
    if rest:
        # which single non-symmetric channel can absorb the rest?
        cands = [c for c in chans if c not in allow]
        single = {}
        for c in cands:
            al = dict(allow)
            al[c] = rest
            r = T.run(knot, "x", prep=prep, allow=al)
            single[T.channel_name(c)] = r["obstruction"]
        out["one_channel_absorbs"] = [k for k, v in single.items() if not v]
        # per-level: allow all non-symmetric channels the rest, then drop channels greedily
        al = dict(allow)
        for c in cands:
            al[c] = rest
        need = {}
        for c in sorted(cands, key=lambda c: -T.level_of(c) if hasattr(T, "level_of") else 0):
            trial = dict(al)
            trial[c] = []
            if not T.run(knot, "x", prep=prep, allow=trial)["obstruction"]:
                al = trial
        # minimise each surviving channel factor by factor
        for c in cands:
            if not al.get(c):
                continue
            ks = list(al[c])
            for k in list(ks):
                trial = dict(al)
                kk = list(trial[c])
                kk.remove(k)
                trial[c] = kk
                if not T.run(knot, "x", prep=prep, allow=trial)["obstruction"]:
                    al = trial
            if al[c]:
                need[T.channel_name(c)] = sorted(al[c])
        out["greedy_nonrect_denominators"] = need
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
