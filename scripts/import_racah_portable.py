"""One-time import of the family-P ("portable", racah_homfly v0.7) inclusive
3-strand blocks into the compact format read by homfly.racah.portable.

    python scripts/import_racah_portable.py --archive <unpacked racah_matrices_upto6> \
        --reps 2 21 31 --out data/racah/portable --jobs 4

Uses the bundled racah_homfly loader (needs sympy + numpy) and converts every
entry of R1, R2, R1_inv, R2_inv to  num/den  polynomials in (A, q) with integer
coefficients.  With --kind exclusive: C, Dbar2 and the vacuum index for two-bridge knots.
Output: data/racah/portable/{inclusive,exclusive}_<R>.json.gz
"""
import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def _entry_converter(sp, A, q):
    """sympy expression -> [num terms, den terms] via the sparse fraction
    field ZZ(A, q) (much faster than cancel/together on large entries)."""
    K = sp.ZZ.frac_field(A, q)

    def terms(p):
        return [[int(i), int(j), int(c)] for (i, j), c in p.terms()]

    def entry(e):
        e = sp.sympify(e)
        try:
            f = K.from_sympy(e)
            return [terms(f.numer), terms(f.denom)]
        except sp.polys.polyerrors.HeuristicGCDFailed:
            # sympy's heuristic GCD occasionally gives up on large entries
            n, d = sp.fraction(sp.cancel(sp.together(e)))
            return [terms(sp.Poly(sp.expand(n), A, q)), terms(sp.Poly(sp.expand(d), A, q))]
    return entry


def rep_key(R):
    return "".join(map(str, R))


def convert(archive, R, out):
    sys.path.insert(0, os.path.join(archive, "software", "RACAH_HOMFLY_PORTABLE"))
    import sympy as sp
    from racah_homfly import A, q, load_matrices

    entry = _entry_converter(sp, A, q)

    t = time.time()
    recs = load_matrices(list(R), "inclusive", Avar=A, qvar=q)
    channels = []
    for r in recs:
        ch = {"Q": list(r["Q"]), "dim": int(r["dimension"]),
              "labels": [str(x) for x in r["basis_labels"]]}
        for key in ("R1", "R2", "R1_inv", "R2_inv"):
            M = r[key]
            ch[key] = [[entry(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
        channels.append(ch)
    path = os.path.join(out, "inclusive_%s.json.gz" % rep_key(R))
    with gzip.open(path, "wt") as f:
        json.dump({"R": list(R), "family": "P", "source": "racah_homfly v0.7",
                   "framing": "topological factor included in R1, R2",
                   "channels": channels}, f)
    return R, len(channels), time.time() - t


def convert_exclusive(archive, R, out):
    """C, Dbar2 (= Tbar^2, diagonal) and the vacuum index: all the two-bridge
    evaluation needs (branch-free, root-free)."""
    sys.path.insert(0, os.path.join(archive, "software", "RACAH_HOMFLY_PORTABLE"))
    import sympy as sp
    from racah_homfly import A, q
    from racah_homfly.exclusive import load_exclusive

    entry = _entry_converter(sp, A, q)

    t = time.time()
    if sum(R) == 3:
        # read only C and Tbar2 (the loader also sympifies S̄, T̄ with roots: slow)
        from racah_homfly.exclusive import _three_data
        from racah_homfly.core import rep_label
        ex = _three_data()["representations"][rep_label(tuple(R))]["exclusive"]
        K = sp.ZZ.frac_field(A, q)
        loc = {"A": A, "q": q}
        C = sp.Matrix([[K.to_sympy(K.from_sympy(sp.sympify(x, locals=loc))) for x in row]
                       for row in ex["C"]["entries"]])
        D = sp.Matrix([[sp.sympify(x, locals=loc) for x in row] for row in ex["Tbar2"]["entries"]])
        m = {"vacuum_index": 0, "framing": "threebox", "effective_q": "q"}
    else:
        m = load_exclusive(list(R), A, q, full=False)
        C, D = m["C"], m["Dbar2"]
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            assert i == j or D[i, j] == 0, "Dbar2 not diagonal"
    d = {"R": list(R), "family": "P", "source": "racah_homfly v0.7",
         "vacuum_index": int(m["vacuum_index"]),
         "C": [[entry(C[i, j]) for j in range(C.shape[1])] for i in range(C.shape[0])],
         "Dbar2": [entry(D[i, i]) for i in range(D.shape[0])],
         "framing": str(m.get("framing", "")), "effective_q": str(m.get("effective_q", "q"))}
    path = os.path.join(out, "exclusive_%s.json.gz" % rep_key(R))
    with gzip.open(path, "wt") as f:
        json.dump(d, f)
    return R, C.shape[0], time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--reps", nargs="+", required=True, help="e.g. 1 2 11 21 311")
    ap.add_argument("--out", default="data/racah/portable")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--kind", choices=["inclusive", "exclusive"], default="inclusive")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reps = [tuple(int(c) for c in s) for s in a.reps]
    with ProcessPoolExecutor(a.jobs) as ex:
        fn = convert if a.kind == "inclusive" else convert_exclusive
        for R, n, sec in ex.map(fn, [a.archive] * len(reps), reps, [a.out] * len(reps)):
            print("%s R=%s  size %d  %.1fs" % (a.kind, list(R), n, sec), flush=True)


if __name__ == "__main__":
    main()
