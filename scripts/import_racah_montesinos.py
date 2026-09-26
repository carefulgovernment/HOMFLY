"""Export the matrices needed by the Montesinos (tangle-sum) method.

Family P (racah_homfly v0.7), multiplicity-free R (rectangular):
    S̄ (antiparallel H <-> V), S (antiparallel <- u-channel parallel), V = S^-1,
    T (parallel eigenvalues), T̄ (antiparallel eigenvalues)  -- all diagonal
    eigenvalue matrices checked diagonal.

    python scripts/import_racah_montesinos.py --archive <racah_matrices_upto6> \
        --reps 1 2 11 3 111 4 22 1111 5 11111 6 111111 --out data/racah/montesinos

Output: <out>/P_<R>.json.gz  with entries [num, den] term lists [[i, j, c]].
"""
import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from import_racah_portable import _entry_converter, rep_key  # noqa: E402


def export_P(archive, R, out):
    sys.path.insert(0, os.path.join(archive, "software", "RACAH_HOMFLY_PORTABLE"))
    import sympy as sp
    from racah_homfly import A, q
    from racah_homfly.exclusive import load_exclusive
    entry = _entry_converter(sp, A, q)
    t = time.time()
    m = load_exclusive(list(R), A, q, full=True)
    res = {"R": list(R), "family": "P", "vacuum_index": int(m["vacuum_index"])}
    for k in ("Sbar", "S", "V"):
        M = m[k]
        res[k] = [[entry(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
    for k in ("T", "Tbar"):
        M = m[k]
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i != j and M[i, j] != 0:
                    raise ValueError("%s not diagonal for %s (multiplicities?)" % (k, R))
        res[k] = [entry(M[i, i]) for i in range(M.shape[0])]
    path = os.path.join(out, "P_%s.json.gz" % rep_key(R))
    with gzip.open(path, "wt") as f:
        json.dump(res, f)
    return R, len(res["Sbar"]), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--reps", nargs="+", required=True)
    ap.add_argument("--out", default="data/racah/montesinos")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reps = [tuple(int(c) for c in s) for s in a.reps]
    with ProcessPoolExecutor(a.jobs) as ex:
        for R, n, sec in ex.map(export_P, [a.archive] * len(reps), reps, [a.out] * len(reps)):
            print("P R=%s size %d %.1fs" % (list(R), n, sec), flush=True)


if __name__ == "__main__":
    main()
