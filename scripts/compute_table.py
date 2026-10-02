"""Compute reduced colored HOMFLY for table knots and all R up to a size,
one text file per knot:

    python scripts/compute_table.py --crossings 7 --max-size 6 --out data/homfly --jobs 3

Each (knot, R) runs as a separate task (method chosen automatically by
homfly.compute), cached in <out>/.cache so interrupted runs resume.  Every
polynomial is checked structurally before it is written:
  * q = 1:  H_R(A, 1) = H_[1](A, 1)^|R|  (special polynomial),
  * transposition: H_{R^T}(A, q) = H_R(A, -1/q) for every computed pair.

``--knots 8_15 --method montesinos --partial`` restricts the run to named knots
and also writes knots for which some representations are not computable
(listed in the file header).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from homfly.compute import homfly  # noqa: E402
from homfly.conventions import transpose_rep  # noqa: E402
from homfly.io.export import from_json, to_json  # noqa: E402
from homfly.knots.table import knots  # noqa: E402
from homfly.reconstruction.pipeline import ReconstructionReport  # noqa: E402
from homfly.reps.partitions import conjugate, partitions  # noqa: E402


def rep_str(R):
    return "[" + ",".join(map(str, R)) + "]"


def task(name, R, cache, method=None, exclude=(), cache_only=False):
    path = os.path.join(cache, "%s_%s.json" % (name, "".join(map(str, R))))
    if os.path.exists(path):
        return json.load(open(path))
    if cache_only:
        raise LookupError("not in cache")
    t = time.time()
    rep = ReconstructionReport()
    try:
        H = homfly(name, R=R, report=rep, method=method, exclude=exclude)
    except Exception:
        if method is None:
            raise
        H = homfly(name, R=R, report=rep, exclude=exclude)       # preferred method not applicable
    res = {"knot": name, "R": list(R), "method": rep.method, "seconds": round(time.time() - t, 1),
           "primes": len(rep.primes), "evaluations": rep.evaluations, "verified": rep.verified,
           "poly": to_json(H)}
    json.dump(res, open(path, "w"))
    return res


def q1(H):
    from homfly.checks.structural import specialise_q1
    return specialise_q1(H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossings", type=int, nargs="+", default=[7])
    ap.add_argument("--max-size", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "homfly"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--method", help="preferred method, e.g. two-bridge (fallback: automatic)")
    ap.add_argument("--knots", nargs="+", help="only these knots (default: all with --crossings)")
    ap.add_argument("--partial", action="store_true",
                    help="also write knots with some representations missing")
    ap.add_argument("--knots-file", help="file with knot names (one per line), instead of --crossings")
    ap.add_argument("--exclude-methods", nargs="+", default=[],
                    help="never use these methods (e.g. cabling-paths racah-3strand-U cabling)")
    ap.add_argument("--cache-only", action="store_true",
                    help="compute nothing: write the files from the cache")
    ap.add_argument("--derive-transposed", action="store_true",
                    help="compute only one of R, R^T; the other by H_{R^T}(A,q) = H_R(A,-1/q) "
                         "(unless it is already in the cache)")
    a = ap.parse_args()
    cache = os.path.join(a.out, ".cache")
    os.makedirs(cache, exist_ok=True)
    if a.knots_file:
        names = [x.strip() for x in open(a.knots_file) if x.strip()]
    else:
        names = a.knots or [k.name for k in knots(max(a.crossings)) if k.crossings in a.crossings]
    reps = [R for k in range(1, a.max_size + 1) for R in partitions(k)]
    # knot by knot (files are written as soon as a knot is finished), big R first inside a knot
    def wanted(n, R):
        if not a.derive_transposed or R >= conjugate(R):
            return True
        return os.path.exists(os.path.join(cache, "%s_%s.json" % (n, "".join(map(str, R)))))
    tasks = [(n, R) for n in names for R in sorted(reps, key=lambda R: -sum(R)) if wanted(n, R)]
    print("%d knots x %d representations = %d tasks" % (len(names), len(reps), len(tasks)), flush=True)
    results, failed = {}, []
    from collections import Counter
    pending = Counter(m for m, _ in tasks)
    written = [0]

    def finish(n):
        for R in reps:          # derived transposes
            RT = conjugate(R)
            if (n, R) not in results and (n, RT) in results:
                v = dict(results[(n, RT)])
                v["poly"] = to_json(transpose_rep(from_json(v["poly"])))
                v["method"] = v["method"] + "+transposed"
                v["derived"] = True
                results[(n, R)] = v
        if write_knot(n, a, reps, results):
            written[0] += 1
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(task, n, R, cache, a.method, tuple(a.exclude_methods), a.cache_only): (n, R)
                for n, R in tasks}
        for f in as_completed(futs):
            n, R = futs[f]
            pending[n] -= 1
            try:
                r = f.result()
            except Exception as e:  # report, keep going
                failed.append((n, R, repr(e)))
                if not a.cache_only:
                    print("FAILED %s %s: %r" % (n, rep_str(R), e), flush=True)
            else:
                results[(n, tuple(R))] = r
                if not a.cache_only:
                    print("%-5s %-14s %-24s %7.1fs" % (n, rep_str(R), r["method"], r["seconds"]), flush=True)
            if pending[n] == 0:
                finish(n)
    print("SUMMARY: %d of %d knots written, %d tasks failed" % (written[0], len(names), len(failed)), flush=True)


def write_knot(n, a, reps, results):
    rows = {R: results.get((n, R)) for R in reps}
    missing = [R for R, v in rows.items() if v is None]
    if (missing and not a.partial) or (1,) in missing:
        return False
    polys = {R: from_json(v["poly"]) for R, v in rows.items() if v is not None}
    path = os.path.join(a.out, "%s.txt" % n)
    if os.path.exists(path):         # never replace a file by one with fewer representations
        with open(path) as f:
            if sum(line.startswith("R = ") for line in f) > len(polys):
                return False
    H1q = q1(polys[(1,)])
    checks = {}
    for R, H in polys.items():
        sp = q1(H) == H1q ** sum(R)
        RT = conjugate(R)
        if rows[R].get("derived") or (rows.get(RT) or {}).get("derived"):
            tr = "derived"
        else:
            tr = polys[RT] == transpose_rep(H) if RT in polys else None
        checks[R] = (sp, tr)
    with open(path, "w") as f:
        f.write("# Reduced colored HOMFLY polynomials H_R(A, q) of knot %s\n" % n)
        f.write("# Convention: A^{-1}H(L+) - A H(L-) = (q - 1/q) H(L0); H_R(unknot) = 1;\n")
        f.write("#   A = v, z = q - 1/q relative to KnotInfo; topological framing (docs/CONVENTIONS.md).\n")
        f.write("# All R with |R| <= %d.  Checks per R: q=1 -> H_[1]^|R|, transposition H_{R^T}(A,q) = H_R(A,-1/q)\n" % a.max_size)
        f.write("#   (transposition_check=derived: one of R, R^T was obtained from the other by this identity).\n")
        if missing:
            f.write("# Not computed: %s\n" % ", ".join(map(rep_str, missing)))
        f.write("# Generated by scripts/compute_table.py\n\n")
        for R in reps:
            v = rows[R]
            if v is None:
                continue
            sp, tr = checks[R]
            H = polys[R]
            f.write("R = %s\n" % rep_str(R))
            f.write("# method=%s primes=%d evaluations=%d verified=%s q1_check=%s transposition_check=%s terms=%d\n"
                    % (v["method"], v["primes"], v["evaluations"], v["verified"], sp, tr, len(H.terms)))
            f.write("H = %s\n\n" % H.to_string(mul="*", pow_="^"))
    bad = [rep_str(R) for R, c in checks.items() if False in c]
    print("wrote %s.txt  (%d of %d representations; failed checks: %s)"
          % (n, len(polys), len(reps), bad or "none"), flush=True)
    return True


if __name__ == "__main__":
    main()
