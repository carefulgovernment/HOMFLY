"""Batch computation of colored HOMFLY for the knot table into a SQLite DB.

    python scripts/tabulate.py --rep 1 --max-crossings 10 --db results.sqlite
    python scripts/tabulate.py --rep 2 --max-crossings 8 --jobs 8

Skips entries already present.  Each knot runs in a worker process; the method
is chosen automatically (compute.choose_method) unless --method is given.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from homfly.compute import homfly  # noqa: E402
from homfly.io.database import ResultDB  # noqa: E402
from homfly.knots.table import knots  # noqa: E402
from homfly.reconstruction.pipeline import ReconstructionReport  # noqa: E402


def work(name, rep, method):
    t = time.time()
    r = ReconstructionReport()
    H = homfly(name, R=rep, method=method, report=r)
    return name, H, r.method, time.time() - t, r.verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="1")
    ap.add_argument("--max-crossings", type=int, default=10)
    ap.add_argument("--min-crossings", type=int, default=3)
    ap.add_argument("--max-strands", type=int, default=None)
    ap.add_argument("--method")
    ap.add_argument("--db", default="results.sqlite")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    rep = tuple(int(x) for x in a.rep.split(","))
    db = ResultDB(a.db)
    todo = [k.name for k in knots(a.max_crossings)
            if k.crossings >= a.min_crossings and k.braid is not None
            and (a.max_strands is None or k.braid.strands <= a.max_strands)
            and not db.has(k.name, rep)]
    print("%d knots to compute in %s" % (len(todo), rep))
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(work, n, rep, a.method) for n in todo]
        for f in as_completed(futs):
            try:
                name, H, method, sec, ok = f.result()
            except Exception as e:  # keep going; report failures
                print("FAILED:", e)
                continue
            db.put(name, rep, method, H, sec, ok)
            print("%-10s %-18s %6.2fs %4d terms" % (name, method, sec, len(H.terms)))


if __name__ == "__main__":
    main()
