"""Command line interface.

    homfly compute 3_1                      # fundamental
    homfly compute 4_1 --rep 2              # [2]
    homfly compute --torus 3,4 --rep 2,1    # Rosso--Jones
    homfly compute --braid 1,1,-2,1,-2 --rep 1,1
    homfly compute --double-braid 2,-3 --rep 4,2,1   # interpolation formula, any R
    homfly info 10_124
"""
from __future__ import annotations

import argparse
import sys
import time

from .compute import homfly
from .io.export import to_mathematica
from .knots.families import TorusKnot
from .knots.table import knot as table_knot
from .reconstruction.pipeline import ReconstructionReport


def _ints(s):
    return tuple(int(x) for x in s.replace("[", "").replace("]", "").split(",") if x)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="homfly")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("name", nargs="?")
    c.add_argument("--rep", default="1")
    c.add_argument("--torus")
    c.add_argument("--braid")
    c.add_argument("--double-braid", help="m,n: antiparallel double braid H(m,n)")
    c.add_argument("--method")
    c.add_argument("--format", choices=["plain", "mathematica"], default="plain")
    i = sub.add_parser("info")
    i.add_argument("name")
    a = ap.parse_args(argv)

    if a.cmd == "info":
        r = table_knot(a.name)
        for k, v in r.__dict__.items():
            if k != "pd":
                print("%-16s %s" % (k, v))
        print("%-16s %s" % ("H_[1] (A,q)", r.homfly_reference()))
        return 0

    if a.torus:
        K = TorusKnot(*_ints(a.torus))
    elif a.double_braid:
        from .knots.families import DoubleBraid
        K = DoubleBraid(*_ints(a.double_braid))
    elif a.braid:
        K = _ints(a.braid)
    else:
        K = a.name
    rep = ReconstructionReport()
    t = time.time()
    H = homfly(K, R=_ints(a.rep), method=a.method, report=rep)
    print(to_mathematica(H) if a.format == "mathematica" else H)
    print("# method=%s primes=%d evaluations=%d A=%s q=%s verified=%s %.2fs"
          % (rep.method, len(rep.primes), rep.evaluations, rep.A_range, rep.q_range,
             rep.verified, time.time() - t), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
