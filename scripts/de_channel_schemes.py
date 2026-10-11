"""Obstruction of the R (x) Rbar channel DE for one knot under several allowances
for the 'end' factors End(mu) of the single channels (x-test, see de_channel_test).

    python scripts/de_channel_schemes.py torus 2 5 --N 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_channel_defect as Dd  # noqa: E402
import de_channel_test as T  # noqa: E402


def kseq(delta, j):
    """symmetric defect sequence of a regular knot: k_j = floor((j-1)/(delta+1))"""
    return (j - 1) // (delta + 1)


def defect_allow(chans, delta):
    """single mu may lose the factors of End(mu) with |k| >= k_{|mu|}"""
    return Dd.end_allow(chans, keep=lambda k, mu: abs(k) < kseq(delta, sum(mu)))


def schemes(chans, delta=None):
    out = {
        "plain": {},
        "sym": Dd.sym_allow(chans),
        "End": Dd.end_allow(chans),
    }
    # End factors with |k| >= t only (the defect eats the 'far' part of the end column)
    if delta is not None:
        out["defect"] = defect_allow(chans, delta)
    return out


def necessary(knot, prep, allow):
    """for each single channel: the allowed factors that cannot be dropped one at a time"""
    out = {}
    for c, ks in allow.items():
        need = []
        for k in sorted(set(ks)):
            trial = dict(allow)
            kk = list(ks)
            kk.remove(k)
            trial[c] = kk
            if T.run(knot, "x", prep=prep, allow=trial)["obstruction"]:
                need.append(k)
        if need:
            out[T.channel_name(c)] = need
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("args", nargs="+")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--delta", type=int)
    ap.add_argument("--necessary", action="store_true")
    a = ap.parse_args()
    knot = T.Knot(a.kind, a.args, a.N)
    prep = T.prepare(knot, "x")
    res = {"knot": knot.label, "N": knot.N}
    sch = schemes(prep["chans"], a.delta)
    for name, al in sch.items():
        res[name] = T.run(knot, "x", prep=prep, allow=al)["obstruction"]
    if a.necessary:
        best = "defect" if a.delta is not None and not res["defect"] else "End"
        res["necessary_in_" + best] = necessary(knot, prep, sch[best])
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
