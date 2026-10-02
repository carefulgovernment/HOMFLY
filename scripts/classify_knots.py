"""Split table knots by the cheapest available description, for batch planning:

    python scripts/classify_knots.py --crossings 10 11 12 --out /tmp/classes

writes <out>/<class>.txt (one knot name per line) and prints the counts.
Classes, cheapest first:
  torus       Rosso-Jones
  two_bridge  exclusive Racah data (all R)
  montesinos  Montesinos tangle calculus (all R), >= 3 tangles, chirality decided
  algebraic   arborescent tangle calculus from the Conway notation (all R)
  braid3      3-strand Racah data (all R; the non-rectangular 6-box ones are slow)
  other       polyhedral and undecided (cabling-paths only, small R)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from homfly.compute import TABLE_TORUS, algebraic_presentation  # noqa: E402
from homfly.knots.families import MontesinosKnot  # noqa: E402
from homfly.knots.table import knots  # noqa: E402
from homfly.methods import montesinos as MO  # noqa: E402


def classify(k):
    if k.name in TABLE_TORUS:
        return "torus"
    if k.is_two_bridge:
        return "two_bridge"
    if k.is_montesinos and k.montesinos.count(";") >= 2:
        fr = MontesinosKnot.from_notation(k.montesinos).fractions()
        if MO.chirality(fr, k.homfly_reference(), k.braid) is not None:
            return "montesinos"
    if algebraic_presentation(k.name) is not None:
        return "algebraic"
    if k.braid is not None and k.braid.strands <= 3:
        return "braid3"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossings", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups = {}
    for k in knots(max(a.crossings)):
        if k.crossings in a.crossings:
            groups.setdefault(classify(k), []).append(k.name)
    for c in ("torus", "two_bridge", "montesinos", "algebraic", "braid3", "other"):
        names = groups.get(c, [])
        with open(os.path.join(a.out, c + ".txt"), "w") as f:
            f.write("".join(n + "\n" for n in names))
        print("%-11s %5d" % (c, len(names)))


if __name__ == "__main__":
    main()
