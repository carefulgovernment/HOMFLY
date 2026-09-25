"""Check the fundamental HOMFLY of every knot up to N crossings against
KnotInfo at a random GF(p) point (seconds for the full 12-crossing table)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from homfly.algebra.fields import GF  # noqa: E402
from homfly.knots.table import knots  # noqa: E402
from homfly.methods import HeckeFundamental  # noqa: E402

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    F = GF(2 ** 61 - 1)
    rng = random.Random(0)
    A, q = F.random_element(rng), F.random_element(rng)
    h = HeckeFundamental()
    bad = 0
    total = 0
    for rec in knots(N):
        if rec.braid is None:
            continue
        total += 1
        if h.evaluate(rec.braid, (1,), F, A, q) != rec.homfly_reference().evaluate([A, q], one=F.one):
            bad += 1
            print("MISMATCH", rec.name)
    print("%d knots checked, %d mismatches" % (total, bad))
