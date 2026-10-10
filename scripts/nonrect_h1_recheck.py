"""Re-evaluate every (algebra, R, knot) flagged as a mismatch in data/nonrect_h1_check.tsv at two other
unit-modulus q.  A flagged case is a float-precision artefact when the vertex value and the data agree to
< 1e-7 at some other q; the scalar deviation (non-scalarity of the (1,1)-tangle, zero in exact arithmetic) is
printed as an error indicator.

usage: python3 scripts/nonrect_h1_recheck.py data/nonrect_h1_check.tsv > data/nonrect_h1_recheck.tsv
"""
import cmath
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_supergroup as ns  # noqa: E402

THETAS = (1.9, 0.6847)


def main():
    rows = ns.knot_rows()
    print("algebra\tR\tknot\ttheta\trel_diff\tscalar_dev\tverdict")
    for line in open(sys.argv[1]):
        f = line.rstrip("\n").split("\t")
        if line.startswith("#") or f[0] == "parities" or f[-1] == "-":
            continue
        par = [int(c) for c in f[0]]
        N, M = par.count(0), par.count(1)
        R = tuple(int(x) for x in f[3].strip("[]").split(","))
        mods = {t: ns.Module(par, R, cmath.exp(1j * t)) for t in THETAS}
        for kn in f[-1].split(","):
            best = None
            for t, mod in mods.items():
                q = cmath.exp(1j * t)
                v, dv = mod.knot_value(ns.braid_word(rows[kn]), check_scalar=True)
                h = ns.data_value(kn, R, q ** (M - N), 1 / q)
                rel = abs(v - h) / max(1, abs(h))
                best = rel if best is None else min(best, rel)
                print("%s\t%s\t%s\t%g\t%.1e\t%.1e\t" % (f[1], f[3], kn, t, rel, dv / max(1, abs(v))), flush=True)
            print("%s\t%s\t%s\t-\t%.1e\t-\t%s" % (f[1], f[3], kn, best, "agree" if best < 1e-7 else "MISMATCH"),
                  flush=True)


if __name__ == "__main__":
    main()
