"""Is the non-rectangular DE of Formula II valid for a given knot?

Test of   H_R - 1 = sum_{c in channels(R)} F_c(R) G_c ,   all R with |R| <= N,
with the knot-independent factors F_c(R) of methods.interpolation (the channel
form of the double-braid DE, channels c = singles [mu,mu] and pairs {Z,Z'} of
R (x) Rbar) and unknown R-independent knot functions G_c.  Pointwise the system
is always solvable (more channels than diagrams), so the content of the DE is
that the G_c can be chosen LAURENT POLYNOMIALS.

Exact two-variable module membership is replaced by two univariate tests over
GF(p) (PIDs), which are necessary conditions:

  x-test:  q = q0 random, entries are Laurent polynomials in x = A^2;
  y-test:  A = A0 random, entries are Laurent polynomials in y = q^2.

For each test the matrix F is brought to lower-triangular column echelon form by
unimodular column operations, the system is solved by forward substitution over
GF(p)(t), and the OBSTRUCTION is the lcm of the denominators of the solution: the
smallest d(t) such that d * (H - 1) is in the image of F, i.e. the denominators
that the G_c must be allowed to have.  Its factors are identified as
differentials D_k = {A q^k} (roots x q^{2k} = 1, or y^k = A0^{-2}) and
cyclotomic factors Phi_n(q^2) (quantum numbers).

Usage:
  python scripts/de_channel_test.py torus 2 5 --N 5
  python scripts/de_channel_test.py table 8_19 --N 5
  python scripts/de_channel_test.py db 2 -1 --N 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import flint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homfly.algebra.fields import GF  # noqa: E402
from homfly.algebra.laurent import parse_laurent  # noqa: E402
from homfly.knots.families import DoubleBraid, TorusKnot  # noqa: E402
from homfly.methods.interpolation import Point, channels_of, paper_value  # noqa: E402
from homfly.methods.rosso_jones import RossoJones  # noqa: E402
from homfly.reps.partitions import partitions  # noqa: E402

P_MOD = (1 << 61) - 1
FLD = GF(P_MOD)


def nm(v):
    return flint.nmod(int(v), P_MOD)


# ---------------------------------------------------------------------------
# knots as point black boxes
# ---------------------------------------------------------------------------

def load_table_file(name):
    path = ROOT / "data" / "homfly" / f"{name}.txt"
    out, R = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("R = "):
            R = tuple(json.loads(line[4:]))
        elif line.startswith("H = ") and R is not None:
            out[R] = parse_laurent(line[4:])
            R = None
    return out


class Knot:
    def __init__(self, kind, args, N):
        self.kind, self.args = kind, args
        reps = [tuple(R) for k in range(1, N + 1) for R in partitions(k)]
        if kind == "table":
            self.table = load_table_file(args[0])
            reps = [R for R in reps if R in self.table]
        self.reps = reps
        self.N = max(sum(R) for R in reps)
        self.label = {"torus": "T[%s,%s]", "db": "H(%s,%s)", "table": "%s"}[kind] % tuple(args)

    def H(self, R, A, q):
        if self.kind == "torus":
            return RossoJones().evaluate(TorusKnot(*map(int, self.args)), R, FLD, A, q)
        if self.kind == "db":
            return paper_value(R, int(self.args[0]), int(self.args[1]), FLD, A, q)
        return self.table[R].evaluate([A, q], one=FLD.one)


# ---------------------------------------------------------------------------
# sampling and univariate interpolation
# ---------------------------------------------------------------------------

def sample(knot, mode, fixed, npts, rng):
    """Values of F_c(R) and H_R - 1 at npts points; returns (ts, cols, rows)."""
    entries = {}
    ts = []
    for _ in range(npts):
        z = FLD.random_element(rng)
        A, q = (z, fixed) if mode == "x" else (fixed, z)
        pt = Point(FLD, A, q)
        t = z * z
        ts.append(int(t))
        # F_c(R) have denominators [2], [3], ... and {A} (from |R| = 5 on); multiply
        # every equation by W = (A^2-1)^2 prod_{k=2}^{N} [k]-type factors (does not change the solutions)
        q2, A2 = q * q, A * A
        W = (A2 - 1) ** 2
        for k in range(2, knot.N + 1):
            W = W * (q2 ** k - 1) / (q2 - 1)
        for R in knot.reps:
            entries.setdefault(("H", R), []).append(int(W * (knot.H(R, A, q) - 1)))
            for c in channels_of(R):
                entries.setdefault((c, R), []).append(int(W * pt.F_c(c, R)))
    return ts, entries


def interpolate_all(ts, entries, shift):
    """Laurent polynomials f with t^shift f of degree < n-4, verified on 4 points."""
    n = len(ts) - 4
    V = flint.nmod_mat([[pow(t, j, P_MOD) for j in range(n)] for t in ts[:n]], P_MOD)
    Vi = V.inv()
    keys = list(entries)
    Y = flint.nmod_mat([[entries[k][i] * pow(ts[i], shift, P_MOD) % P_MOD for k in keys] for i in range(n)], P_MOD)
    C = Vi * Y
    out = {}
    bad = []
    for j, k in enumerate(keys):
        poly = flint.nmod_poly([int(C[i, j]) for i in range(n)], P_MOD)
        ok = all(int(poly(ts[i])) == entries[k][i] * pow(ts[i], shift, P_MOD) % P_MOD for i in range(n, n + 4))
        if not ok:
            bad.append(k)
        out[k] = poly
    return out, bad


# ---------------------------------------------------------------------------
# column echelon form over GF(p)[t] and forward substitution over GF(p)(t)
# ---------------------------------------------------------------------------

ZERO = flint.nmod_poly([], P_MOD)
ONE = flint.nmod_poly([1], P_MOD)


def strip_t(f):
    """Remove the factor t^k (t is a unit of the Laurent ring)."""
    c = f.coeffs()
    k = 0
    while k < len(c) and int(c[k]) == 0:
        k += 1
    return flint.nmod_poly(c[k:], P_MOD)


def monic(f):
    return f * flint.nmod(1, P_MOD) / f.leading_coefficient() if not f.is_zero() else f


def echelon(M):
    """M: list of rows (lists of nmod_poly).  Column ops only.  Returns
    (M', pivots) with pivots[i] = column of row i's pivot or None."""
    nr, nc = len(M), len(M[0])
    M = [row[:] for row in M]
    col = 0
    pivots = []
    for i in range(nr):
        while True:
            nz = [j for j in range(col, nc) if not M[i][j].is_zero()]
            if len(nz) <= 1:
                break
            j0 = min(nz, key=lambda j: M[i][j].degree())
            for j in nz:
                if j == j0:
                    continue
                qt = M[i][j] // M[i][j0]
                for r in range(nr):
                    if not M[r][j0].is_zero():
                        M[r][j] = M[r][j] - qt * M[r][j0]
        nz = [j for j in range(col, nc) if not M[i][j].is_zero()]
        if not nz:
            pivots.append(None)
            continue
        j = nz[0]
        for r in range(nr):
            M[r][col], M[r][j] = M[r][j], M[r][col]
        pivots.append(col)
        col += 1
    return M, pivots


class Frac:
    __slots__ = ("n", "d")

    def __init__(self, n, d=ONE):
        g = n.gcd(d) if not n.is_zero() else d
        self.n, self.d = n // g, d // g
        lc = self.d.leading_coefficient()
        if int(lc) != 1:
            inv = flint.nmod(1, P_MOD) / lc
            self.n, self.d = self.n * inv, self.d * inv

    def __sub__(self, o):
        return Frac(self.n * o.d - o.n * self.d, self.d * o.d)

    def mulpoly(self, f):
        return Frac(self.n * f, self.d)

    def divpoly(self, f):
        return Frac(self.n, self.d * f)


def solve(M, b):
    E, piv = echelon(M)
    y = {}
    inconsistent = []
    for i, pc in enumerate(piv):
        acc = Frac(b[i])
        for k in range(len(E[i]) if pc is None else pc):
            if k in y and not E[i][k].is_zero():
                acc = acc - y[k].mulpoly(E[i][k])
        if pc is None:
            if not acc.n.is_zero():
                inconsistent.append(i)
            continue
        y[pc] = acc.divpoly(E[i][pc])
    den = ONE
    for v in y.values():
        den = den * v.d // den.gcd(v.d)
    diag = ONE
    for i, pc in enumerate(piv):
        if pc is not None:
            diag = diag * E[i][pc]
    return strip_t(monic(den)), strip_t(monic(diag)), inconsistent, piv


# ---------------------------------------------------------------------------
# factor identification
# ---------------------------------------------------------------------------

def identify(f, mode, fixed, K=60):
    """List of (name, multiplicity) for the irreducible factors of f."""
    if f.degree() <= 0:
        return []
    fx = int(fixed)
    out = []
    _, facs = f.factor()
    for g, e in facs:
        g = monic(g)
        name = None
        if mode == "x":
            if g.degree() == 1:
                x0 = int(-g.coeffs()[0])
                for k in range(-K, K + 1):
                    w = x0 * pow(fx, 2 * k, P_MOD) % P_MOD
                    if w == 1:
                        name = "D_%d" % k
                        break
                    if w == P_MOD - 1:
                        name = "{A^2q^%d}/D_%d" % (2 * k, k)
                        break
        else:
            a2 = fx * fx % P_MOD
            for n in range(1, K + 1):
                yn1 = flint.nmod_poly([P_MOD - 1] + [0] * (n - 1) + [1], P_MOD)
                if (yn1 % g).is_zero():
                    name = "Phi_%d(q^2)" % n
                    break
            if name is None:
                for k in range(1, K + 1):
                    dk = flint.nmod_poly([P_MOD - 1] + [0] * (k - 1) + [a2], P_MOD)    # A^2 y^k - 1
                    dmk = flint.nmod_poly([a2] + [0] * (k - 1) + [P_MOD - 1], P_MOD)   # A^2 - y^k
                    if (dk % g).is_zero():
                        name = "D_%d" % k
                        break
                    if (dmk % g).is_zero():
                        name = "D_-%d" % k
                        break
                if name is None:
                    yA = flint.nmod_poly([P_MOD - a2, 1], P_MOD)
                    if (yA % g).is_zero():
                        name = "D_0"
        out.append((name or "deg%d-factor" % g.degree(), e))
    # merge factors of the same name (e.g. D_k splits into several y-factors)
    merged = {}
    for nme, e in out:
        merged.setdefault(nme, []).append(e)
    return [(k, min(v) if k.startswith("D_") and k != "D_0" and mode == "y" else max(v)) for k, v in merged.items()]


def channel_name(c):
    t, Z, Zp = c
    f = lambda la: "[" + ",".join(map(str, la)) + "]"
    return f(Z) if t == "s" else "{" + f(Z) + "," + f(Zp) + "}"


# ---------------------------------------------------------------------------

def prepare(knot, mode, seed=1, npts=None):
    """Sample and interpolate; returns everything the solver needs."""
    rng = random.Random(seed)
    fixed = FLD.random_element(rng)
    t0 = time.time()
    n = npts or 64
    ts, ent = [], {}
    while True:
        ts2, ent2 = sample(knot, mode, fixed, n - len(ts), rng)
        ts += ts2
        for k, v in ent2.items():
            ent.setdefault(k, []).extend(v)
        polys, bad = interpolate_all(ts, ent, (len(ts) - 4) // 2)
        if not bad:
            break
        if n >= 1200:
            raise RuntimeError("interpolation failed for %d entries, e.g. %s" % (len(bad), bad[:3]))
        n *= 2
    chans = []
    for R in knot.reps:
        for c in channels_of(R):
            if c not in chans:
                chans.append(c)
    return {"knot": knot, "mode": mode, "fixed": fixed, "polys": polys, "chans": chans,
            "points": len(ts), "t0": t0}


def D_poly(k, mode, fixed):
    """D_k = {A q^k} as a polynomial in t up to units (x = A^2 or y = q^2)."""
    f = int(fixed)
    if mode == "x":
        return flint.nmod_poly([P_MOD - 1, pow(f, 2 * k, P_MOD)], P_MOD)
    a2 = f * f % P_MOD
    if k == 0:
        return ONE
    if k > 0:
        return flint.nmod_poly([P_MOD - 1] + [0] * (k - 1) + [a2], P_MOD)
    return flint.nmod_poly([a2] + [0] * (-k - 1) + [P_MOD - 1], P_MOD)


def solve_prepared(prep, allow=None):
    """allow: {channel: [k, ...]} -- G_c may have the denominator prod D_k."""
    knot, mode, fixed, polys, chans = prep["knot"], prep["mode"], prep["fixed"], prep["polys"], prep["chans"]
    reps = knot.reps
    allow = allow or {}
    dens = {c: ONE for c in chans}
    for c, ks in allow.items():
        if c in dens:
            for k in ks:
                dens[c] = dens[c] * D_poly(k, mode, fixed)
    L = ONE
    for c in chans:
        L = L * dens[c] // L.gcd(dens[c])
    mult = {c: L // dens[c] for c in chans}
    M = [[polys[(c, R)] * mult[c] if (c, R) in polys else ZERO for c in chans] for R in reps]
    b = [polys[("H", R)] * L for R in reps]
    den, diag, inconsistent, piv = solve(M, b)
    return den, diag, inconsistent, piv


def run(knot, mode, seed=1, npts=None, allow=None, prep=None):
    prep = prep or prepare(knot, mode, seed, npts)
    fixed, chans, reps = prep["fixed"], prep["chans"], knot.reps
    den, diag, inconsistent, piv = solve_prepared(prep, allow)
    # remove the equation multipliers W from the product of pivots
    if mode == "x":
        Wp = flint.nmod_poly([P_MOD - 1, 1], P_MOD) ** 2
    else:
        Wp = ONE
        for k in range(2, knot.N + 1):
            Wp = Wp * flint.nmod_poly([1] * k, P_MOD)
    rank = sum(p is not None for p in piv)
    for _ in range(rank):
        qt, rm = divmod(diag, Wp)
        if not rm.is_zero():
            break
        diag = qt
    return {
        "knot": knot.label, "mode": mode, "N": knot.N, "nreps": len(reps), "nchan": len(chans),
        "points": prep["points"], "rank": rank,
        "obstruction": identify(den, mode, fixed), "obstruction_deg": den.degree(),
        "F_content": identify(diag, mode, fixed), "inconsistent_rows": [list(reps[i]) for i in inconsistent],
        "seconds": round(time.time() - prep["t0"], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["torus", "db", "table"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--modes", default="xy")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    knot = Knot(a.kind, a.args, a.N)
    for mode in a.modes:
        print(json.dumps(run(knot, mode, a.seed)), flush=True)


if __name__ == "__main__":
    main()
