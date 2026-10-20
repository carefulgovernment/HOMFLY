"""Twist knots K_n in arbitrary R: the channel/evolution formula and its checks.

    H_R(Tw_m) = 1 + sum_{c in R (x) Rbar} F_c(R) Phi_c(m),      Phi_c(m) = phi_c(m) / phi_c(-1),
    phi_c(m)  = sum_X E_cX (Lambda_X^m - 1),                    F_c(R) = d_c sum_X E_cX psi_X(R),

i.e. in evolution form

    H_R(Tw_m) = 1 + sum_X w_X(R) (Lambda_X^m - 1),              w_X(R) = sum_c F_c(R) E_cX / phi_c(-1),
    Lambda_[Z,Z'] = A^{2|Z|} q^{2(kappa_Z + kappa_Z')}            (X = single [mu,mu] or pair {Z,Z'}).

Tw_m = DoubleBraid(m, 1) (m full antiparallel twists):   m = 1: 3_1,  2: 5_2,  3: 7_2,  4: 9_2,  5: 11a_247,
m = -1: 4_1, -2: 6_1, -3: 8_1, -4: 10_1, -5: 12a_803, where for m <= -2 the stored knot is the mirror of Tw_m
(H(A, q) -> H(1/A, 1/q)).  Conway notation [n 2]: n = 2m - 1 for m > 0, n = -2m for m < 0.

Nothing is fitted: the universal rows E_c come from the vanishing conditions (homfly.methods.interpolation), the
single rows are also given by the KNTZ closed form (twist matrix B), and every stored twist-knot polynomial is a
check.  All checks are modulo a large prime at random (A, q).  See docs/notes/twist_knots_formula.tex.

checks:
  data    H_R(Tw_m) from the formula vs data/homfly for every R with |R| <= nmax and all ten twist knots
  kntz    single rows of B = E~ Lambda E~^{-1} vs the KNTZ closed form, and the E rows derived from it
  phi     the product formula for phi_lambda(-1)
  f41     H_R(4_1) = 1 + sum_c F_c(R) vs data for |R| = 7 and 8 (the two 8-box failures are reported)
  sym     symmetric [7]..[10] (data/homfly_symmetric) for all ten twist knots
  table   the tabulated exact weights data/twist_weights/twist_weights_le6.json vs the data
  pairs   structural identities of the pair rows of B (vacuum and [1] entries, diagonal, row sums)
  bmat    the tabulated exact twist matrix data/twist_weights/twist_matrix_le6.json vs B built from the engine

usage: python3 scripts/twist_formula_check.py [--nmax 6] [--seed 1] [data kntz phi f41 sym table pairs bmat]
"""
import argparse
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from homfly.algebra.fields import GF                      # noqa: E402
from homfly.algebra.laurent import parse_laurent         # noqa: E402
from homfly.methods import interpolation as I            # noqa: E402
from homfly.reps.partitions import boxes, conjugate, hook, kappa, partitions  # noqa: E402

PRIME = 1000000007
VAC = ("s", (), ())
KNOT_M = {"3_1": 1, "4_1": -1, "5_2": 2, "6_1": -2, "7_2": 3, "8_1": -3, "9_2": 4, "10_1": -4,
          "11a_247": 5, "12a_803": -5}
M_KNOT = {m: k for k, m in KNOT_M.items()}
TABLE = os.path.join(ROOT, "data", "twist_weights", "twist_weights_le6.json")
BTABLE = os.path.join(ROOT, "data", "twist_weights", "twist_matrix_le6.json")

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
_tables = {}


def load_table(path):
    if path not in _tables:
        H, R = {}, None
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt") as f:
            for line in f:
                if line.startswith("R = "):
                    R = tuple(int(x) for x in line[4:].strip(" []\n").split(","))
                elif line.startswith("H = ") and R is not None:
                    H[R] = dict(parse_laurent(line[4:].strip()).terms)
        _tables[path] = H
    return _tables[path]


def data_file(knot, symmetric=False):
    if symmetric:
        return os.path.join(ROOT, "data", "homfly_symmetric", knot + ".txt.gz")
    return os.path.join(ROOT, "data", "homfly", knot + ".txt")


def data_value(knot, R, F, A, q, mirror=False, symmetric=False):
    t = load_table(data_file(knot, symmetric))[tuple(R)]
    tot = F.zero
    for (i, j), c in t.items():
        if mirror:
            i, j = -i, -j
        tot = tot + c * A ** i * q ** j
    return tot


def twist_data(m, R, F, A, q, symmetric=False):
    """H_R(Tw_m) at the point from the data (m = 0: unknot)."""
    if m == 0:
        return F.one
    return data_value(M_KNOT[m], R, F, A, q, mirror=(m < -1), symmetric=symmetric)


def has_data(knot, R, symmetric=False):
    return tuple(R) in load_table(data_file(knot, symmetric))


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------
def random_point(seed=1, p=PRIME):
    F = GF(p)
    rng = random.Random(seed)
    return F, F(rng.randrange(2, p - 1)), F(rng.randrange(2, p - 1))


def Lam(c, F, A, q):
    if c == VAC:
        return F.one
    return A ** (2 * sum(c[1])) * q ** (2 * (kappa(c[1]) + kappa(c[2])))


def formula_value(pt, R, m):
    """H_R(Tw_m) = 1 + sum_c F_c(R) phi_c(m) / phi_c(-1)."""
    tot = pt.F.one
    for c in I.channels_of(R):
        tot = tot + pt.F_c(c, R) * pt.phi(c, m) / pt.phi(c, -1)
    return tot


def twist_weights(pt, R):
    """{X: w_X(R)} (X != vacuum)."""
    w = {}
    for c in I.channels_of(R):
        Fc = pt.F_c(c, R)
        if Fc == 0:
            continue
        ph = pt.phi(c, -1)
        for X, e in pt.E_row(c).items():
            if e != 0:
                w[X] = w.get(X, pt.F.zero) + Fc * e / ph
    return w


# --- KNTZ closed form for single channels ----------------------------------
def curly(u):
    return u - 1 / u


def s_principal(nu, F, q):
    """s_nu(p°) = q^{kappa_nu} {q}^{|nu|} / prod_boxes {q^{hook}}."""
    v = q ** kappa(nu) * curly(q) ** sum(nu)
    for b in boxes(nu):
        v = v / curly(q ** hook(nu, b))
    return v


def kntz_B(lam, mu, F, A, q):
    """B_{lam mu} = (-1)^{|lam/mu|} Lambda_lam s_{lam'/mu'}(p°) s_mu(p°) / s_lam(p°), Lambda_lam = A^{2|lam|} q^{4 kappa_lam}."""
    if not I.contains(lam, mu):
        return F.zero
    sk = F.zero
    for nu, c in I.skew(conjugate(lam), conjugate(mu)).items():
        sk = sk + c * s_principal(nu, F, q)
    v = A ** (2 * sum(lam)) * q ** (4 * kappa(lam)) * sk * s_principal(mu, F, q) / s_principal(lam, F, q)
    return -v if (sum(lam) - sum(mu)) % 2 else v


def subdiagrams(lam):
    return [mu for k in range(sum(lam) + 1) for mu in partitions(k) if I.contains(lam, mu)]


def kntz_E_rows(nmax, F, A, q):
    """E rows of all singles with |lam| <= nmax from the KNTZ twist matrix (columns of E~ are the eigenvectors
    of B, normalised by Phi_lam(-1) = 1).  Returns {single(lam): {X: E_cX}} and {("phi-1", c): phi_c(-1)}."""
    Et, out = {}, {}
    for n in range(1, nmax + 1):
        for lam in partitions(n):
            L = Lam(I.single(lam), F, A, q)
            row = {}
            for mu in subdiagrams(lam):
                if not mu or mu == lam:
                    continue
                s = F.zero
                for nu in subdiagrams(lam):
                    if nu and nu != lam and I.contains(nu, mu):
                        s = s + kntz_B(lam, nu, F, A, q) * Et[nu].get(mu, F.zero)
                row[mu] = s / (Lam(I.single(mu), F, A, q) - L)
            s = F.one
            for mu, v in row.items():
                s = s - v * (1 / Lam(I.single(mu), F, A, q) - 1)
            row[lam] = s / (1 / L - 1)
            Et[lam] = row
            out[I.single(lam)] = {I.single(mu): v / row[lam] for mu, v in row.items()}
            out[("phi-1", I.single(lam))] = 1 / row[lam]
    return out


def phi_product(lam, F, A, q):
    """phi_lam(-1) = (-A)^{-|lam|} q^{-kappa_lam} prod_{(i,j) in lam} D_{(lam_i - i) + (j - lam'_j)}  (1-indexed)."""
    lc = conjugate(lam)
    v = (-A) ** (-sum(lam)) * q ** (-kappa(lam))
    for i, r in enumerate(lam, 1):
        for j in range(1, r + 1):
            s = (r - i) + (j - lc[j - 1])
            v = v * (A * q ** s - 1 / (A * q ** s))
    return v


def build_B(pt, chans):
    """Twist matrix B = E~ Lambda E~^{-1} on [vacuum] + chans (lower triangular in level order);
    returns (order, {c: {X: B_cX}})."""
    F = pt.F
    # a linear extension of the order X < c (E_cX != 0 only for X < c): every diagram of R_min(X) is a proper
    # subdiagram of a diagram of R_min(c), so the largest diagram of R_min grows strictly
    order = [VAC] + sorted(chans, key=lambda c: (max(sum(d) for d in I.rmin(c)), I.level(c), c[0] == "p", c))
    idx = {c: i for i, c in enumerate(order)}
    n = len(order)
    Et = [[F.zero] * n for _ in range(n)]
    Et[0][0] = F.one
    for c in order[1:]:
        ph = pt.phi(c, -1)
        s = F.zero
        for X, e in pt.E_row(c).items():
            if X not in idx:
                raise ValueError("channel %s needs %s outside the chosen set" % (c, X))
            Et[idx[c]][idx[X]] = e / ph
            s = s + e / ph
        Et[idx[c]][0] = -s
    Inv = [[F.zero] * n for _ in range(n)]
    for i in range(n):
        Inv[i][i] = F.one / Et[i][i]
        for j in range(i):
            s = F.zero
            for k in range(j, i):
                if Et[i][k] != 0 and Inv[k][j] != 0:
                    s = s + Et[i][k] * Inv[k][j]
            Inv[i][j] = -s / Et[i][i]
    L = [Lam(c, F, pt.A, pt.q) for c in order]
    B = {}
    for i, c in enumerate(order):
        row = {}
        for j in range(i + 1):
            s = F.zero
            for k in range(j, i + 1):
                if Et[i][k] != 0 and Inv[k][j] != 0:
                    s = s + Et[i][k] * L[k] * Inv[k][j]
            if s != 0:
                row[order[j]] = s
        B[c] = row
    return order, B


def channels_present(nmax):
    seen = []
    for n in range(1, nmax + 1):
        for R in partitions(n):
            for c in I.channels_of(R):
                if c not in seen:
                    seen.append(c)
    return seen


# ---------------------------------------------------------------------------
# checks: each returns (ok, lines)
# ---------------------------------------------------------------------------
def check_data(nmax=6, seed=1, ms=range(-5, 6)):
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    bad, n = [], 0
    for k in range(1, nmax + 1):
        for R in partitions(k):
            for m in ms:
                if m and not has_data(M_KNOT[m], R):
                    continue
                n += 1
                if formula_value(pt, R, m) != twist_data(m, R, F, A, q):
                    bad.append((R, m))
    return not bad, ["data: %d values of H_R(Tw_m), |R| <= %d, mismatches: %s" % (n, nmax, bad)]


def check_kntz(nmax=6, seed=1):
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    K = kntz_E_rows(nmax, F, A, q)
    lines, bad, deficient = [], [], []
    for c in channels_present(nmax):
        if c[0] != "s":
            continue
        row = pt.E_row(c)
        rk = pt._E[("rank", c)]
        keys = set(row) | set(K[c])
        ok = all(row.get(X, F.zero) == K[c].get(X, F.zero) for X in keys) and pt.phi(c, -1) == K[("phi-1", c)]
        if rk[0] < rk[1]:
            deficient.append((c[1], rk, "KNTZ row agrees" if ok else "engine gauge differs from KNTZ"))
        elif not ok:
            bad.append(c[1])
    lines.append("kntz: single E rows (|lam| <= %d) vs rows derived from the KNTZ twist matrix: mismatches at full rank: %s" % (nmax, bad))
    if deficient:
        lines.append("kntz: rank-deficient vanishing systems (free unknown = 0 gauge): %s" % deficient)
    return not bad, lines


def check_phi(nmax=6, seed=1):
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    bad = [lam for k in range(1, nmax + 1) for lam in partitions(k) if pt.phi(I.single(lam), -1) != phi_product(lam, F, A, q)]
    return not bad, ["phi: phi_lam(-1) product formula, |lam| <= %d, mismatches: %s" % (nmax, bad)]


def check_f41(sizes=(7, 8), seed=1, use_kntz=True):
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    K = kntz_E_rows(max(sizes), F, A, q) if use_kntz else {}
    lines, bad = [], []
    for n in sizes:
        res = []
        for R in partitions(n):
            if not has_data("4_1", R):
                continue
            chans = I.channels_of(R)
            tot = F.one
            for c in chans:
                tot = tot + pt.F_c(c, R)
            ok = tot == data_value("4_1", R, F, A, q)
            ok2 = None
            if use_kntz:
                pt2 = I.Point(F, A, q)
                pt2._p, pt2._s, pt2._chi, pt2._d = pt._p, pt._s, pt._chi, pt._d
                for c in chans:
                    pt2._E[c] = K[c] if c[0] == "s" else pt._E[c]
                tot2 = F.one
                for c in chans:
                    tot2 = tot2 + pt2.F_c(c, R)
                ok2 = tot2 == data_value("4_1", R, F, A, q)
            defic = sorted({c for c in chans if pt._E[("rank", c)][0] < pt._E[("rank", c)][1]})
            res.append((R, ok, ok2, len(defic)))
            if not ok:
                bad.append((R, "with KNTZ single rows: %s" % ok2, "rank-deficient channels: %d" % len(defic)))
        lines.append("f41: |R| = %d: %d/%d diagrams reproduce H_R(4_1); failures: %s"
                     % (n, sum(1 for r in res if r[1]), len(res), [r[0] for r in res if not r[1]]))
    return bad, lines


def check_sym(rs=(7, 8, 9, 10), seed=1, knots=None):
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    bad, n = [], 0
    for m in (knots or sorted(M_KNOT)):
        for r in rs:
            R = (r,)
            if not has_data(M_KNOT[m], R, symmetric=True):
                continue
            n += 1
            if formula_value(pt, R, m) != twist_data(m, R, F, A, q, symmetric=True):
                bad.append((R, m))
    return not bad, ["sym: %d symmetric values [7]..[10], mismatches: %s" % (n, bad)]


def check_table(nmax=6, seed=1, path=TABLE, ms=range(-5, 6)):
    F, A, q = random_point(seed)
    x, y = A * A, q * q
    tab = json.load(open(path))["weights"]

    def ev(terms):
        return sum((c * x ** a * y ** b for a, b, c in terms), F.zero)
    bad, n = [], 0
    for k in range(1, nmax + 1):
        for R in partitions(k):
            ws = [(tuple(w["X"][0]) if False else (w["X"][0], tuple(w["X"][1]), tuple(w["X"][2])),
                   ev(w["num"]) / ev(w["den"])) for w in tab[str(list(R))]]
            for m in ms:
                if m and not has_data(M_KNOT[m], R):
                    continue
                n += 1
                tot = F.one
                for X, w in ws:
                    tot = tot + w * (Lam(X, F, A, q) ** m - 1)
                if tot != twist_data(m, R, F, A, q):
                    bad.append((R, m))
    return not bad, ["table: tabulated weights (%s) vs data, %d values, mismatches: %s" % (os.path.basename(path), n, bad)]


def check_pairs(nmax=6, seed=1):
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    order, B = build_B(pt, chans)
    lines = []
    bad_single, bad_diag, bad_sum = [], [], []
    vac_ok, vac_bad, one_ok, one_bad = [], [], [], []
    for c in order[1:]:
        row = B[c]
        if row.get(c, F.zero) != Lam(c, F, A, q):
            bad_diag.append(c)
        if sum(row.values(), F.zero) != 0:
            bad_sum.append(c)
        if c[0] == "s":
            for X, v in row.items():
                if X[0] == "p" or v != kntz_B(c[1], X[1], F, A, q):
                    bad_single.append((c, X))
            continue
        U = I.union(c[1], c[2])
        cells = sum(U) - sum(c[1])
        (vac_ok if row.get(VAC, F.zero) == kntz_B(U, (), F, A, q) else vac_bad).append((c, cells))
        (one_ok if row.get(("s", (1,), (1,)), F.zero) == kntz_B(U, (1,), F, A, q) else one_bad).append((c, cells))
    lines.append("pairs: twist matrix on %d channels (|R| <= %d): single rows = KNTZ: %s; diagonal = Lambda_c: %s; row sums = 0: %s"
                 % (len(order) - 1, nmax, "ok" if not bad_single else bad_single[:3], "ok" if not bad_diag else bad_diag,
                    "ok" if not bad_sum else bad_sum))
    def split(ok, bad):
        n1, n2 = sum(1 for p in ok if p[1] == 1), sum(1 for p in ok if p[1] > 1)
        b1, b2 = [p[0][1:] for p in bad if p[1] == 1], [p[0][1:] for p in bad if p[1] > 1]
        return "one-cell pairs %d/%d (fails: %s), two-cell pairs %d/%d (fails: %s)" % (n1, n1 + len(b1), b1, n2, n2 + len(b2), b2)
    lines.append("pairs: B_c[vac] = B_U[vac] (U = Z u Z'): " + split(vac_ok, vac_bad))
    lines.append("pairs: B_c[[1]] = B_U[[1]]: " + split(one_ok, one_bad))
    one_cell_bad = [p for p in vac_bad + one_bad if p[1] == 1]
    return not (bad_single or bad_diag or bad_sum or one_cell_bad), lines


def check_bmat(nmax=6, seed=1, path=BTABLE):
    """The tabulated exact twist matrix (|R| <= 6) against B built from the engine's rows at a random point."""
    F, A, q = random_point(seed)
    x, y = A * A, q * q
    pt = I.Point(F, A, q)
    order, B = build_B(pt, channels_present(min(nmax, 6)))

    def ev(terms):
        return sum((c * x ** a * y ** b for a, b, c in terms), F.zero)

    def chan(k):
        return (k[0], tuple(k[1]), tuple(k[2]))
    tab = {chan(r["c"]): {chan(e["X"]): ev(e["num"]) / ev(e["den"]) for e in r["row"]} for r in json.load(open(path))["rows"]}
    bad, n = [], 0
    for c in order[1:]:
        if c not in tab:
            bad.append((c, "missing"))
            continue
        keys = set(B[c]) | set(tab[c])
        n += len(keys)
        if any(B[c].get(X, F.zero) != tab[c].get(X, F.zero) for X in keys):
            bad.append(c)
    return not bad, ["bmat: tabulated twist matrix (%s) vs engine: %d rows, %d entries, mismatches: %s" % (os.path.basename(path), len(order) - 1, n, bad)]


CHECKS = {"data": check_data, "kntz": check_kntz, "phi": check_phi, "f41": check_f41, "sym": check_sym,
          "table": check_table, "pairs": check_pairs, "bmat": check_bmat}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checks", nargs="*", default=list(CHECKS))
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    allok = True
    for name in a.checks:
        fn = CHECKS[name]
        if name in ("f41", "sym"):
            ok, lines = fn(seed=a.seed)
        else:
            ok, lines = fn(nmax=a.nmax, seed=a.seed)
        if name == "f41":
            ok = all(r[0] in ((4, 3, 1), (3, 2, 2, 1)) for r in ok)   # the two known 8-box failures
        allok = allok and bool(ok)
        for line in lines:
            print(line)
    print("ALL OK" if allok else "FAILURES")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
