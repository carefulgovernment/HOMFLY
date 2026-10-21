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
  binom   the binomial matrix N = E~^{-1} (Lambda_c^m = sum_X N_cX Phi_X(m)): the closed reverse-tableau formula for
          all single rows, N_c[vac] = 1 and the [1]-column formula for all rows
  pairn   pair rows of N: the one-cell diagonal, the first pair column and the two-box columns [1,1], [2]
  union   pair rows of B from the union diagram (one-cell pairs, all columns except the resonant ones)
  hook    the closed rows of B for the hook family {[k-1,1],[k]} and its transpose
  khook   the K-hook law for the pair columns of the pair rows of N (diagonal, first pair column and all adjacent
          non-landing entries), and the B entries it implies
  fourrow the four-row formula: every single column of a one-cell pair row of N (in the merged basis at the resonant
          columns) from the closed single rows of V, Z, Z', U
  adjres  the residue formula for every adjacent pair column (|U_c| = |U_X| + 1) of a one-cell pair row, landing
          entries included: zero rule, type B (two shared corners) and one shared corner; checked on N and on B
  gap2    the gap-two formula T_cX = sum_Y T_cY T_YX / (1 + q^{a_1(Y) - a_2(Y)}) for the pair columns with
          |U_c| = |U_X| + 2, built from the adjacent residues only; B_cX and (via the chain relation) N_cX
  roots   every pair column of every one-cell pair row at every gap: the root formula (the chain expansion of N_cX
          must be regular at the roots of Lambda_c = Lambda_X that are not zeros of a D_k), in its rational form in
          F[y]/(cyclotomic factor), from the adjacent residues only; B and N compared with the engine
  fullrow the whole one-cell pair rows of N and B: single columns of N from the four-row formula minus the root-formula
          pair entries at the resonant columns, single columns of B from N B = Lambda N with the KNTZ single rows
  kernel  every single column of every pair row of N (one-cell and non-resonant two-cell pairs, merged basis) from one
          sign-flip kernel over the states of the moved cells applied to the closed single rows
  topweight  the top-weight law: w_c(R) = m_c(R) d_c rho(R), rho(R) = (-A)^{|R|} q^{kappa_R} prod_b {q^{h(b)}}^2 / D_{c(b)},
          for every channel c whose only minimal diagram is R (one rho for all top channels of R; m_c(R) the
          multiplicity of the composite of c in R (x) Rbar)
  mult    the multiplicity factor of the top-weight law at the minimal diagrams of all pair channels (combinatorics,
          at least 12 boxes): 1, except 2 for every two-cell separate/separate pair (first at [4,3,2,1])
  closed  end to end from the closed formulas only: H_R(Tw_m) for every R whose channels are singles and one-cell
          pairs, from the closed rows of N (no vanishing condition solved), against the data and Rosso-Jones T(2,3)
  vacuum  the vacuum law: E~_{c,vac} (the constant term of Phi_c(m)) = pi(R_min(c)) = prod_b A q^{c(b)} / D_{c(b)} for every
          channel with one minimal diagram, and rho = (-1)^{|R|} pi prod_b {q^{h(b)}}^2
  twocell the pair diagonal N_cc (one-cell and non-resonant two-cell pairs, hinge shift for domino/separate), the first pair
          column Ebar_1^2/(2 A^3 D_0), and for the two-cell rows the partner (P) and X* (P*) entries of B and the zeros of N
  lagrange  the light columns of the non-resonant rows of N (singles, one-cell, two-cell) as Lagrange interpolants (L) in
          the Ebar-even composite power sums on the channels of one domino orientation (node census (U) to weight 10),
          and the explicit weight-3 columns

usage: python3 scripts/twist_formula_check.py [--nmax 6] [--seed 1] [--nb 10] [data kntz phi f41 sym table pairs bmat binom pairn union hook khook fourrow adjres gap2 roots fullrow kernel topweight mult closed vacuum
       twocell lagrange]
"""
import argparse
import gzip
import json
import math
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


def _order(chans):
    # a linear extension of the order X < c (E_cX != 0 only for X < c): every diagram of R_min(X) is a proper
    # subdiagram of a diagram of R_min(c), so the largest diagram of R_min grows strictly
    return [VAC] + sorted(chans, key=lambda c: (max(sum(d) for d in I.rmin(c)), I.level(c), c[0] == "p", c))


def build_EN(pt, chans):
    """E~ = diag(1/phi(-1)) E with the vacuum column -sum, and its inverse N (both lower triangular) on
    [vacuum] + chans; returns (order, Et, N) as lists of lists."""
    F = pt.F
    order = _order(chans)
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
    return order, Et, Inv


def build_B(pt, chans):
    """Twist matrix B = E~ Lambda E~^{-1} on [vacuum] + chans (lower triangular in level order);
    returns (order, {c: {X: B_cX}})."""
    F = pt.F
    order, Et, Inv = build_EN(pt, chans)
    n = len(order)
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


def clean_channels(pt, chans):
    """Channels whose vanishing system has full rank and whose support closure contains no rank-deficient channel
    (the engine's gauge rows of the others are not reliable)."""
    bad = set()
    for c in chans:
        pt.E_row(c)
        r, n = pt._E[("rank", c)]
        if r < n:
            bad.add(c)
    changed = True
    while changed:
        changed = False
        for c in chans:
            if c not in bad and any(X in bad for X in pt.E_row(c)):
                bad.add(c)
                changed = True
    return [c for c in chans if c not in bad], sorted(bad)


# ---------------------------------------------------------------------------
# the binomial matrix N = E~^{-1}: closed single sector (reverse tableaux)
# ---------------------------------------------------------------------------
def qnum(n, F, q):
    return (q ** n - q ** (-n)) / (q - q ** (-1)) if n >= 0 else -qnum(-n, F, q)


def qfact(n, F, q):
    v = F.one
    for i in range(2, n + 1):
        v = v * qnum(i, F, q)
    return v


def qbinom(n, m, F, q):
    if m < 0 or m > n:
        return F.zero
    return qfact(n, F, q) / (qfact(m, F, q) * qfact(n - m, F, q))


def Dk(k, F, A, q):
    return A * q ** k - F.one / (A * q ** k)


def reverse_tableaux(mu, n):
    """Fillings T of the diagram mu (boxes (i, j), 0-based) with entries in 0..n-1, weakly decreasing along rows,
    strictly decreasing down columns."""
    bx = boxes(mu)
    T = {}

    def rec(k):
        if k == len(bx):
            yield dict(T)
            return
        i, j = bx[k]
        hi = n - 1
        if j > 0:
            hi = min(hi, T[(i, j - 1)])
        if i > 0:
            hi = min(hi, T[(i - 1, j)] - 1)
        for v in range(0, hi + 1):
            T[(i, j)] = v
            yield from rec(k + 1)
        T.pop((i, j), None)
    yield from rec(0)


def tableau_P(lam, mu, F, A, q):
    """P_mu(lambda) = sum over reverse tableaux T of shape mu, entries = rows of lambda, of
    prod_{alpha in mu} [lambda_i - c(alpha)] D_{lambda_i - 2i - 1 + c(alpha)},  i = T(alpha), c = content of alpha."""
    tot = F.zero
    for T in reverse_tableaux(mu, len(lam)):
        term = F.one
        for (ia, ja), i in T.items():
            c = ja - ia
            term = term * qnum(lam[i] - c, F, q) * Dk(lam[i] - 2 * i - 1 + c, F, A, q)
            if term == 0:
                break
        tot = tot + term
    return tot


def binom_N(lam, mu, F, A, q):
    """Closed form of the single block of N:  N_{lambda mu} = (-A)^{-|mu|} q^{-kappa_mu} P_mu(lambda) / prod_b [h_mu(b)]."""
    v = (-A) ** (-sum(mu)) * q ** (-kappa(mu)) if sum(mu) else F.one
    for b in boxes(mu):
        v = v / qnum(hook(mu, b), F, q)
    return v * tableau_P(lam, mu, F, A, q)


def one_cell(c):
    """For a one-cell pair c = {Z, Z'} return (Z, Z', U, b, b') with b in Z minus Z', b' in Z' minus Z, else None."""
    if c[0] != "p":
        return None
    Z, Zp = c[1], c[2]
    U = I.union(Z, Zp)
    if sum(U) != sum(Z) + 1:
        return None
    bZ = [b for b in boxes(Z) if b not in boxes(Zp)]
    bZp = [b for b in boxes(Zp) if b not in boxes(Z)]
    if len(bZ) != 1 or len(bZp) != 1:
        return None
    return Z, Zp, U, bZ[0], bZp[0]


def cont(b):
    return b[1] - b[0]


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



def check_binom(nmax=6, seed=1):
    """N = E~^{-1} from the engine vs the closed reverse-tableau formula (single rows), plus N_c[vac] = 1 and
    N_c[[1]] = -(1/2A) sum_{b in Z u Z'} D_{2c(b)} for every clean channel."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    idx = {c: i for i, c in enumerate(order)}
    one = ("s", (1,), (1,))
    n_single = bad_single = 0
    bad_pair_entry, bad_vac, bad_one = [], [], []
    for c in clean:
        i = idx[c]
        if N[i][0] != F.one:
            bad_vac.append(c)
        s1 = sum((Dk(2 * cont(b), F, A, q) for Z in (c[1], c[2]) for b in boxes(Z)), F.zero)
        if N[i][idx[one]] != -s1 / (2 * A):
            bad_one.append(c)
        if c[0] != "s":
            continue
        for X in order[1:]:
            v = N[i][idx[X]]
            if X[0] == "p":
                if v != 0:
                    bad_pair_entry.append((c, X))
                continue
            n_single += 1
            if v != binom_N(c[1], X[1], F, A, q):
                bad_single += 1
    lines = ["binom: N = E~^{-1} on %d channels (|R| <= %d), %d clean (excluded: %s)" % (len(order) - 1, nmax, len(clean), excluded),
             "binom: single rows vs the reverse-tableau formula: %d entries, mismatches %d; pair entries in single rows: %s"
             % (n_single, bad_single, "none" if not bad_pair_entry else bad_pair_entry[:3]),
             "binom: N_c[vac] = 1: %s; N_c[[1]] = -(1/2A) sum_{Z u Z'} D_{2c}: %s"
             % ("ok" if not bad_vac else bad_vac[:3], "ok" if not bad_one else bad_one[:3])]
    return not (bad_single or bad_pair_entry or bad_vac or bad_one), lines


def check_pairn(nmax=6, seed=1):
    """Pair rows of N (clean one-cell pairs): diagonal, first pair column {[1,1],[2]}, columns [1,1] and [2]."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    idx = {c: i for i, c in enumerate(order)}
    curlyq = q - q ** (-1)
    p0 = ("p", (1, 1), (2,))
    n_pairs = 0
    bad_diag, bad_col, bad_two = [], [], []
    for c in clean:
        oc = one_cell(c)
        if oc is None:
            continue
        Z, Zp, U, b, bp = oc
        n_pairs += 1
        i = idx[c]
        d, s_ = cont(bp) - cont(b), cont(b) + cont(bp)
        curly_d2 = (q ** d - q ** (-d)) ** 2
        # diagonal: 2 N_cc = -(-A)^{-|U|} q^{-kappa_U} {q^d}^2 prod_{cells of U except b, b'} D_{a-l+2c}
        prod = F.one
        for x in boxes(U):
            if x in (b, bp):
                continue
            a_, l_ = U[x[0]] - x[1] - 1, conjugate(U)[x[1]] - x[0] - 1
            prod = prod * Dk(a_ - l_ + 2 * cont(x), F, A, q)
        if 2 * N[i][i] != -((-A) ** (-sum(U))) * q ** (-kappa(U)) * curly_d2 * prod:
            bad_diag.append(c)
        # first pair column: N_{c,{[1,1],[2]}} = [d]^2 {q}^2 D_s^2 / (2 A^3 D_0)
        if p0 in idx and idx[p0] < i:
            want = qnum(d, F, q) ** 2 * curlyq ** 2 * Dk(s_, F, A, q) ** 2 / (2 * A ** 3 * Dk(0, F, A, q))
            if N[i][idx[p0]] != want:
                bad_col.append(c)
        # columns [1,1], [2]:  2 N_{c,mu} (-A)^2 q^{kappa_mu} [2] = P_mu(Z) + P_mu(Z') - {q^d}^2
        for mu in ((1, 1), (2,)):
            X = ("s", mu, mu)
            lhs = 2 * N[i][idx[X]] * A * A * q ** kappa(mu) * qnum(2, F, q)
            if lhs != tableau_P(Z, mu, F, A, q) + tableau_P(Zp, mu, F, A, q) - curly_d2:
                bad_two.append((c, mu))
    lines = ["pairn: %d clean one-cell pairs (|R| <= %d): diagonal formula: %s; first pair column: %s; columns [1,1],[2] = P(Z)+P(Z')-{q^d}^2: %s"
             % (n_pairs, nmax, "ok" if not bad_diag else bad_diag[:3], "ok" if not bad_col else bad_col[:3], "ok" if not bad_two else bad_two[:3])]
    return not (bad_diag or bad_col or bad_two), lines


def check_union(nmax=6, seed=1):
    """One-cell pair rows of B from the KNTZ rows of U, Z, Z':
    B_{c,X} = B_{U,X}/(A^2 q^{2s}) + Lambda_c (delta_{Xc} - delta_{XU}) + (A D_s/{q^d}) (B_{Z,X} - B_{Z',X})
    for every column X that is neither a pair below c nor the union of one (there the row has poles)."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, B = build_B(pt, chans)
    n_pairs = n_entries = 0
    bad = []
    for c in clean:
        oc = one_cell(c)
        if oc is None:
            continue
        Z, Zp, U, b, bp = oc
        n_pairs += 1
        d, s_ = cont(bp) - cont(b), cont(b) + cont(bp)
        pairs_below = [X for X in B[c] if X[0] == "p" and X != c]
        resonant = set(pairs_below) | set(I.single(I.union(X[1], X[2])) for X in pairs_below)
        Lc = Lam(c, F, A, q)
        for X in order:
            if X[0] == "p" and X != c:
                continue            # pair columns: resonant (below c) or zero
            if X in resonant:
                continue
            n_entries += 1
            want = F.zero
            if X[0] == "s":
                mu = X[1]
                want = kntz_B(U, mu, F, A, q) / (A * A * q ** (2 * s_)) + A * Dk(s_, F, A, q) / (q ** d - q ** (-d)) * (kntz_B(Z, mu, F, A, q) - kntz_B(Zp, mu, F, A, q))
                if X == I.single(U):
                    want = want - Lc
            else:
                want = Lc
            if B[c].get(X, F.zero) != want:
                bad.append((c, X))
    lines = ["union: %d clean one-cell pairs (|R| <= %d), %d non-resonant entries of B from the KNTZ rows of U, Z, Z': mismatches %s"
             % (n_pairs, nmax, n_entries, "none" if not bad else bad[:4])]
    return not bad, lines


def hook_pair_row(k, F, A, q, transpose=False):
    """Closed row of B for c = {[k-1,1],[k]} (U = [k,1]); transpose: {[2,1^{k-2}],[1^k]} via q -> -1/q."""
    if transpose:
        qq = -F.one / q
        row = hook_pair_row(k, F, A, qq, False)
        return {I.pair(conjugate(X[1]), conjugate(X[2])) if X[0] == "p" else
                (I.single(conjugate(X[1])) if X != VAC else VAC): v for X, v in row.items()}
    c = ("p", (k - 1, 1), (k,)) if k > 2 else ("p", (1, 1), (2,))
    U = (k, 1)
    Lc = A ** (2 * k) * q ** (2 * k * (k - 2))
    LU = Lam(I.single(U), F, A, q)
    cq = lambda n: q ** n - q ** (-n)
    v = {}
    for j in range(2, k + 1):
        v[("s", (j,), (j,))] = (-1) ** (k - j + 1) * q ** (-(k - j) * (k - 1) - j) * qbinom(k - 2, j - 2, F, q) / cq(j)
    v[("s", (1, 1), (1, 1))] = (-1) ** k * q ** (2 - (k - 1) * (k - 2)) / cq(2)
    for j in range(2, k):
        Xj = ("p", (j - 1, 1), (j,)) if j > 2 else ("p", (1, 1), (2,))
        v[Xj] = (-1) ** (k - j) * qbinom(k - 2, j - 2, F, q) / (A * q ** ((k - j) * (k - 1) + j - 2) * Dk(j - 2, F, A, q))
        v[("s", (j, 1), (j, 1))] = -A * A * q ** (2 * j - 4) * v[Xj] - q ** (2 * j + 2) * v[("s", (j + 1,), (j + 1,))]
    row = {}
    for mu in subdiagrams(U):
        val = kntz_B(U, mu, F, A, q)
        if val != 0:
            row[I.single(mu) if mu else VAC] = val
    row[I.single(U)] = row.get(I.single(U), F.zero) - LU
    row[c] = row.get(c, F.zero) + Lc
    for X, val in v.items():
        row[X] = row.get(X, F.zero) + (LU - Lc) * val
    return {X: val for X, val in row.items() if val != 0}


def check_hook(nmax=6, seed=1):
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, B = build_B(pt, chans)
    bad, n = [], 0
    for k in range(2, nmax):
        for tr in (False, True):
            c = ("p", (k - 1, 1), (k,)) if k > 2 else ("p", (1, 1), (2,))
            if tr:
                c = I.pair(conjugate(c[1]), conjugate(c[2]))
            if c not in clean or c not in B:
                continue
            row = hook_pair_row(k, F, A, q, tr)
            n += 1
            keys = set(row) | set(B[c])
            if any(row.get(X, F.zero) != B[c].get(X, F.zero) for X in keys):
                bad.append(c)
    return not bad, ["hook: closed rows of B for {[k-1,1],[k]} and transposes, k = 2..%d: %d rows, mismatches %s" % (nmax - 1, n, "none" if not bad else bad)]


def arm(lam, b):
    return lam[b[0]] - b[1] - 1


def leg(lam, b):
    return conjugate(lam)[b[1]] - b[0] - 1


def idx(lam, b):
    """a - l + 2c of the box b in lam (the D-index of the product formulas)."""
    return arm(lam, b) - leg(lam, b) + 2 * cont(b)


def add_cells(lam, cells):
    """The diagram lam with the cells added (None if the result is not a partition)."""
    rows = list(lam)
    for i, j in cells:
        while len(rows) <= i:
            rows.append(0)
        if j >= rows[i]:
            rows[i] = j + 1
    while rows and rows[-1] == 0:
        rows.pop()
    if any(rows[i] > rows[i - 1] for i in range(1, len(rows))):
        return None
    return tuple(rows)


def pair_info(c):
    """One-cell pair c = {V+b, V+b'} with c(b) < c(b'): U, V, b, b', the hinge h = (row of b', column of b) in V,
    d = c(b') - c(b), s = c(b) + c(b').  None for other channels."""
    oc = one_cell(c)
    if oc is None:
        return None
    Z, Zp, U, b, bp = oc
    if cont(b) > cont(bp):
        b, bp = bp, b
    V = tuple(v for v in (min(Z[i], Zp[i]) for i in range(min(len(Z), len(Zp)))) if v)
    return {"U": U, "V": V, "b": b, "bp": bp, "h": (bp[0], b[1]), "d": cont(bp) - cont(b), "s": cont(b) + cont(bp)}


def khook_N(c, X, F, A, q):
    """The K-hook law for the pair column X of the one-cell pair row c of N:
    N_cX = -(1/2)(-A)^{-|U_X|} q^{-kappa(U_X)} {q^{d_c}}^2 D_{s_c}^2 / D_{s_X}
           * prod_{x in V_X minus h_X} [h_{V_c+K}(x)]/[h_{V_X+K}(x)] D_{idx_{U_c}(x)},   K = {b_c,b'_c} cap {b_X,b'_X}.
    None where a cell of V_X minus h_X lies outside V_c + K (the law is not claimed there)."""
    ic, iX = pair_info(c), pair_info(X)
    K = {ic["b"], ic["bp"]} & {iX["b"], iX["bp"]}
    VcK, VXK = add_cells(ic["V"], K), add_cells(iX["V"], K)
    if VcK is None or VXK is None:
        return None
    u = F.one
    for x in boxes(iX["V"]):
        if x == iX["h"]:
            continue
        if x not in boxes(VcK):
            return None
        u = u * qnum(hook(VcK, x), F, q) / qnum(hook(VXK, x), F, q) * Dk(idx(ic["U"], x), F, A, q)
    cq = q ** ic["d"] - q ** (-ic["d"])
    P = -(F.one / 2) * (-A) ** (-sum(iX["U"])) * q ** (-kappa(iX["U"])) * cq * cq
    P = P * Dk(ic["s"], F, A, q) ** 2 / Dk(iX["s"], F, A, q)
    return P * u


def check_khook(nmax=6, seed=1):
    """Pair columns X of the one-cell pair rows c of N (U_X inside U_c): N_cX = 0 when U_X = U_c; the K-hook law
    on the diagonal, the first pair column and every adjacent (|U_c| = |U_X| + 1) non-landing (h_c not in V_X minus h_X)
    entry; for those adjacent entries also B_cX = N_cX (Lambda_c - Lambda_X) / phi_c(-1).  Landing and non-adjacent
    entries are only counted."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    orderB, B = build_B(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    first = ("p", (1, 1), (2,))
    cnt = {"diag": [0, 0], "first": [0, 0], "adjacent": [0, 0], "adjacentB": [0, 0], "sameU": [0, 0]}
    other = {"landing": 0, "nonadjacent": 0}
    bad = []
    pairs = [c for c in clean if pair_info(c) is not None]
    for c in pairs:
        ic = pair_info(c)
        for X in pairs:
            iX = pair_info(X)
            if not I.contains(ic["U"], iX["U"]):
                continue
            v = N[pos[c]][pos[X]]
            if X == c:
                key = "diag"
            elif iX["U"] == ic["U"]:
                key = "sameU"
            elif X == first:
                key = "first"
            elif sum(ic["U"]) != sum(iX["U"]) + 1:
                other["nonadjacent"] += 1
                continue
            elif ic["h"] in boxes(iX["V"]) and ic["h"] != iX["h"]:
                other["landing"] += 1
                continue
            else:
                key = "adjacent"
            if key == "sameU":
                ok = v == 0
            else:
                want = khook_N(c, X, F, A, q)
                ok = want is not None and v == want
                if key == "adjacent" and want is not None:
                    Lc, LX = Lam(c, F, A, q), Lam(X, F, A, q)
                    okB = B[c].get(X, F.zero) == want * (Lc - LX) / N[pos[c]][pos[c]]
                    cnt["adjacentB"][0 if okB else 1] += 1
                    if not okB:
                        bad.append(("B", c, X))
            cnt[key][0 if ok else 1] += 1
            if not ok:
                bad.append((key, c, X))
    lines = ["khook: %d clean one-cell pairs (|R| <= %d); N_cX = 0 for U_X = U_c: %d/%d; K-hook law: diagonal %d/%d, "
             "first pair column %d/%d, adjacent non-landing %d/%d; implied B entries %d/%d; not claimed (counted only): "
             "landing %d, non-adjacent %d; mismatches %s"
             % (len(pairs), nmax, cnt["sameU"][0], sum(cnt["sameU"]), cnt["diag"][0], sum(cnt["diag"]),
                cnt["first"][0], sum(cnt["first"]), cnt["adjacent"][0], sum(cnt["adjacent"]),
                cnt["adjacentB"][0], sum(cnt["adjacentB"]), other["landing"], other["nonadjacent"],
                "none" if not bad else bad[:4])]
    return not bad, lines


def check_fourrow(nmax=6, seed=1):
    """Four-row formula for the single columns mu of a one-cell pair row c = {Z, Z'} of N (V = Z cap Z', U = Z cup Z'):
        2 N_c[mu] + 2 sum_{pairs X: U_X = mu} N_c[X]
            = ( D_s^2 (N_Z[mu] + N_Z'[mu]) - {q^d}^2 (N_U[mu] + N_V[mu]) ) / (D_{2c(b)} D_{2c(b')}),
    the four single rows taken from the closed tableau formula; the left side is the coefficient of Phi_mu in the basis
    Phi_mu, Phi_X - Phi_{U_X}.  Every single column mu <= U is checked, the vacuum and the diagonal (mu = U) included."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    unions = {}
    for X in order:
        if X[0] == "p":
            unions.setdefault(I.union(X[1], X[2]), []).append(X)
    n_pairs = n_cols = 0
    bad = []
    for c in clean:
        info = pair_info(c)
        if info is None:
            continue
        Z, Zp, U, V = c[1], c[2], info["U"], info["V"]
        b, bp = info["b"], info["bp"]
        n_pairs += 1
        cq = q ** info["d"] - q ** (-info["d"])
        den = Dk(2 * cont(b), F, A, q) * Dk(2 * cont(bp), F, A, q)
        for X in order:
            if X[0] != "s" or not I.contains(U, X[1]):
                continue
            mu = X[1]
            lhs = 2 * N[pos[c]][pos[X]]
            for Y in unions.get(mu, []):
                if Y in pos:
                    lhs = lhs + 2 * N[pos[c]][pos[Y]]
            rhs = Dk(info["s"], F, A, q) ** 2 * (binom_N(Z, mu, F, A, q) + binom_N(Zp, mu, F, A, q))
            rhs = rhs - cq * cq * (binom_N(U, mu, F, A, q) + binom_N(V, mu, F, A, q))
            n_cols += 1
            if lhs != rhs / den:
                bad.append((c, mu))
    return not bad, ["fourrow: %d clean one-cell pairs (|R| <= %d), %d single columns of N from the four closed single rows "
                     "V, Z, Z', U: mismatches %s" % (n_pairs, nmax, n_cols, "none" if not bad else bad[:4])]


def hookprod(lam, F, q):
    """H(lam) = prod_{x in lam} [h_lam(x)]."""
    v = F.one
    for x in boxes(lam):
        v = v * qnum(hook(lam, x), F, q)
    return v


def adjacent_residue(c, X, F, A, q):
    """Residue r_cX (B_cX = Lambda_c r_cX q^{s_c+s_X} D_{s_c}/D_{s_X}) for an adjacent pair column X of the one-cell pair
    row c (U_c = U_X + e), landing entries included.  With k, k' the moved cells of X and K = {b, b'} cap {k, k'}:
      zero rule:  r = 0 unless {b, b'} is contained in {k, k', e};
      |K| = 2:    r = -q^{-2 s_X - c(e)} H(U_c)/H(U_X);
      |K| = 1:    r = -q^{-2 s_X - c(e)} H(V_c)/H(V_X) [dX+1][dX-1] / ([dX][dc][d3]^2),
                  k the shared corner, k' the other one, dX = c(k') - c(k), dc = c(e) - c(k), d3 = c(e) - c(k')
    (signed quantum numbers, [-n] = -[n]).  None if the formula has a vanishing denominator."""
    ic, iX = pair_info(c), pair_info(X)
    extra = [x for x in boxes(ic["U"]) if x not in boxes(iX["U"])]
    if len(extra) != 1:
        return None
    e = extra[0]
    corners = {iX["b"], iX["bp"]}
    moved = {ic["b"], ic["bp"]}
    if not moved <= corners | {e}:
        return F.zero
    K = moved & corners
    pre = -q ** (-2 * iX["s"] - cont(e))
    if len(K) == 2:
        return pre * hookprod(ic["U"], F, q) / hookprod(iX["U"], F, q)
    k = next(iter(K))
    kp = next(iter(corners - K))
    dX, dc, d3 = cont(kp) - cont(k), cont(e) - cont(k), cont(e) - cont(kp)
    den = qnum(dX, F, q) * qnum(dc, F, q) * qnum(d3, F, q) ** 2
    if den == 0:
        return None
    return pre * hookprod(ic["V"], F, q) / hookprod(iX["V"], F, q) * qnum(dX + 1, F, q) * qnum(dX - 1, F, q) / den


def check_adjres(nmax=6, seed=1):
    """Every adjacent pair column X (|U_c| = |U_X| + 1, one-cell X) of every clean one-cell pair row c: B_cX and N_cX
    from the residue formula (landing entries included), N_cX = N_cc B_cX / (Lambda_c - Lambda_X)."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    orderB, B = build_B(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    pairs = [c for c in clean if pair_info(c) is not None]
    cnt = {"zero": [0, 0], "two": [0, 0], "one": [0, 0]}
    n_land = 0
    bad, undefined = [], []
    for c in pairs:
        ic = pair_info(c)
        for X in pairs:
            iX = pair_info(X)
            if X == c or not I.contains(ic["U"], iX["U"]) or sum(ic["U"]) != sum(iX["U"]) + 1:
                continue
            r = adjacent_residue(c, X, F, A, q)
            if r is None:
                undefined.append((c, X))
                continue
            landing = ic["h"] in boxes(iX["V"]) and ic["h"] != iX["h"]
            n_land += landing
            key = "zero" if r == 0 else ("two" if len({ic["b"], ic["bp"]} & {iX["b"], iX["bp"]}) == 2 else "one")
            Lc, LX = Lam(c, F, A, q), Lam(X, F, A, q)
            Bc = Lc * r * q ** (ic["s"] + iX["s"]) * Dk(ic["s"], F, A, q) / Dk(iX["s"], F, A, q)
            Nc = N[pos[c]][pos[c]] * Bc / (Lc - LX)
            ok = B[c].get(X, F.zero) == Bc and N[pos[c]][pos[X]] == Nc
            cnt[key][0 if ok else 1] += 1
            if not ok:
                bad.append((c, X))
    tot = sum(sum(v) for v in cnt.values())
    lines = ["adjres: %d clean one-cell pairs (|R| <= %d), %d adjacent pair entries of N and B from the residue formula "
             "(%d of them landing): zero rule %d/%d, two shared corners %d/%d, one shared corner %d/%d; undefined %s; "
             "mismatches %s" % (len(pairs), nmax, tot, n_land, cnt["zero"][0], sum(cnt["zero"]), cnt["two"][0],
                                sum(cnt["two"]), cnt["one"][0], sum(cnt["one"]), "none" if not undefined else undefined[:3],
                                "none" if not bad else bad[:4])]
    return not (bad or undefined), lines


def kap2(c):
    """kappa_Z + kappa_Z' of a channel (Lambda_c = A^{2|Z|} q^{2 kap2})."""
    return kappa(c[1]) + kappa(c[2])


def adjacent_B(c, X, F, A, q):
    """B_cX of an adjacent pair column from the residue formula (None if undefined)."""
    r = adjacent_residue(c, X, F, A, q)
    if r is None:
        return None
    ic, iX = pair_info(c), pair_info(X)
    return Lam(c, F, A, q) * r * q ** (ic["s"] + iX["s"]) * Dk(ic["s"], F, A, q) / Dk(iX["s"], F, A, q)


def check_gap2(nmax=6, seed=1):
    """Pair columns X of a one-cell pair row c at gap two (|U_c| = |U_X| + 2, X one-cell): with T_cX := r_cX q^{2 s_X}
    (B_cX = Lambda_c T_cX q^{s_c - s_X} D_{s_c}/D_{s_X}),
        T_cX = sum_Y T_cY T_YX / (1 + q^{a_1(Y) - a_2(Y)}),   Lambda_c/Lambda_Y = A^2 q^{2 a_1},  Lambda_Y/Lambda_X = A^2 q^{2 a_2},
    the sum over the one-cell pairs Y with U_X < U_Y < U_c, all T's from the adjacent residue formula; and then
        N_cX = (N_cc B_cX + sum_Y N_cY B_YX) / (Lambda_c - Lambda_X)      (chain relation of N B = Lambda N)
    with N_cc, N_cY from the closed diagonal and adjacent formulas.  Both compared with the engine."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    orderB, B = build_B(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    pairs = [c for c in clean if pair_info(c) is not None]
    nB = nN = nzero = 0
    bad, undefined = [], []
    for c in pairs:
        ic = pair_info(c)
        Lc = Lam(c, F, A, q)
        Ncc = khook_N(c, c, F, A, q)
        for X in pairs:
            iX = pair_info(X)
            if not I.contains(ic["U"], iX["U"]) or sum(ic["U"]) != sum(iX["U"]) + 2:
                continue
            LX = Lam(X, F, A, q)
            T = F.zero
            chain = F.zero
            ok_def = True
            for Y in pairs:
                iY = pair_info(Y)
                if sum(iY["U"]) != sum(iX["U"]) + 1 or not I.contains(ic["U"], iY["U"]) or not I.contains(iY["U"], iX["U"]):
                    continue
                rcY, rYX = adjacent_residue(c, Y, F, A, q), adjacent_residue(Y, X, F, A, q)
                if rcY is None or rYX is None:
                    ok_def = False
                    break
                TcY, TYX = rcY * q ** (2 * iY["s"]), rYX * q ** (2 * iX["s"])
                a1 = kap2(c) - kap2(Y)
                a2 = kap2(Y) - kap2(X)
                T = T + TcY * TYX / (F.one + q ** (a1 - a2))
                LY = Lam(Y, F, A, q)
                BcY, BYX = adjacent_B(c, Y, F, A, q), adjacent_B(Y, X, F, A, q)
                NcY = Ncc * BcY / (Lc - LY)
                chain = chain + NcY * BYX
            if not ok_def:
                undefined.append((c, X))
                continue
            BcX = Lc * T * q ** (ic["s"] - iX["s"]) * Dk(ic["s"], F, A, q) / Dk(iX["s"], F, A, q)
            NcX = (Ncc * BcX + chain) / (Lc - LX)
            vB, vN = B[c].get(X, F.zero), N[pos[c]][pos[X]]
            nzero += vB == 0
            nB += 1
            nN += 1
            if vB != BcX or vN != NcX:
                bad.append((c, X))
    return not (bad or undefined), ["gap2: %d clean one-cell pairs (|R| <= %d), %d gap-two pair entries (%d of them zero) from the adjacent "
                                     "residues: B via the gap-two formula and N via the chain relation; undefined %s; mismatches %s"
                                     % (len(pairs), nmax, nB, nzero, "none" if not undefined else undefined[:3],
                                        "none" if not bad else bad[:4])]


# ---------------------------------------------------------------------------
# every gap: the root formula for the pair-pair block, and the whole one-cell pair rows of N and B
# ---------------------------------------------------------------------------
_CYC = {}


def cyclotomic(n):
    """Integer coefficients (low to high) of the cyclotomic polynomial Phi_n (exact division of u^n - 1)."""
    if n in _CYC:
        return _CYC[n]
    p = [-1] + [0] * (n - 1) + [1]
    for d in range(1, n):
        if n % d == 0:
            d_ = cyclotomic(d)
            out, r = [0] * (len(p) - len(d_) + 1), list(p)
            for i in range(len(out) - 1, -1, -1):
                out[i] = r[i + len(d_) - 1] // d_[-1]
                for j in range(len(d_)):
                    r[i + j] -= out[i] * d_[j]
            assert not any(r)
            p = out
    _CYC[n] = p
    return p


class QRing:
    """F[y]/(Psi) with y = A^{-2} and Psi(y) = q^{m1 phi(e)} Phi_e(y^{g1} q^{-m1}), e = gcd(g, 2a), g1 = g/e, m1 = 2a/e:
    the factor of y^g - q^{2a} (the equation Lambda_c = Lambda_X) whose roots are zeta q^{2a/g} with zeta a primitive
    e-th root of unity.  These roots are never zeros of a D_k (where the regularity of N is supplied by N_cc) and never
    a resonance Lambda_Y = Lambda_c.  Elements are coefficient lists of length deg Psi."""

    def __init__(self, F, q, g, a):
        e = math.gcd(g, 2 * a)
        g1, m1 = g // e, (2 * a) // e
        cyc = cyclotomic(e)
        phi_e = len(cyc) - 1
        d = g1 * phi_e
        Psi = [F.zero] * (d + 1)
        for j, cj in enumerate(cyc):
            if cj:
                Psi[g1 * j] = F(cj) * q ** (m1 * (phi_e - j))
        assert Psi[d] == F.one
        self.F, self.d, self.Psi, self.e = F, d, Psi, e

    def red(self, p):
        F, d, Psi = self.F, self.d, self.Psi
        p = list(p)
        for i in range(len(p) - 1, d - 1, -1):
            c = p[i]
            if c != 0:
                for j in range(d + 1):
                    p[i - d + j] = p[i - d + j] - c * Psi[j]
        return p[:d] + [F.zero] * (d - len(p))

    def const(self, k):
        return [k] + [self.F.zero] * (self.d - 1)

    def ymon(self, n, k):
        return self.red([self.F.zero] * n + [k])

    def mul(self, p, r):
        out = [self.F.zero] * (len(p) + len(r) - 1)
        for i, a in enumerate(p):
            if a != 0:
                for j, b in enumerate(r):
                    if b != 0:
                        out[i + j] = out[i + j] + a * b
        return self.red(out)

    @staticmethod
    def add(p, r):
        return [a + b for a, b in zip(p, r)]

    @staticmethod
    def scal(p, k):
        return [a * k for a in p]


def root_T(c, X, T, F, q, pairs):
    """T_cX at gap g = |U_c| - |U_X| >= 2 from the T's at smaller gaps (dict T[(Y, Y')]) by the root formula
        T_cX = -(1/(g-1)) sum_{zeta^g = 1, zeta != 1} S(A^2 = zeta q^{-2a/g}),
        S = sum_{chains c = Y_0 > ... > Y_k = X, k >= 2} prod_i T_{Y_{i-1} Y_i} prod_{i<k} rho_i/(1 - rho_i),
        rho_i = Lambda_{Y_i}/Lambda_c,  Lambda_c/Lambda_X = A^{2g} q^{2a},
    the expression of the requirement that the chain expansion of N_cX be regular at the roots of Lambda_c = Lambda_X
    that are not zeros of a D_k.  Rational form: with y = A^{-2}, rho_Y = y^{g_cY} q^{-2 a_cY}, the chain sum is formed
    as P/Q with Q = prod_Y (1 - rho_Y) by a transfer sum over the intermediate pairs Y, and T_cX = -P/Q in F[y]/(Psi)
    must be a constant (deg Psi coefficient equations for one unknown).  Returns (T_cX, consistent), or (None, False)
    when an ingredient is missing."""
    ic, iX = pair_info(c), pair_info(X)
    g, a = sum(ic["U"]) - sum(iX["U"]), kap2(c) - kap2(X)
    R = QRing(F, q, g, a)
    infos = {Y: pair_info(Y) for Y in pairs}
    inter = [Y for Y in pairs if sum(iX["U"]) < sum(infos[Y]["U"]) < sum(ic["U"])
             and I.contains(ic["U"], infos[Y]["U"]) and I.contains(infos[Y]["U"], iX["U"])]
    inter.sort(key=lambda Y: (-sum(infos[Y]["U"]), Y))
    rho, skip = {}, {}
    for Y in inter:
        rho[Y] = R.ymon(sum(ic["U"]) - sum(infos[Y]["U"]), q ** (-2 * (kap2(c) - kap2(Y))))
        skip[Y] = R.add(R.const(F.one), R.scal(rho[Y], -F.one))
    state = {c: R.const(F.one)}            # chains from c, keyed by their last pair; Q = prod (1 - rho) built in
    for Y in inter:
        new = {}
        for last, val in state.items():
            v = R.mul(val, skip[Y])
            new[last] = R.add(new[last], v) if last in new else v
            if last != c and not (sum(infos[last]["U"]) > sum(infos[Y]["U"]) and I.contains(infos[last]["U"], infos[Y]["U"])):
                continue
            t = T.get((last, Y))
            if t is None:
                return None, False
            if t != 0:
                v = R.mul(R.scal(val, t), rho[Y])
                new[Y] = R.add(new[Y], v) if Y in new else v
        state = new
    Q, P = state[c], R.const(F.zero)
    for last, val in state.items():
        if last == c:
            continue
        t = T.get((last, X))
        if t is None:
            return None, False
        if t != 0:
            P = R.add(P, R.scal(val, t))
    vals = []
    for j in range(R.d):
        if Q[j] != 0:
            vals.append(-P[j] / Q[j])
        elif P[j] != 0:
            return None, False
    if not vals:
        return None, False
    return vals[0], all(v == vals[0] for v in vals)


def closed_pair_block(nmax, F, A, q):
    """The pair-pair block of T, B and N for all one-cell pairs present at |R| <= nmax, from the closed formulas only:
    T at gap one from the adjacent residues (eq. adjres), at gap >= 2 from the root formula, B_cX = Lambda_c T_cX
    q^{s_c - s_X} D_{s_c}/D_{s_X}, N_cc from the closed diagonal and N_cX from the chain relation
        N_cX (Lambda_c - Lambda_X) = N_cc B_cX + sum_{X < Y < c} N_cY B_YX.
    Returns (pairs, T, B, N, undefined, inconsistent) with dicts keyed by (c, X)."""
    pairs = [c for c in channels_present(nmax) if pair_info(c) is not None]
    info = {c: pair_info(c) for c in pairs}
    below = [(c, X) for c in pairs for X in pairs
             if sum(info[c]["U"]) > sum(info[X]["U"]) and I.contains(info[c]["U"], info[X]["U"])]
    below.sort(key=lambda cx: sum(info[cx[0]]["U"]) - sum(info[cx[1]]["U"]))
    T, Bf, Nf = {}, {}, {}
    undefined, inconsistent = [], []
    for c, X in below:
        if sum(info[c]["U"]) - sum(info[X]["U"]) == 1:
            r = adjacent_residue(c, X, F, A, q)
            if r is None:
                undefined.append((c, X))
                continue
            T[(c, X)] = r * q ** (2 * info[X]["s"])
        else:
            t, ok = root_T(c, X, T, F, q, pairs)
            if t is None:
                undefined.append((c, X))
                continue
            if not ok:
                inconsistent.append((c, X))
            T[(c, X)] = t
        Bf[(c, X)] = (Lam(c, F, A, q) * T[(c, X)] * q ** (info[c]["s"] - info[X]["s"])
                      * Dk(info[c]["s"], F, A, q) / Dk(info[X]["s"], F, A, q))
    for c in pairs:
        Nf[(c, c)] = khook_N(c, c, F, A, q)
    for c, X in below:
        if (c, X) not in Bf:
            continue
        s = Nf[(c, c)] * Bf[(c, X)]
        for Y in pairs:
            if (c, Y) in Bf and (Y, X) in Bf:
                if (c, Y) not in Nf:
                    s = None
                    break
                s = s + Nf[(c, Y)] * Bf[(Y, X)]
        if s is None:
            undefined.append((c, X))
            continue
        Nf[(c, X)] = s / (Lam(c, F, A, q) - Lam(X, F, A, q))
    return pairs, T, Bf, Nf, undefined, inconsistent


def check_roots(nmax=6, seed=1):
    """Every pair column X of every one-cell pair row c of N and B (U_X inside U_c), at every gap, from the closed
    formulas: adjacent residues at gap one, the root formula (root_T) at gap >= 2, N by the chain relation.  The
    quotient-ring form must return a constant (consistency) and agree with the engine."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    orderB, B = build_B(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    pairs, T, Bf, Nf, undefined, inconsistent = closed_pair_block(nmax, F, A, q)
    cleanset = set(clean)
    counts, zeros, bad = {}, {}, []
    for (c, X), v in Bf.items():
        if c not in cleanset or X not in cleanset:
            continue
        g = sum(pair_info(c)["U"]) - sum(pair_info(X)["U"])
        counts[g] = counts.get(g, 0) + 1
        vB, vN = B[c].get(X, F.zero), N[pos[c]][pos[X]]
        zeros[g] = zeros.get(g, 0) + (vB == 0)
        if vB != v or (c, X) not in Nf or Nf[(c, X)] != vN:
            bad.append((c, X))
    und = [cx for cx in undefined if cx[0] in cleanset and cx[1] in cleanset]
    return not (bad or und or inconsistent), [
        "roots: %d one-cell pairs (|R| <= %d, %d clean); pair-pair entries of B and N by gap: %s (zeros %s) from the "
        "adjacent residues and the root formula; non-constant %s; undefined %s; mismatches %s"
        % (len(pairs), nmax, len([c for c in pairs if c in cleanset]),
           " ".join("g%d:%d" % kv for kv in sorted(counts.items())), " ".join("%d" % zeros[g] for g in sorted(zeros)),
           "none" if not inconsistent else inconsistent[:3], "none" if not und else und[:3],
           "none" if not bad else bad[:4])]


def fourrow_merged(c, mu, F, A, q):
    """N_{c mu} + sum_{pairs X: U_X = mu} N_cX for a one-cell pair c and a single column mu inside U_c, from the four
    closed single rows (eq. fourrow)."""
    info = pair_info(c)
    Z, Zp, U, V = c[1], c[2], info["U"], info["V"]
    cq = q ** info["d"] - q ** (-info["d"])
    den = Dk(2 * cont(info["b"]), F, A, q) * Dk(2 * cont(info["bp"]), F, A, q)
    v = Dk(info["s"], F, A, q) ** 2 * (binom_N(Z, mu, F, A, q) + binom_N(Zp, mu, F, A, q))
    v = v - cq * cq * (binom_N(U, mu, F, A, q) + binom_N(V, mu, F, A, q))
    return v / (2 * den)


def check_fullrow(nmax=6, seed=1):
    """The whole row of a one-cell pair in N and in B from the closed formulas, every column: single columns of N
    from the four-row formula minus the pair entries of the root formula at the resonant columns (N_{c mu} =
    fourrow_merged - sum_{U_X = mu} N_cX), pair columns of N from the root formula, and then every single column of B
    from N B = Lambda N,
        B_{c mu} = ( Lambda_c N_{c mu} - sum_{nu single} N_{c nu} B^KNTZ_{nu mu} - sum_{Y pair < c} N_cY B_{Y mu} ) / N_cc,
    the pair rows Y of smaller union first.  N_{c U_c} = B_{c U_c} = 0 come out automatically.  Compared with the
    engine on every column of every clean one-cell pair."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    orderB, B = build_B(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    pairs, T, Bf, Nf, undefined, inconsistent = closed_pair_block(nmax, F, A, q)
    info = {c: pair_info(c) for c in pairs}
    singles = [X[1] for X in order if X[0] == "s" and X != VAC]
    Nrow, Brow = {}, {}
    for c in sorted(pairs, key=lambda c: (sum(info[c]["U"]), c)):
        U = info[c]["U"]
        cols = [mu for mu in singles if I.contains(U, mu)] + [()]
        Nc = {}
        for mu in cols:
            v = fourrow_merged(c, mu, F, A, q)
            for X in pairs:
                if info[X]["U"] == mu and (c, X) in Nf:
                    v = v - Nf[(c, X)]
            Nc[mu] = v
        Ncc = Nf[(c, c)]
        Lc = Lam(c, F, A, q)
        Bc = {}
        for mu in cols:
            s = Lc * Nc[mu]
            for nu in cols:
                if I.contains(nu, mu) and Nc[nu] != 0:
                    s = s - Nc[nu] * (kntz_B(nu, mu, F, A, q) if nu != () else (F.one if mu == () else F.zero))
            for Y in pairs:
                if (c, Y) in Nf and Y in Brow and mu in Brow[Y]:
                    s = s - Nf[(c, Y)] * Brow[Y][mu]
            Bc[mu] = s / Ncc
        Nrow[c], Brow[c] = Nc, Bc
    cleanset = set(clean)
    nN = nB = 0
    bad = []
    for c in pairs:
        if c not in cleanset:
            continue
        for mu, v in Nrow[c].items():
            X = VAC if mu == () else ("s", mu, mu)
            nN += 1
            if N[pos[c]][pos[X]] != v:
                bad.append((c, mu, "N"))
        for mu, v in Brow[c].items():
            X = VAC if mu == () else ("s", mu, mu)
            nB += 1
            if B[c].get(X, F.zero) != v:
                bad.append((c, mu, "B"))
    return not (bad or inconsistent), [
        "fullrow: %d clean one-cell pairs (|R| <= %d): every single column of N (%d entries, resonant columns and "
        "U_c included) from the four-row formula and the root formula, and every single column of B (%d entries) from "
        "N B = Lambda N with the KNTZ single rows: mismatches %s"
        % (len([c for c in pairs if c in cleanset]), nmax, nN, nB, "none" if not bad else bad[:4])]


# ---------------------------------------------------------------------------
# all pair rows on single columns: the sign-flip kernel (one, two and more moved cells)
# ---------------------------------------------------------------------------
def pair_cells(c):
    """(V, X, Y) for a pair c = {Z, Z'}: V = Z cap Z' and the moved cells X = Z minus V, Y = Z' minus V (as sorted lists)."""
    Z, Zp = c[1], c[2]
    n = max(len(Z), len(Zp))
    Zz, Zpz = tuple(Z) + (0,) * (n - len(Z)), tuple(Zp) + (0,) * (n - len(Zp))
    V = tuple(v for v in (min(a, b) for a, b in zip(Zz, Zpz)) if v)
    bV = set(boxes(V))
    return V, sorted(b for b in boxes(Z) if b not in bV), sorted(b for b in boxes(Zp) if b not in bV)


def orient(cells):
    """'row' / 'col' for cells on one line (a domino), None otherwise or for a single cell."""
    if len(cells) < 2:
        return None
    if len(set(b[0] for b in cells)) == 1:
        return "row"
    if len(set(b[1] for b in cells)) == 1:
        return "col"
    return None


def pair_kind(c):
    """'1-cell', 'n-cell row/row', 'n-cell col/col', 'n-cell domino/separate', 'n-cell separate/separate' or
    'n-cell resonant' (a vertical against a horizontal domino: Lambda_c equals the eigenvalue of a single)."""
    V, X, Y = pair_cells(c)
    n = len(X)
    if n == 1:
        return "1-cell"
    oX, oY = orient(X), orient(Y)
    if oX and oY:
        return "%d-cell resonant" % n if oX != oY else "%d-cell %s/%s" % (n, oX, oY)
    if oX or oY:
        return "%d-cell domino/separate" % n
    return "%d-cell separate/separate" % n


def kernel_K(c, sig, F, A, q):
    """The sign-flip kernel K_c(sigma) of a pair c = {V+X, V+Y} (n = |X| = |Y| moved cells on each side) for the sign
    pattern sig (dict cell -> +-1), with u_e = z_e^{sig_e}, z_e = A q^{2c(e)}, {t} = t - 1/t:
        K = (-1)^n prod_{(x,y) in P} {sqrt(u_y/u_x)} / prod_{e in X u Y} {u_e}
              * prod_{a<b in X} C(u_a,u_b) * prod_{a<b in Y} C(u_a,u_b),      C(u_a,u_b) = {xi sqrt(u_a u_b)}/{sqrt(u_a u_b)},
    xi = q^{-1} for a horizontal domino of c, q for a vertical one (the same on both sides; 1 without a domino, untested),
    P = ends(X) x Y (for n = 1 the cell of X twice: the kernel of the four-row formula) when n <= 2 or no or all cells
    of X carry +, P = X x ends(Y) when no or all cells of Y carry +; None for the middle states of n >= 3, which are
    not given by a product.  All square roots are monomials."""
    V, X, Y = pair_cells(c)
    n = len(X)
    xi = 0
    for S in (X, Y):
        o = orient(S)
        if o:
            xi = -1 if o == "row" else 1
            break
    k, kp = sum(1 for x in X if sig[x] == 1), sum(1 for y in Y if sig[y] == 1)
    ends = lambda S: (min(S, key=cont), max(S, key=cont))
    if n <= 2 or k in (0, n):
        P = [(x, y) for x in ends(X) for y in Y]
    elif kp in (0, n):
        P = [(x, y) for x in X for y in ends(Y)]
    else:
        return None
    curly_ = lambda t: t - F.one / t
    v = F.one if n % 2 == 0 else -F.one
    for x, y in P:                                   # sqrt(u_y/u_x) = A^{(s_y - s_x)/2} q^{s_y c(y) - s_x c(x)}
        v = v * curly_(A ** ((sig[y] - sig[x]) // 2) * q ** (sig[y] * cont(y) - sig[x] * cont(x)))
    for e in X + Y:
        v = v / curly_((A * q ** (2 * cont(e))) ** sig[e])
    for S in (X, Y):
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                a, b = S[i], S[j]
                s = A ** ((sig[a] + sig[b]) // 2) * q ** (sig[a] * cont(a) + sig[b] * cont(b))
                v = v * curly_(q ** xi * s) / curly_(s)
    return v


def kernel_states(c):
    """[(sig, lambda(sig))] over the sign patterns of the moved cells of c whose state V + {e: sig_e = +} is a diagram."""
    import itertools
    V, X, Y = pair_cells(c)
    cells = X + Y
    out = []
    for signs in itertools.product((1, -1), repeat=len(cells)):
        sig = dict(zip(cells, signs))
        lam = add_cells(V, [e for e in cells if sig[e] == 1])
        if lam is not None:
            out.append((sig, lam))
    return out


def check_kernel(nmax=6, seed=1):
    """Every single column mu of every pair row of N (merged basis: 2 N_{c mu} + 2 sum_{pairs W: U_W = mu} N_cW) from the
    sign-flip kernel over the states of the moved cells,  2 R_c(mu) = sum_sigma K_c(sigma) N_{lambda(sigma) mu},  with the
    closed single rows: one-cell pairs (the four-row formula) and the two-cell pairs that are not resonant.  The resonant
    two-cell pairs (vertical against horizontal domino) are listed, not checked: their rows are not of this form."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    unions = {}
    for X in order:
        if X[0] == "p":
            unions.setdefault(I.union(X[1], X[2]), []).append(X)
    counts, cols, bad, skipped, middle = {}, {}, [], [], []
    for c in clean:
        if c[0] != "p":
            continue
        kind = pair_kind(c)
        if "resonant" in kind:
            skipped.append(c)
            continue
        U = I.union(c[1], c[2])
        states = kernel_states(c)
        Ks = [(kernel_K(c, sig, F, A, q), lam) for sig, lam in states]
        if any(K is None for K, lam in Ks):
            middle.append(c)
            continue
        counts[kind] = counts.get(kind, 0) + 1
        for X in order:
            if X[0] != "s" or not I.contains(U, X[1]):
                continue
            mu = X[1]
            lhs = 2 * N[pos[c]][pos[X]]
            for W in unions.get(mu, []):
                if W in pos:
                    lhs = lhs + 2 * N[pos[c]][pos[W]]
            rhs = F.zero
            for K, lam in Ks:
                if I.contains(lam, mu):
                    rhs = rhs + K * binom_N(lam, mu, F, A, q)
            cols[kind] = cols.get(kind, 0) + 1
            if lhs != rhs:
                bad.append((c, mu))
    return not bad, ["kernel: pair rows of N on single columns (|R| <= %d) from the sign-flip kernel: %s; resonant two-cell "
                     "pairs not covered: %d; pairs with middle states (n >= 3) not checked: %d; mismatches %s"
                     % (nmax, ", ".join("%s %d pairs / %d columns" % (k, counts[k], cols.get(k, 0)) for k in sorted(counts)),
                        len(skipped), len(middle), "none" if not bad else bad[:4])]


def rho_top(R, F, A, q):
    """rho(R) = (-A)^{|R|} q^{kappa_R} prod_{b in R} {q^{h(b)}}^2 / D_{c(b)}, the common value of w_c(R)/d_c."""
    v = (-A) ** sum(R) * q ** kappa(R)
    for b in boxes(R):
        v = v * curly(q ** hook(R, b)) ** 2 / Dk(cont(b), F, A, q)
    return v


def check_topweight(nmax=6, seed=1):
    """Top-weight law: every clean channel c whose only minimal diagram is R (the top channels of R, Z u Z' = R, and
    the resonant pairs with R_min = R) has the evolution coefficient w_c(R) = F_c(R)/phi_c(-1) = m_c(R) d_c rho(R), with
    one closed rho(R) for all of them (no channel above c is present at R, so only F_c contributes) and m_c(R) the
    multiplicity of the composite of c in R (x) Rbar (1 here; first 2 at [4,3,2,1], see check_mult)."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    clean, excluded = clean_channels(pt, channels_present(nmax))
    counts, mults, bad = {}, {}, []
    for c in clean:
        rm = I.rmin(c)
        if len(rm) != 1:
            continue
        R = rm[0]
        kind = "single" if c[0] == "s" else pair_kind(c)
        counts[kind] = counts.get(kind, 0) + 1
        m = I.presence(c, R)
        mults[m] = mults.get(m, 0) + 1
        if pt.F_c(c, R) / pt.phi(c, -1) != m * pt.d(c) * rho_top(R, F, A, q):
            bad.append(c)
    return not bad, ["topweight: w_c(R_min) = m_c(R_min) d_c rho(R_min) for every clean channel with one minimal "
                     "diagram, |R| <= %d: %s (multiplicities %s); mismatches %s"
                     % (nmax, ", ".join("%s %d" % (k, counts[k]) for k in sorted(counts)),
                        ", ".join("m=%d: %d" % (m, mults[m]) for m in sorted(mults)), "none" if not bad else bad[:4])]


def minimal_pairs(n):
    """(U, c) for every pair channel c = {Z, Z'} and every minimal diagram U of c with |U| = n, with m_c(U) > 0."""
    out = []
    for U in I.partitions(n):
        subs = subdiagrams(U)
        bysize = {}
        for Z in subs:
            bysize.setdefault(sum(Z), []).append(Z)
        for Zs in bysize.values():
            for i, Z in enumerate(Zs):
                for Zp in Zs[i + 1:]:
                    c = I.pair(Z, Zp)
                    if not I.presence(c, U):
                        continue
                    base = I.union(Z, Zp)
                    if not any(S != U and I.contains(S, base) and I.presence(c, S) for S in subs):
                        out.append((U, c))
    return out


def check_mult(nmax=6, seed=1):
    """Multiplicity factor of the top-weight law (pure combinatorics, seed unused): for every pair channel c = {Z, Z'}
    and every minimal diagram U of c with |U| <= max(nmax, 12), the Koike multiplicity m_c(U) = <s_{U/Z}, s_{U/Z'}> of
    the composite in U (x) Ubar is 1, except for the separate/separate pairs: every two-cell pair whose sides are two
    separate cells each has m_c(U) = 2 (U/Z and U/Z' are two separate corners of U, so m = <h_1^2, h_1^2>; first at
    U = [4,3,2,1]), and some three-cell ones do (from 13 boxes).  Singles have m = 1 at R_min = lambda."""
    n_max = max(nmax, 12)
    counts, first, bad = {}, {}, []
    for n in range(2, n_max + 1):
        for U, c in minimal_pairs(n):
            m = I.presence(c, U)
            kind = pair_kind(c)
            sep = kind.endswith("separate/separate")
            if m > 2 or (m == 2 and not sep) or (kind == "2-cell separate/separate" and m != 2):
                bad.append((U, c, m))
            key = (kind, m) if m > 1 or sep else ("other", 1)
            counts[key] = counts.get(key, 0) + 1
            if m > 1 and kind not in first:
                first[kind] = U
    return not bad, ["mult: m_c(U) at the minimal diagrams of all pair channels, |U| <= %d: %s; first m = 2 at %s; "
                     "mismatches %s" % (n_max, ", ".join("%s m=%d: %d" % (k, m, counts[(k, m)]) for k, m in sorted(counts)),
                                        ", ".join("%s (%s)" % (U, k) for k, U in sorted(first.items())),
                                        "none" if not bad else bad[:4])]


def closed_N(nmax, F, A, q):
    """Rows {c: {X: N_cX}} of the binomial matrix for every single and one-cell pair channel present at |R| <= nmax,
    from the closed formulas only: the tableau formula (singles); the four-row formula minus the root-formula pair
    entries (single columns), the root formula with the chain relation (pair columns) and the closed diagonal
    (one-cell pairs).  No vanishing condition is solved."""
    pairs, T, Bf, Nf, undefined, inconsistent = closed_pair_block(nmax, F, A, q)
    info = {c: pair_info(c) for c in pairs}
    lab = lambda mu: VAC if not mu else ("s", mu, mu)
    rows = {VAC: {VAC: F.one}}
    for c in channels_present(nmax):
        if c[0] == "s":
            rows[c] = {lab(mu): binom_N(c[1], mu, F, A, q) for mu in subdiagrams(c[1])}
        elif c in info:
            row = {}
            for mu in subdiagrams(info[c]["U"]):
                v = fourrow_merged(c, mu, F, A, q)
                for X in pairs:
                    if info[X]["U"] == mu and (c, X) in Nf:
                        v = v - Nf[(c, X)]
                row[lab(mu)] = v
            for X in pairs:
                if (c, X) in Nf:
                    row[X] = Nf[(c, X)]
            rows[c] = row
    return rows, undefined, inconsistent


def closed_H(R, ms, pt, rows):
    """H_R(Tw_m) = 1 + sum_c d_c N_cc Psi_c(R) Phi_c(m), Psi = N^{-1} psi(R), Phi(m) = N^{-1} (Lambda^m - 1), from the
    rows of N alone; None when a channel below R has no closed row."""
    F, A, q = pt.F, pt.A, pt.q
    chans = I.channels_of(R)
    S, stack = {VAC}, list(chans)
    while stack:
        c = stack.pop()
        if c in S:
            continue
        if c not in rows:
            return None
        S.add(c)
        stack.extend(X for X in rows[c] if X not in S)
    Et = {VAC: {VAC: F.one}}
    for c in _order([c for c in S if c != VAC])[1:]:
        acc = {c: F.one}
        for Y, v in rows[c].items():
            if Y != c and v != 0:
                for X, w in Et[Y].items():
                    acc[X] = acc.get(X, F.zero) - v * w
        Ncc = rows[c][c]
        Et[c] = {X: w / Ncc for X, w in acc.items() if w != 0}
    out = {}
    Psi = {c: sum((w * pt.psi(X, R) for X, w in Et[c].items() if X != VAC), F.zero) for c in chans}
    for m in ms:
        tot = F.one
        for c in chans:
            Phi = sum((w * (Lam(X, F, A, q) ** m - 1) for X, w in Et[c].items() if X != VAC), F.zero)
            tot = tot + pt.d(c) * rows[c][c] * Psi[c] * Phi
        out[m] = tot
    return out


def check_closed(nmax=6, seed=1, ms=range(-5, 6)):
    """End to end from the closed formulas only: for every R with |R| <= nmax whose channels (and all channels below
    them) are singles and one-cell pairs, H_R(Tw_m) from closed_N / closed_H against every stored polynomial (ten twist
    knots at |R| <= 6, 4_1 at seven and eight boxes, the symmetric tables) and against Rosso--Jones T(2,3) = Tw_1."""
    from homfly.methods.rosso_jones import RossoJones
    from homfly.knots.families import TorusKnot
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    rows, undefined, inconsistent = closed_N(nmax, F, A, q)
    RJ = RossoJones()
    n_R, n_val, n_rj, bad, open_R = 0, 0, 0, [], 0
    for k in range(1, nmax + 1):
        for R in partitions(k):
            R = tuple(R)
            vals = closed_H(R, [m for m in ms if m], pt, rows)
            if vals is None:
                open_R += 1
                continue
            n_R += 1
            for m, v in vals.items():
                sym = len(R) == 1 and not has_data(M_KNOT[m], R) and k >= 7
                if has_data(M_KNOT[m], R, symmetric=sym):
                    n_val += 1
                    if v != twist_data(m, R, F, A, q, symmetric=sym):
                        bad.append((R, m))
            n_rj += 1
            if vals[1] != RJ.evaluate(TorusKnot(2, 3), R, F, A, q):
                bad.append((R, "RJ"))
    return not (bad or inconsistent), [
        "closed: %d diagrams |R| <= %d entirely closed (singles and one-cell pairs; %d others need multi-cell rows): "
        "H_R(Tw_m) from the closed rows of N only vs %d stored values and %d Rosso-Jones values: mismatches %s"
        % (n_R, nmax, open_R, n_val, n_rj, "none" if not bad else bad[:4])]


def pi_vac(R, F, A, q):
    """pi(R) = prod_{b in R} 1 / (1 - A^{-2} q^{-2c(b)}) = prod_b A q^{c(b)} / D_{c(b)}."""
    v = F.one
    for b in boxes(R):
        v = v * A * q ** cont(b) / Dk(cont(b), F, A, q)
    return v


def check_vacuum(nmax=6, seed=1):
    """The vacuum law: E~_{c,vac} = -sum_{X != vac} E_cX / phi_c(-1) (the constant term of Phi_c(m) in the eigenvalue
    expansion) equals pi(R_min(c)) for every channel c with one minimal diagram; and rho(R) = (-1)^{|R|} pi(R)
    prod_b {q^{h(b)}}^2 (the top-weight law and the vacuum law carry the same R_min product)."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    clean, excluded = clean_channels(pt, channels_present(nmax))
    n, bad = 0, []
    for c in clean:
        rm = I.rmin(c)
        if len(rm) != 1:
            continue
        R = rm[0]
        Etv = -sum(pt.E_row(c).values(), F.zero) / pt.phi(c, -1)
        n += 1
        if Etv != pi_vac(R, F, A, q):
            bad.append(c)
    rel_bad = []
    for k in range(1, nmax + 1):
        for R in partitions(k):
            R = tuple(R)
            h = F.one
            for b in boxes(R):
                h = h * curly(q ** hook(R, b)) ** 2
            if rho_top(R, F, A, q) != (-1) ** k * pi_vac(R, F, A, q) * h:
                rel_bad.append(R)
    return not (bad or rel_bad), ["vacuum: E~_{c,vac} = pi(R_min(c)) for %d channels with one minimal diagram "
                                  "(|R| <= %d): mismatches %s; rho = (-1)^|R| pi prod {q^h}^2: %s"
                                  % (n, nmax, "none" if not bad else bad[:4], "ok" if not rel_bad else rel_bad[:4])]


# --- the non-resonant two-cell pair rows: diagonal, first pair column, partners, X*, zero pattern ------------------
def side_kind(cells):
    """'h' / 'v' for a domino (two cells in one row / column), 's' for two separate cells, '1' for one cell, 'n' else."""
    if len(cells) != 2:
        return "1" if len(cells) == 1 else "n"
    a, b = sorted(cells, key=cont)
    if a[0] == b[0] and abs(a[1] - b[1]) == 1:
        return "h"
    if a[1] == b[1] and abs(a[0] - b[0]) == 1:
        return "v"
    return "s"


def twocell_class(c):
    """'row/row', 'col/col', 'dom/sep' for the non-resonant two-cell pairs, None otherwise."""
    if c[0] != "p":
        return None
    V, X, Y = pair_cells(c)
    if len(X) != 2:
        return None
    kx, ky = side_kind(X), side_kind(Y)
    if (kx, ky) == ("h", "h"):
        return "row/row"
    if (kx, ky) == ("v", "v"):
        return "col/col"
    if "s" in (kx, ky) and (kx in "hv" or ky in "hv"):
        return "dom/sep"
    return None


def pair_diag(c, F, A, q):
    """The diagonal N_cc = phi_c(-1) of a pair with n moved cells per side (one-cell, non-resonant two-cell, lines):
        N_cc = (-1)^n / 2 (-A)^{-|U|} q^{-kappa_U} prod_{x in ends(X), y in Y} {q^{c(y)-c(x)}} prod_{v in V} D_{idx_U(v)+delta(v)},
    ends(X) = the two extreme cells of X by content (for n = 1 the cell twice: {q^d}^2), delta = 0 except at the hinge
    (row of the upper-right, column of the lower-left cell) of the separate side of a domino/separate pair, where it is
    -1 (horizontal domino) or +1 (vertical domino)."""
    V, X, Y = pair_cells(c)
    n = len(X)
    U = I.union(c[1], c[2])
    sk = (side_kind(X), side_kind(Y))
    xi = -1 if "h" in sk else (1 if "v" in sk else 0)
    hinge = {}
    for cells, k in ((X, sk[0]), (Y, sk[1])):
        if k == "s":
            a, b = sorted(cells, key=cont)
            hinge[(b[0], a[1])] = xi
    ends = (min(X, key=cont), max(X, key=cont))
    v = (F.one if n % 2 == 0 else -F.one) / 2 * (-A) ** (-sum(U)) * q ** (-kappa(U))
    for x in ends:
        for y in Y:
            v = v * curly(q ** (cont(y) - cont(x)))
    for b in boxes(V):
        v = v * Dk(idx(U, b) + hinge.get(b, 0), F, A, q)
    return v


def first_pair_column(c, F, A, q):
    """N_{c,{[1,1],[2]}} = Ebar_1(c)^2 / (2 A^3 D_0),  Ebar_1 = sum_{e in Z} (z_e + 1/z_e) - sum_{e in Z'} (z_e + 1/z_e)."""
    Eb = F.zero
    for lam, sg in ((c[1], 1), (c[2], -1)):
        for e in boxes(lam):
            z = A * q ** (2 * cont(e))
            Eb = Eb + sg * (z + 1 / z)
    return Eb * Eb / (2 * A ** 3 * Dk(0, F, A, q))


def _is_diagram(cells):
    S = set(cells)
    return all((i == 0 or (i - 1, j) in S) and (j == 0 or (i, j - 1) in S) for i, j in S)


def _shape(cells):
    rows = {}
    for i, j in cells:
        rows[i] = rows.get(i, 0) + 1
    return tuple(rows[i] for i in sorted(rows))


def twocell_partners(c):
    """[(type, r, m, x', y', X)]: the one-cell pairs X of the same weight as the two-cell pair c = {V+X_c, V+Y_c} obtained
    by removing a moved cell r of one side and adding a moved cell m of the other side to it (type I: r in Y_c, m in X_c;
    type II: r in X_c, m in Y_c); x', y' are the cells of X_c, Y_c that are neither r nor m."""
    V, Xc, Yc = pair_cells(c)
    Z, Zp = set(boxes(c[1])), set(boxes(c[2]))
    out = []
    for tp, S1, S2 in (("I", Yc, Xc), ("II", Xc, Yc)):
        for r in S1:
            for m in S2:
                Zn, Zpn = (Z, (Zp - {r}) | {m}) if tp == "I" else ((Z - {r}) | {m}, Zp)
                if not (_is_diagram(Zn) and _is_diagram(Zpn)):
                    continue
                xp = [x for x in Xc if x not in (r, m)][0]
                yp = [y for y in Yc if y not in (r, m)][0]
                out.append((tp, r, m, xp, yp, I.pair(_shape(Zn), _shape(Zpn))))
    return out


def partner_B(c, part, F, A, q):
    """(P) B_cX / Lambda_c for a partner X of a non-resonant two-cell pair c (None if c has no boundary domino):
        sigma A q^{c(m)} {q}^{-1} Q Dfac,  sigma = +1 (type I), -1 (type II); with the D-side the side whose cells are a
        domino on the boundary (eps = +1 horizontal in row 0, -1 vertical in column 0), F-side the other one, s' and d'
        the cells of the F- and D-side that stay:
        r on F:  Q = [c(r)-c(s')+eps] / ([c(r)-c(s')][c(r)-c(d')]),  Dfac = D_{2c(d')} D_{c(d')+al} / D_{c(d')+c(r)+al},
        m on F:  Q = [c(m)-c(s')-eps] / ([c(m)-c(s')][c(m)-c(d')]),  Dfac = D_{2c(m)-eps t} D_{c(m)+eps t} / D_{c(m)+c(d')-eps},
    al = 0, t = 1 for two separate F-cells; for an F-domino with f1 its cell next to V: t = 1 + column(f1) (horizontal) or
    1 + row(f1) (vertical), al = -3 eps when that coordinate is > 0.  q-numbers of content differences are taken with |n|."""
    tp, r, m, xp, yp, X = part
    V, Xc, Yc = pair_cells(c)
    kx, ky = side_kind(Xc), side_kind(Yc)
    onb = lambda cells, k: all(i == 0 for i, j in cells) if k == "h" else all(j == 0 for i, j in cells)
    cand = [(nm, cells, k) for nm, cells, k in (("x", Xc, kx), ("y", Yc, ky)) if k in "hv" and onb(cells, k)]
    if len(cand) != 1:
        return None
    dnm, Dcells, dk = cand[0]
    eps = 1 if dk == "h" else -1
    Fcells = Yc if dnm == "x" else Xc
    fk = side_kind(Fcells)
    sp, dp = (yp, xp) if dnm == "x" else (xp, yp)
    al, t = 0, 1
    if fk in "hv":
        lo = min(Fcells, key=lambda b: (b[1], b[0]) if fk == "h" else (b[0], b[1]))
        coord = lo[1] if fk == "h" else lo[0]
        t, al = 1 + coord, (-3 * eps if coord > 0 else 0)
    if r in Fcells:
        a1, a2, a3 = cont(r) - cont(sp) + eps, cont(r) - cont(sp), cont(r) - cont(dp)
        Dn, Dd = [2 * cont(dp), cont(dp) + al], [cont(dp) + cont(r) + al]
    else:
        a1, a2, a3 = cont(m) - cont(sp) - eps, cont(m) - cont(sp), cont(m) - cont(dp)
        Dn, Dd = [2 * cont(m) - eps * t, cont(m) + eps * t], [cont(m) + cont(dp) - eps]
    if a1 == 0:
        return F.zero
    v = (F.one if tp == "I" else -F.one) * A * q ** cont(m) / curly(q)
    v = v * qnum(abs(a1), F, q) / (qnum(abs(a2), F, q) * qnum(abs(a3), F, q))
    for k in Dn:
        v = v * Dk(k, F, A, q)
    for k in Dd:
        v = v / Dk(k, F, A, q)
    return v


def xstar_B(c, F, A, q):
    """(P*) the fifth top pair column of a domino/separate pair c: X* = {V+d1+s1, V+d1+s2} (d1 the domino cell next to V,
    d2 the other one, s1, s2 the separate cells) and
        B_cX*/Lambda_c = sigma A q^{c(d1)} {q}^{-1} [k-1][k+1] D_{c(d2)} / prod_{d in {d1,d2}, s in {s1,s2}} [c(d)-c(s)],
    sigma = +1 if the domino is on the Z side of c, -1 on the Z' side, k = |c(s1) - c(s2)|.  Returns (X*, value) or None."""
    if twocell_class(c) != "dom/sep":
        return None
    V, Xc, Yc = pair_cells(c)
    dside = "x" if side_kind(Xc) in "hv" else "y"
    Dc, Sc = (Xc, Yc) if dside == "x" else (Yc, Xc)
    bV = set(boxes(V))
    for d1 in Dc:
        if not _is_diagram(bV | {d1}):
            continue
        A1, B1 = bV | {d1, Sc[0]}, bV | {d1, Sc[1]}
        if not (_is_diagram(A1) and _is_diagram(B1)):
            continue
        d2 = [e for e in Dc if e != d1][0]
        k = abs(cont(Sc[0]) - cont(Sc[1]))
        Xs = I.pair(_shape(A1), _shape(B1))
        if k <= 1:
            return Xs, F.zero
        v = (F.one if dside == "x" else -F.one) * A * q ** cont(d1) / curly(q)
        v = v * qnum(k - 1, F, q) * qnum(k + 1, F, q) * Dk(cont(d2), F, A, q)
        for d in (d1, d2):
            for s in Sc:
                v = v / qnum(abs(cont(d) - cont(s)), F, q)
        return Xs, v
    return None


def twocell_zero(c, Y):
    """(Z) N_cY = 0 predicted for a non-resonant two-cell pair c: a single containing >= 3 of the 4 moved cells of c; a
    one-cell pair whose V contains >= 2 moved cells of c; a resonant two-cell pair; a two-cell pair not contained in c
    side by side in either orientation; every pair with >= 3 moved cells per side."""
    V, Xc, Yc = pair_cells(c)
    mv = set(Xc) | set(Yc)
    if Y[0] == "s":
        return Y != VAC and len(set(boxes(Y[1])) & mv) >= 3
    VY, XY, YY = pair_cells(Y)
    if len(XY) == 1:
        return len(set(boxes(VY)) & mv) >= 2
    if len(XY) == 2:
        if twocell_class(Y) is None:
            return True
        Zc, Zpc, Zw, Zpw = set(boxes(c[1])), set(boxes(c[2])), set(boxes(Y[1])), set(boxes(Y[2]))
        return not ((Zw <= Zc and Zpw <= Zpc) or (Zw <= Zpc and Zpw <= Zc))
    return True


def check_twocell(nmax=6, seed=1):
    """The non-resonant two-cell pair rows of N and B (and the pair diagonals and first pair columns of all pairs):
    (D) N_cc by pair_diag for every clean pair (one-cell and non-resonant two-cell); the first pair column
    N_{c,{[1,1],[2]}} = Ebar_1(c)^2/(2A^3 D_0) for the same pairs; for the two-cell rows (P) B_cX at every partner, (P*) at
    X* (domino/separate rows), and the zero pattern (Z) of the N row, against N, B built from the engine rows."""
    F, A, q = random_point(seed)
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    _, B = build_B(pt, chans)
    first = ("p", (1, 1), (2,))
    cnt = {"diag": 0, "first": 0, "P": 0, "P*": 0, "zero": 0, "rows": 0}
    bad = []
    for c in clean:
        if c[0] != "p":
            continue
        kind = pair_kind(c)
        if kind != "1-cell" and twocell_class(c) is None:
            continue
        i = pos[c]
        cnt["diag"] += 1
        if N[i][i] != pair_diag(c, F, A, q):
            bad.append(("diag", c))
        if c != first:
            cnt["first"] += 1
            if N[i][pos[first]] != first_pair_column(c, F, A, q):
                bad.append(("first", c))
        if kind == "1-cell":
            continue
        cnt["rows"] += 1
        L = Lam(c, F, A, q)
        for part in twocell_partners(c):
            X = part[5]
            val = partner_B(c, part, F, A, q)
            if val is None or X not in pos:
                continue
            cnt["P"] += 1
            if B[c].get(X, F.zero) != L * val:
                bad.append(("P", c, X))
        xs = xstar_B(c, F, A, q)
        if xs is not None and xs[0] in pos:
            cnt["P*"] += 1
            if B[c].get(xs[0], F.zero) != L * xs[1]:
                bad.append(("P*", c, xs[0]))
        for Y in order[:i]:
            if Y != VAC and twocell_zero(c, Y):
                cnt["zero"] += 1
                if N[i][pos[Y]] != 0:
                    bad.append(("Z", c, Y))
    return not bad, ["twocell: |R| <= %d: diagonal (D) %d pairs, first pair column %d pairs, two-cell rows %d: partners "
                     "(P) %d, X* (P*) %d, predicted zeros (Z) %d; mismatches %s"
                     % (nmax, cnt["diag"], cnt["first"], cnt["rows"], cnt["P"], cnt["P*"], cnt["zero"],
                        "none" if not bad else bad[:4])]


def lagrange_family(c):
    """Interpolation families of (L): {'h', 'v'} for the vacuum, the singles, the one-cell pairs and the separate/separate
    two-cell pairs; {'h'} for row/row pairs and domino/separate pairs with a horizontal domino; {'v'} for col/col pairs
    and those with a vertical domino; set() for the resonant pairs and the pairs with three or more moved cells."""
    if c == VAC or c[0] == "s":
        return {"h", "v"}
    kind = pair_kind(c)
    if kind in ("1-cell", "2-cell separate/separate"):
        return {"h", "v"}
    if twocell_class(c) is None:
        return set()
    V, X, Y = pair_cells(c)
    return {side_kind(X)} if side_kind(X) in ("h", "v") else {side_kind(Y)}


def weight(c):
    return 0 if c == VAC else sum(c[1])


def lagrange_nodes(w):
    """{xi: Omega^xi_w}: all channels of weight <= w in the family xi (every pair {Z, Z'}, |Z| = |Z'| <= w, and single)."""
    out = {"h": [VAC], "v": [VAC]}
    for k in range(1, w + 1):
        ps = list(partitions(k))
        for c in [I.single(mu) for mu in ps] + [I.pair(Z, Zp) for i, Z in enumerate(ps) for Zp in ps[i + 1:]]:
            for xi in sorted(lagrange_family(c)):
                out[xi].append(c)
    return out


def pi_monomials(w):
    """The monomials prod_i Obar_{a_i} prod_j Ebar_{b_j} of weight sum a + sum b <= w with an even number of Ebar
    factors, as pairs of partitions (a, b): a basis of Pi_w."""
    out = []
    for n in range(w + 1):
        for k in range(n + 1):
            for a in partitions(k):
                for b in partitions(n - k):
                    if len(b) % 2 == 0:
                        out.append((tuple(a), tuple(b)))
    return out


def power_sums(c, kmax, F, A, q):
    """Obar_k = sum_{e in Z u Z'} {z_e^k},  Ebar_k = sum_{e in Z} (z_e^k + z_e^-k) - sum_{e in Z'} (z_e^k + z_e^-k),
    z_e = A q^{2c(e)}, k = 1..kmax (a single is Z = Z' = mu, the vacuum Z = Z' = ())."""
    O, E = [F.zero] * (kmax + 1), [F.zero] * (kmax + 1)
    for lam, sg in ((c[1], 1), (c[2], -1)):
        for e in boxes(lam):
            z, zk = A * q ** (2 * cont(e)), F.one
            for k in range(1, kmax + 1):
                zk = zk * z
                O[k] = O[k] + zk - 1 / zk
                E[k] = E[k] + sg * (zk + 1 / zk)
    return O, E


def monomial_values(c, mons, F, A, q):
    """The values m(c) of the monomials of Pi_w at the channel c, as integers mod p."""
    O, E = power_sums(c, max([sum(a) + sum(b) for a, b in mons] + [1]), F, A, q)
    vals = []
    for a, b in mons:
        v = F.one
        for k in a:
            v = v * O[k]
        for k in b:
            v = v * E[k]
        vals.append(int(v))
    return vals


def inverse_mod(M, p):
    """Inverse of a square matrix of integers mod p (Gauss-Jordan), or None if singular."""
    n = len(M)
    W = [[x % p for x in row] + [int(i == j) for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = next((r for r in range(col, n) if W[r][col]), None)
        if piv is None:
            return None
        W[col], W[piv] = W[piv], W[col]
        s = pow(W[col][col], p - 2, p)
        W[col] = [x * s % p for x in W[col]]
        for r in range(n):
            f = W[r][col]
            if r != col and f:
                W[r] = [(x - f * y) % p for x, y in zip(W[r], W[col])]
    return [row[n:] for row in W]


def low_columns(c, F, A, q):
    """The explicit light columns of weight 3 of every non-resonant row (from (L)):
        N_{c,{[2,1],[3]}}     =  Ebar_1 (2 b(q) b(A q^2) Ebar_1 - {q} Ebar_1 Obar_1 - b(q) Ebar_2) / (4 A^4 q^2 {q^2} D_1),
        N_{c,{[1,1,1],[2,1]}} = -q^2 Ebar_1 (2 b(q) b(A q^-2) Ebar_1 + {q} Ebar_1 Obar_1 - b(q) Ebar_2) / (4 A^4 {q^2} D_-1),
    with b(u) = u + 1/u."""
    O, E = power_sums(c, 2, F, A, q)
    b = lambda u: u + 1 / u
    x = E[1] * (2 * b(q) * b(A * q ** 2) * E[1] - curly(q) * E[1] * O[1] - b(q) * E[2]) / (
        4 * A ** 4 * q ** 2 * curly(q ** 2) * Dk(1, F, A, q))
    y = -q ** 2 * E[1] * (2 * b(q) * b(A / q ** 2) * E[1] + curly(q) * E[1] * O[1] - b(q) * E[2]) / (
        4 * A ** 4 * curly(q ** 2) * Dk(-1, F, A, q))
    return {("p", (2, 1), (3,)): x, ("p", (1, 1, 1), (2, 1)): y}


def check_lagrange(nmax=6, seed=1):
    """(L) The light columns of the non-resonant rows of N are Lagrange interpolants.  With the families of
    lagrange_family, the space Pi_w of Ebar-even polynomials of weight <= w in the composite power sums, and the nodes
    Omega^xi_w (all channels of weight <= w of the family), (U) |Omega^xi_w| = dim Pi_w and the node matrix is invertible;
    for a row c of the family and a column Y in Omega^xi_w of weight w < wt c,
        N_cY = sum_{c' in Omega^xi_w, wt c' = w} N_{c'Y} L_{c'}(c),     L_{c'} the Lagrange basis of Pi_w on Omega^xi_w,
    and N_cY = 0 for every light Y outside the family.  The top values N_{c'Y} are read from the engine rows (entries whose
    needed top value has no clean row are counted as not tested); also the explicit weight-3 columns of low_columns."""
    F, A, q = random_point(seed)
    p = F.p
    pt = I.Point(F, A, q)
    chans = channels_present(nmax)
    clean, excluded = clean_channels(pt, chans)
    order, Et, N = build_EN(pt, chans)
    pos = {c: i for i, c in enumerate(order)}
    have = set(clean) | {VAC}
    rows = [c for c in clean if lagrange_family(c)]
    wmax = max(weight(c) for c in rows) - 1
    nodes = lagrange_nodes(wmax)
    cnt = {"L": 0, "L nonzero": 0, "outside": 0, "untested": 0, "low": 0}
    units, bad = [], []
    allnodes, census = lagrange_nodes(10), []
    for w in range(11):
        sizes = [len([c for c in allnodes[xi] if weight(c) <= w]) for xi in ("h", "v")]
        census.append((w, sizes, len(pi_monomials(w))))
        if w <= 9 and sizes != [census[-1][2]] * 2:
            bad.append(("census", w, sizes, census[-1][2]))
    kinds = {}
    for c in rows:
        k = "single" if c[0] == "s" else ("one-cell" if pair_kind(c) == "1-cell" else "two-cell")
        kinds[k] = kinds.get(k, 0) + 1
    for w in range(wmax + 1):
        mons = pi_monomials(w)
        for xi in ("h", "v"):
            Om = [c for c in nodes[xi] if weight(c) <= w]
            units.append(len(Om) == len(mons))
            if len(Om) != len(mons):
                bad.append(("count", xi, w, len(Om), len(mons)))
                continue
            Vi = inverse_mod([monomial_values(c, mons, F, A, q) for c in Om], p)
            if Vi is None:
                bad.append(("singular", xi, w))
                continue
            top = [j for j, c in enumerate(Om) if weight(c) == w]
            omset = set(Om)
            for c in rows:
                if xi not in lagrange_family(c) or weight(c) <= w:
                    continue
                m = monomial_values(c, mons, F, A, q)
                L = {j: sum(m[k] * Vi[k][j] for k in range(len(mons))) % p for j in top}
                i = pos[c]
                for jY in top:
                    Y = Om[jY]
                    if Y not in pos:
                        continue
                    pred, ok = 0, True
                    for j in top:
                        if L[j]:
                            if Om[j] not in have:
                                ok = False
                                break
                            pred += int(N[pos[Om[j]]][pos[Y]]) * L[j]
                    if not ok:
                        cnt["untested"] += 1
                        continue
                    cnt["L"] += 1
                    cnt["L nonzero"] += N[i][pos[Y]] != 0
                    if N[i][pos[Y]] != F(pred % p):
                        bad.append(("L", xi, c, Y))
                for Y in order[1:i]:
                    if weight(Y) == w and Y not in omset:
                        cnt["outside"] += 1
                        if N[i][pos[Y]] != 0:
                            bad.append(("outside", xi, c, Y))
    for c in rows:
        if weight(c) > 3:
            for Y, v in low_columns(c, F, A, q).items():
                if Y in pos:
                    cnt["low"] += 1
                    if N[pos[c]][pos[Y]] != v:
                        bad.append(("low", c, Y))
    w10 = census[10]
    return not bad, ["lagrange: (U) |Omega_w| = dim Pi_w for w <= 9 in both families (dim Pi_9 = %d; at w = 10 %d nodes "
                     "against %d); |R| <= %d, rows %s, columns of weight <= %d: node matrices invertible %d/%d; (L) %d "
                     "entries (%d non-zero), %d light columns outside the family = 0, %d not tested (a top value without a "
                     "clean row); weight-3 columns %d; mismatches %s"
                     % (census[9][2], w10[1][0], w10[2], nmax, ", ".join("%s %d" % (k, kinds[k]) for k in sorted(kinds)),
                        wmax, sum(units), len(units), cnt["L"], cnt["L nonzero"], cnt["outside"], cnt["untested"],
                        cnt["low"], "none" if not bad else bad[:4])]


CHECKS = {"data": check_data, "kntz": check_kntz, "phi": check_phi, "f41": check_f41, "sym": check_sym,
          "table": check_table, "pairs": check_pairs, "bmat": check_bmat, "binom": check_binom, "pairn": check_pairn,
          "union": check_union, "hook": check_hook, "khook": check_khook, "fourrow": check_fourrow,
          "adjres": check_adjres, "gap2": check_gap2, "roots": check_roots, "fullrow": check_fullrow, "kernel": check_kernel,
          "topweight": check_topweight, "mult": check_mult, "closed": check_closed, "vacuum": check_vacuum,
          "twocell": check_twocell, "lagrange": check_lagrange}


def set_node_bound(nb):
    """Impose the vanishing conditions on all diagrams with at most nb boxes (the engine's default is
    max(7, |R_min| + 1)); the engine looks the node list up at call time."""
    orig = I.nodes.__wrapped__
    I.nodes = lambda c, extra_size=None: orig(c, nb if extra_size is None else extra_size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checks", nargs="*", default=list(CHECKS))
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nb", type=int, default=None,
                    help="node bound of the vanishing conditions (default: the engine's max(7, |R_min|+1)); "
                         "--nb 10 makes every channel with |R| <= 7 full rank")
    a = ap.parse_args(argv)
    if a.nb is not None:
        set_node_bound(a.nb)
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
