"""Interpolation formula for antiparallel double-braid knots in any R
("An interpolation formula for colored HOMFLY polynomials of the figure-eight
knot and of double-braid knots in arbitrary representations", Sept 2026).

    H_R(m, n) = 1 + sum_c F_c(R) phi_c(m) phi_c(n) / (phi_c(1) phi_c(-1)),
    F_c(R)    = d_c sum_X E_cX psi_X(R),     phi_c(m) = sum_X E_cX (Lambda_X^m - 1).

No Racah matrices: R enters only through normalised composite characters
chî_X(R) = chi_X(q^{2(R+rho)}) / d_X (Hopf-link values), psi_X = chî_X - 1.

* channels c: singles [mu, mu] (mu != ∅) and pairs {Z, Z'} (Z != Z', |Z| = |Z'|);
* Lambda_[Z,Z'] = A^{2|Z|} q^{2(kappa_Z + kappa_Z')};
* chi_[alpha,beta](z) = sum_gamma (-1)^|gamma| s_{alpha/gamma}(z) s_{beta/gamma'}(1/z)  (Koike),
  Schur functions from the generic-A power sums
      p_k  = (1 - A^{-2k})/(1 - q^{-2k}) + (q^{2k} - 1)  C_k(R),
      p_-k = (1 - A^{2k}) /(1 - q^{2k})  + (q^{-2k} - 1) C_-k(R),  C_k = sum_boxes q^{2k c};
* presence m_c(R) = sum_gamma c^R_{Z gamma} c^R_{Z' gamma};
* R_min(c) = minimal R with m_c(R) > 0; X < c iff every diagram of R_min(X) is a
  proper subdiagram of some diagram of R_min(c);
  B(c) = {X < c present in some R in R_min(c)};
* row E_c: E_cc = 1 and sum_{X in {c} ∪ B(c)} E_cX psi_X(nu) = 0 for every nu
  with m_c(nu) = 0 (|nu| <= max(7, |R_min(c)| + 1)).

Everything is evaluated at a point (A, q) of a field (GF(p) in production), so
the method plugs into the reconstruction pipeline.  The paper's convention
(H(1,-1) = 4_1, H(1,1) = trefoil) is mapped to ours in ``DoubleBraidInterpolation``
and pinned against the Racah two-bridge data in tests/test_interpolation.py.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from ..algebra.fields import BadPoint
from ..knots.families import DoubleBraid
from ..reps.characters import character, lr_product, z_rho
from ..reps.partitions import P, conjugate, kappa, partitions
from .base import Method

# ---------------------------------------------------------------------------
# combinatorics (R-, point-independent; cached)
# ---------------------------------------------------------------------------


def contains(big, small):
    return len(small) <= len(big) and all(s <= b for s, b in zip(small, big))


@lru_cache(maxsize=None)
def schur_in_power_sums(lam):
    n = sum(lam)
    out = []
    for rho in partitions(n):
        c = character(lam, rho)
        if c:
            out.append((rho, Fraction(c, z_rho(rho))))
    return tuple(out)


@lru_cache(maxsize=None)
def skew(alpha, gamma):
    """{nu: c^alpha_{gamma nu}}  (s_{alpha/gamma} = sum c s_nu)."""
    if not contains(alpha, gamma):
        return {}
    out = {}
    for nu in partitions(sum(alpha) - sum(gamma)):
        c = lr_product(gamma, nu).get(alpha, 0)
        if c:
            out[nu] = c
    return out


@lru_cache(maxsize=None)
def presence(c, R):
    """m_c(R) = sum_gamma c^R_{Z gamma} c^R_{Z' gamma}."""
    _, Z, Zp = c
    a, b = skew(R, Z), skew(R, Zp)
    return sum(v * b.get(g, 0) for g, v in a.items())


def single(mu):
    return ("s", mu, mu)


def pair(Z, Zp):
    Z, Zp = sorted((Z, Zp))
    return ("p", Z, Zp)


def composites(c):
    return [(c[1], c[2])] if c[0] == "s" else [(c[1], c[2]), (c[2], c[1])]


def level(c):
    return sum(c[1])


def union(a, b):
    n = max(len(a), len(b))
    a, b = list(a) + [0] * (n - len(a)), list(b) + [0] * (n - len(b))
    return P([max(x, y) for x, y in zip(a, b)])


@lru_cache(maxsize=None)
def channels_up_to(L):
    out = []
    for k in range(1, L + 1):
        ps = partitions(k)
        out += [single(mu) for mu in ps]
        out += [pair(Z, Zp) for Z, Zp in combinations(ps, 2)]
    return tuple(out)


def _supersets(base, extra):
    """Partitions containing ``base`` with exactly ``extra`` more boxes."""
    cur = {base}
    for _ in range(extra):
        nxt = set()
        for la in cur:
            for i in range(len(la) + 1):
                row = la[i] if i < len(la) else 0
                if i == 0 or la[i - 1] > row:
                    l2 = list(la) + [0]
                    l2[i] += 1
                    nxt.add(P(l2))
        cur = nxt
    return cur


@lru_cache(maxsize=None)
def rmin(c):
    if c[0] == "s":
        return (c[1],)
    base = union(c[1], c[2])
    found = []
    for extra in range(0, 7):
        for R in sorted(_supersets(base, extra)):
            if presence(c, R) and not any(contains(R, f) for f in found):
                found.append(R)
    if not found:
        raise RuntimeError("no minimal diagram found for %s" % (c,))
    return tuple(sorted(found))


def precedes(x, c):
    rc = rmin(c)
    return all(any(contains(r, s) and r != s for r in rc) for s in rmin(x))


@lru_cache(maxsize=None)
def support(c):
    """B(c) (ordered by level, singles before pairs)."""
    rc = rmin(c)
    L = max(sum(r) for r in rc)
    out = []
    for x in channels_up_to(L):
        if x != c and any(presence(x, r) for r in rc) and precedes(x, c):
            out.append(x)
    out.sort(key=lambda x: (level(x), x[0] == "p", x))
    return tuple(out)


NODES_MIN_BOUND = 9   # PATCHED (racah_homfly_all): was 7.  With max(7, |R_min|+1) the vanishing system for some
                      # channels of |R| = 7 (e.g. several channels of R = [4,2,1]) is rank deficient, and the free
                      # unknowns silently set to 0 give wrong rows E_c (the bridge w = E^T G then disagrees with the
                      # Racah/tableau weights).  Nodes up to |nu| <= 9 remove every deficiency for |R| <= 7;
                      # for |R| = 8 the bound max(9, |R_min|+2) = 10 is used.  See probes/formula1_fast2_L7.py.


@lru_cache(maxsize=None)
def nodes(c, extra_size=None):
    """Partitions nu with m_c(nu) = 0, |nu| <= max(NODES_MIN_BOUND, |R_min|+2), small first."""
    bound = max(NODES_MIN_BOUND, max(sum(r) for r in rmin(c)) + 2) if extra_size is None else extra_size
    out = []
    for k in range(0, bound + 1):
        for nu in partitions(k):
            if not presence(c, nu):
                out.append(nu)
    return tuple(out)


def channels_of(R):
    R = P(R)
    return tuple(c for c in channels_up_to(sum(R)) if presence(c, R))


# ---------------------------------------------------------------------------
# point evaluation
# ---------------------------------------------------------------------------


class Point:
    """All point-dependent quantities at (A, q) in F, with caches."""

    def __init__(self, F, A, q):
        self.F, self.A, self.q = F, A, q
        self.q2 = q * q
        self.A2 = A * A
        self._p, self._s, self._chi, self._d, self._E = {}, {}, {}, {}, {}

    def power(self, nu, k):
        key = (nu, k)
        if key not in self._p:
            q2k = self.q2 ** k
            den = 1 - q2k ** -1
            if den == 0:
                raise BadPoint("q^{2k} = 1")
            v = (1 - self.A2 ** (-k)) / den
            C = self.F.zero
            for i, r in enumerate(nu):
                for j in range(r):
                    C = C + q2k ** (j - i)
            self._p[key] = v + (q2k - 1) * C
        return self._p[key]

    def schur(self, nu, lam, sign):
        key = (nu, lam, sign)
        if key not in self._s:
            if not lam:
                self._s[key] = self.F.one
            else:
                tot = self.F.zero
                for rho, c in schur_in_power_sums(lam):
                    t = self.F(c)
                    for r in rho:
                        t = t * self.power(nu, sign * r)
                    tot = tot + t
                self._s[key] = tot
        return self._s[key]

    def skew_schur(self, nu, alpha, gamma, sign):
        tot = self.F.zero
        for lam, c in skew(alpha, gamma).items():
            tot = tot + c * self.schur(nu, lam, sign)
        return tot

    def chi(self, nu, alpha, beta):
        key = (nu, alpha, beta)
        if key not in self._chi:
            tot = self.F.zero
            for k in range(0, min(sum(alpha), sum(beta)) + 1):
                for g in partitions(k):
                    if not contains(alpha, g) or not contains(beta, conjugate(g)):
                        continue
                    t = self.skew_schur(nu, alpha, g, 1) * self.skew_schur(nu, beta, conjugate(g), -1)
                    tot = tot + t if k % 2 == 0 else tot - t
            self._chi[key] = tot
        return self._chi[key]

    def d(self, c):
        if c not in self._d:
            self._d[c] = sum((self.chi((), a, b) for a, b in composites(c)), self.F.zero)
        return self._d[c]

    def psi(self, c, nu):
        d = self.d(c)
        if d == 0:
            raise BadPoint("vanishing composite dimension")
        return sum((self.chi(nu, a, b) for a, b in composites(c)), self.F.zero) / d - 1

    def Lam(self, c):
        Z, Zp = c[1], c[2]
        return self.A2 ** sum(Z) * self.q2 ** (kappa(Z) + kappa(Zp))

    def E_row(self, c):
        """{X: E_cX} for X in {c} ∪ B(c), solved from the vanishing conditions."""
        if c in self._E:
            return self._E[c]
        B = support(c)
        F = self.F
        n = len(B)
        rows = []          # reduced rows [coeffs..., rhs], pivots
        pivots = []
        for nu in nodes(c):
            if len(pivots) == n:
                break
            r = [self.psi(x, nu) for x in B] + [-self.psi(c, nu)]
            for (pc, pr) in zip(pivots, rows):
                if r[pc] != 0:
                    f = r[pc]
                    r = [a - f * b for a, b in zip(r, pr)]
            piv = next((i for i in range(n) if r[i] != 0), None)
            if piv is None:
                continue          # dependent condition (consistency not checked here)
            inv = F.one / r[piv]
            r = [a * inv for a in r]
            for k, pr in enumerate(rows):
                if pr[piv] != 0:
                    f = pr[piv]
                    rows[k] = [a - f * b for a, b in zip(pr, r)]
            rows.append(r)
            pivots.append(piv)
        sol = [F.zero] * n    # free unknowns (rank deficiency) set to 0
        for pc, pr in zip(pivots, rows):
            sol[pc] = pr[n]
        row = {c: F.one}
        for x, v in zip(B, sol):
            row[x] = v
        self._E[c] = row
        self._E[("rank", c)] = (len(pivots), n)
        return row

    def F_c(self, c, R):
        E = self.E_row(c)
        return self.d(c) * sum((e * self.psi(x, R) for x, e in E.items()), self.F.zero)

    def phi(self, c, m):
        E = self.E_row(c)
        return sum((e * (self.Lam(x) ** m - 1) for x, e in E.items()), self.F.zero)


def row_violations(pt, c):
    """Number of vanishing conditions (all nodes of c) violated by the solved
    row E_c -- 0 means the over-determined interpolation system is consistent."""
    E = pt.E_row(c)
    bad = 0
    for nu in nodes(c):
        if sum((e * pt.psi(x, nu) for x, e in E.items()), pt.F.zero) != 0:
            bad += 1
    return bad


def paper_value(R, m, n, F, A, q, point=None):
    """H_R(m, n) exactly as in the paper (its own convention)."""
    R = P(R)
    pt = point or Point(F, A, q)
    tot = F.one
    for c in channels_of(R):
        den = pt.phi(c, 1) * pt.phi(c, -1)
        if den == 0:
            raise BadPoint("phi_c(1) phi_c(-1) = 0")
        tot = tot + pt.F_c(c, R) * pt.phi(c, m) * pt.phi(c, n) / den
    return tot


def paper_value_41(R, F, A, q):
    """H_R(4_1) = 1 + sum_c F_c(R)."""
    R = P(R)
    pt = Point(F, A, q)
    return F.one + sum((pt.F_c(c, R) for c in channels_of(R)), F.zero)


# ---------------------------------------------------------------------------
# method
# ---------------------------------------------------------------------------

# paper convention -> ours: identical.  H_paper(m, n)(A, q) is our STANDARD H of
# DoubleBraid(m, n) (4-plat cf (-2m, -2n)); pinned against Racah data in tests.
def PAPER_POINT(A, q):
    return A, q


class DoubleBraidInterpolation(Method):
    """Antiparallel double braids DoubleBraid(m, n), |R| <= 7 (no Racah data)."""

    name = "double-braid-interpolation"
    #: the naive formula is exact up to this size; beyond it methods.formula2.DoubleBraidStrong (|R| <= 10)
    max_boxes = 7

    def supports(self, knot, R):
        return (isinstance(knot, DoubleBraid) and knot.antiparallel == (True, True)
                and sum(R) <= self.max_boxes)

    def evaluate(self, knot, R, F, A, q):
        a, b = PAPER_POINT(A, q)
        return paper_value(R, knot.m, knot.n, F, a, b)
