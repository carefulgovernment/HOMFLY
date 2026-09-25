"""Symmetric-group characters, Adams operations (plethysm with p_m) and
Littlewood--Richardson coefficients.

Border strips are handled with beta-numbers: adding a rim hook of length s to a
partition <-> moving one bead of the beta-set up by s; the sign is
(-1)^{#beads jumped over} = (-1)^{height of the hook}.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from math import factorial

from .partitions import P, partitions


def _beta(la, L):
    la = list(la) + [0] * (L - len(la))
    return [la[i] + L - 1 - i for i in range(L)]


def _from_beta(beta):
    b = sorted(beta, reverse=True)
    L = len(b)
    return P([b[i] - (L - 1 - i) for i in range(L)])


def add_rim_hooks(la, s):
    """All (mu, sign) with mu/la a rim hook of size s."""
    L = len(la) + s
    beta = _beta(la, L)
    bs = set(beta)
    out = []
    for x in beta:
        y = x + s
        if y not in bs:
            jumped = sum(1 for z in beta if x < z < y)
            nb = [y if z == x else z for z in beta]
            out.append((_from_beta(nb), -1 if jumped % 2 else 1))
    return out


@lru_cache(maxsize=None)
def characters_on_class(rho):
    """{la: chi^la(rho)} for all la |- |rho| via Murnaghan--Nakayama
    (building the diagram by adding rim hooks of sizes rho_i)."""
    cur = {(): 1}
    for s in rho:
        nxt = defaultdict(int)
        for la, v in cur.items():
            for mu, sg in add_rim_hooks(la, s):
                nxt[mu] += sg * v
        cur = {k: v for k, v in nxt.items() if v}
    return cur


def character(la, rho):
    return characters_on_class(tuple(sorted(rho, reverse=True))).get(P(la), 0)


def z_rho(rho):
    from collections import Counter
    z = 1
    for k, m in Counter(rho).items():
        z *= k ** m * factorial(m)
    return z


@lru_cache(maxsize=None)
def adams(la, m):
    """Adams operation psi_m s_la = s_la[p_m] = sum_Q c_Q s_Q.  Returns {Q: c_Q}
    (all c_Q in {0, +-1} -- asserted).  Needed by Rosso--Jones."""
    la = P(la)
    n = sum(la)
    acc = defaultdict(Fraction)
    for rho in partitions(n):
        chi = character(la, rho)
        if not chi:
            continue
        w = Fraction(chi, z_rho(rho))
        for Q, v in characters_on_class(tuple(m * r for r in rho)).items():
            acc[Q] += w * v
    out = {}
    for Q, c in acc.items():
        if c:
            assert c.denominator == 1, (la, m, Q, c)
            out[Q] = int(c)
    return out


# ----------------------------------------------------------------------------
# Littlewood--Richardson
# ----------------------------------------------------------------------------
def _horizontal_strips(la, k, max_row=None):
    """All mu ⊃ la with mu/la a horizontal strip of size k."""
    la = list(la)
    rows = len(la) + 1
    res = []

    def rec(i, left, cur):
        if i == rows:
            if left == 0:
                res.append(tuple(cur))
            return
        prev = la[i - 1] if i > 0 else None
        base = la[i] if i < len(la) else 0
        cap = (prev - base) if prev is not None else left
        for a in range(min(cap, left), -1, -1):
            rec(i + 1, left - a, cur + [a])

    rec(0, k, [])
    return res


@lru_cache(maxsize=None)
def lr_product(mu, nu):
    """s_mu * s_nu = sum_la c^la_{mu nu} s_la, returned as {la: c}."""
    mu, nu = P(mu), P(nu)
    # states: (shape, filling rows) where filling rows[i] = list of labels in row i
    states = [(list(mu), [[] for _ in range(len(mu) + len(nu))])]
    for label, k in enumerate(nu, start=1):
        new_states = []
        for shape, fill in states:
            for strip in _horizontal_strips(tuple(shape), k):
                shp = list(shape) + [0] * (len(strip) - len(shape))
                f2 = [list(r) for r in fill]
                while len(f2) < len(shp):
                    f2.append([])
                for i, a in enumerate(strip):
                    f2[i].extend([label] * a)
                    shp[i] += a
                if _lattice(f2, label):
                    new_states.append((shp, f2))
        states = new_states
    out = defaultdict(int)
    for shp, _ in states:
        out[P(shp)] += 1
    return dict(out)


def _lattice(fill, upto):
    cnt = [0] * (upto + 2)
    for row in fill:
        for x in reversed(row):
            cnt[x] += 1
            if x > 1 and cnt[x] > cnt[x - 1]:
                return False
    return True


def tensor_square(R):
    """R ⊗ R decomposition {X: multiplicity}."""
    return lr_product(R, R)
