"""Montesinos knots K(p1/q1; ...; pk/qk) by tangle calculus with Racah matrices.

The knot is the numerator closure of the sum T_1 + ... + T_k of rational
tangles.  Each rational tangle is built from the 0 or ∞ tangle by twists of the
right pair (H) and of the bottom pair (V); the orientation of every endpoint at
every step is tracked (``_orientations``).  A 4-point tangle is a coordinate
vector in one of two channel bases:

    H basis: fuse the left pair (a,d) and the right pair (b,c)
    V basis: fuse the top pair (a,b) and the bottom pair (d,c)

and each pair is parallel (R⊗R, channels Q) or antiparallel (R⊗R̄, channels X).
Per orientation pattern (H type, V type):

    (anti, anti):  c_H = S̄ v_V
    (par,  anti):  c_H = V v_V        (V = S^-1 of the family-P data)
    (anti, par ):  c_H = S v_V

Twists are diagonal: parallel pairs T^e, antiparallel pairs T̄^e with
exponents  H-anti: +s,  H-par: -s,  V-anti: +s,  V-par: -s  times the relative
handedness GROUP = -1 of the (H-anti, V-par) group (pinned on the fundamental
representation against KnotInfo for all 721 Montesinos knots <= 12 crossings).

Tangle sum = block product C_1 E^-1 C_2 E^-1 ... in the H basis of the vertex
channel type, where E are the coordinates of the 0-tangle in the *plain*
pattern of that type (antiparallel 'ioio' -> S̄ e0, parallel 'iooi' -> V e0);
the closure is  sum_X d_X Tr(C E^-1 ...).  This is gauge covariant: no Gram
matrices are needed.  Channel dimensions are recovered gauge-invariantly,
d_X = S̄_0X S̄_X0 d_R^2 and d_Q = S_0Q V_Q0 d_R^2 (multiplicity-free data).

Data: data/racah/montesinos/P_<R>.json.gz (scripts/import_racah_montesinos.py),
family P, multiplicity-free (rectangular) R.
"""
from __future__ import annotations

import gzip
import json
import os
import random
from fractions import Fraction
from functools import lru_cache

from ..algebra.fields import BadPoint, GF
from ..knots.families import MontesinosKnot
from ..reps.partitions import P, conjugate
from ..reps.qdim import qdim
from .base import Method

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                         "data", "racah", "montesinos"))
GROUP = -1


# ---------------------------------------------------------------------------
# rational tangles and orientations
# ---------------------------------------------------------------------------

def _ops_for(fr):
    x = Fraction(fr)
    ops = []
    while True:
        if x == 0:
            return "0", ops[::-1]
        if abs(x) >= 1:
            s = 1 if x > 0 else -1
            ops.append(("H", s))
            x -= s
        else:
            y = 1 / x
            s = 1 if y > 0 else -1
            ops.append(("V", s))
            y -= s
            if y == 0:
                return "inf", ops[::-1]
            x = 1 / y


def _build(fr):
    base, ops = _ops_for(fr)
    st = ({"a": (0, 0), "b": (0, 1), "d": (1, 0), "c": (1, 1)} if base == "0" else
          {"a": (0, 0), "d": (0, 1), "b": (1, 0), "c": (1, 1)})
    hist = [dict(st)]
    for kind, _ in ops:
        x, y = ("b", "c") if kind == "H" else ("c", "d")
        st[x], st[y] = st[y], st[x]
        hist.append(dict(st))
    return base, ops, hist


@lru_cache(maxsize=None)
def _orientations(fracs):
    """(components, [(base, ops, [orientation per step])]) for N(T_1 + ... + T_k)."""
    T = [_build(f) for f in fracs]
    k = len(T)
    ext = {}
    for i in range(k - 1):
        ext[(i, "b")], ext[(i + 1, "a")] = (i + 1, "a"), (i, "b")
        ext[(i, "c")], ext[(i + 1, "d")] = (i + 1, "d"), (i, "c")
    ext[(0, "a")], ext[(k - 1, "b")] = (k - 1, "b"), (0, "a")
    ext[(0, "d")], ext[(k - 1, "c")] = (k - 1, "c"), (0, "d")
    inner = []
    for _, _, hist in T:
        st = hist[-1]
        inner.append({e1: e2 for e1 in st for e2 in st if e1 != e2 and st[e1][0] == st[e2][0]})
    orient, seen, comps = {}, set(), 0
    for s0 in [(i, e) for i in range(k) for e in "abcd"]:
        if s0 in seen:
            continue
        comps += 1
        cur = s0
        while cur not in seen:
            i, e = cur
            seen.add(cur)
            orient[cur] = 1              # in
            e2 = inner[i][e]
            seen.add((i, e2))
            orient[(i, e2)] = 0          # out
            cur = ext[(i, e2)]
    out = []
    for i, (base, ops, hist) in enumerate(T):
        endo = {hist[-1][e]: orient[(i, e)] for e in "abcd"}
        steps = [{e: endo[st[e]] for e in "abcd"} for st in hist]
        out.append((base, tuple(ops), steps))
    return comps, out


def _types(o):
    return ("par" if o["a"] == o["d"] else "anti", "par" if o["a"] == o["b"] else "anti")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class _Poly:
    __slots__ = ("terms",)

    def __init__(self, terms):
        self.terms = [tuple(t) for t in terms]

    def ev(self, F, A, q, cache):
        s = F.zero
        for i, j, c in self.terms:
            k = (i, j)
            if k not in cache:
                cache[k] = A ** i * q ** j
            s = s + cache[k] * c
        return s


class PData:
    """Family P, multiplicity-free R: S̄, S, V, T, T̄."""

    def __init__(self, R, root=DATA_DIR):
        with gzip.open(os.path.join(root, "P_%s.json.gz" % "".join(map(str, R))), "rt") as f:
            d = json.load(f)
        self.R = tuple(R)
        self.vac = d["vacuum_index"]
        self.mats = {k: [[(_Poly(n), _Poly(dd)) for n, dd in row] for row in d[k]]
                     for k in ("Sbar", "S", "V")}
        self.diags = {k: [(_Poly(n), _Poly(dd)) for n, dd in d[k]] for k in ("T", "Tbar")}
        self.n = len(d["Sbar"])

    def evaluate(self, F, A, q):
        cache = {}

        def ev(e):
            den = e[1].ev(F, A, q, cache)
            if den == 0:
                raise BadPoint("pole in Racah data")
            return e[0].ev(F, A, q, cache) / den
        out = {k: [[ev(e) for e in row] for row in M] for k, M in self.mats.items()}
        out.update({k: [ev(e) for e in v] for k, v in self.diags.items()})
        dR = qdim(self.R, A, q)
        Sb, S, V = out["Sbar"], out["S"], out["V"]
        z = self.vac
        out["d_anti"] = [Sb[z][X] * Sb[X][z] * dR * dR for X in range(self.n)]
        out["d_par"] = [S[z][Qi] * V[Qi][z] * dR * dR for Qi in range(self.n)]
        out["dR"] = dR
        out["blocks_anti"] = [[[X]] for X in range(self.n)]     # 1x1 blocks
        out["blocks_par"] = [[[Qi]] for Qi in range(self.n)]
        return out


class HData:
    """Family H (Hecke Y-gauge): S̄, mixed S with explicit (X,a,b)/(Q,a,b)
    labels -> multiplicity blocks."""

    def __init__(self, R, root=DATA_DIR, family="H"):
        with gzip.open(os.path.join(root, "%s_%s.json.gz" % (family, "".join(map(str, R)))), "rt") as f:
            d = json.load(f)
        self.R = tuple(R)
        self.family = family
        self.anti, self.par = d["anti"], d["par"]
        self.mats = {k: [[(_Poly(n), _Poly(dd)) for n, dd in row] for row in d[k]]
                     for k in ("Sbar", "S") if k in d}
        self.Tpar = d["T"]
        self.n = len(self.anti)
        self.blocks_anti = self._blocks([((tuple(x["Z"]), tuple(x["Zp"])), x["a"], x["b"])
                                         for x in self.anti])
        self.blocks_par = self._blocks([(tuple(x["Q"]), x["a"], x["b"]) for x in self.par])
        self.vac = next(i for i, x in enumerate(self.anti) if not x["Z"] and not x["Zp"])

    @staticmethod
    def _blocks(labels):
        groups = {}
        for i, (key, a, b) in enumerate(labels):
            groups.setdefault(key, []).append((a, b, i))
        out = []
        for key, items in groups.items():
            As = sorted({a for a, _, _ in items})
            Bs = sorted({b for _, b, _ in items})
            pos = {(a, b): i for a, b, i in items}
            out.append((key, [[pos[(a, b)] for b in Bs] for a in As]))
        return out

    def evaluate(self, F, A, q):
        from ..reps.partitions import kappa
        from .interpolation import Point
        cache = {}

        def ev(e):
            den = e[1].ev(F, A, q, cache)
            if den == 0:
                raise BadPoint("pole in Racah data")
            return e[0].ev(F, A, q, cache) / den
        S = [[ev(e) for e in row] for row in self.mats["S"]]
        Sb = [[ev(e) for e in row] for row in self.mats["Sbar"]] if "Sbar" in self.mats else None
        Sinv = _inv(S, F)
        dR = qdim(self.R, A, q)
        theta = A ** sum(self.R) * q ** (2 * kappa(self.R))
        T = [sg * A ** ea * q ** eq / theta for sg, ea, eq in self.Tpar]
        Tb = [x["eps"] * A ** sum(x["Z"]) * q ** (kappa(tuple(x["Z"])) + kappa(tuple(x["Zp"])))
              for x in self.anti]
        if self.family == "G":
            # family H satisfies T̄^-1 S̄ T̄^-1 = S^-1 T S; family G publishes S̄
            # in a vacuum-dual gauge (rows and columns normalised differently),
            # so S̄ is rebuilt from this identity in the gauge of S's columns
            K = _mm(Sinv, [[T[i] * S[i][j] for j in range(len(S))] for i in range(len(S))], F)
            Sb = [[Tb[i] * K[i][j] * Tb[j] for j in range(len(S))] for i in range(len(S))]
        pt = Point(F, A, q)
        d_anti = [F.zero] * self.n
        for (Z, Zp), idx in self.blocks_anti:
            dX = pt.chi((), Z, Zp)
            for row in idx:
                for i in row:
                    d_anti[i] = dX
        d_par = [F.zero] * self.n
        for Q, idx in self.blocks_par:
            dQ = qdim(Q, A, q)
            for row in idx:
                for i in row:
                    d_par[i] = dQ
        # engine slots: V = (par,anti) transform c_Hpar = V v_Vanti ; S = (anti,par)
        return {"Sbar": Sb, "V": S, "S": Sinv, "T": T, "Tbar": Tb, "dR": dR,
                "d_anti": d_anti, "d_par": d_par,
                "blocks_anti": [idx for _, idx in self.blocks_anti],
                "blocks_par": [idx for _, idx in self.blocks_par], "vac": self.vac}


@lru_cache(maxsize=None)
def load(R, root=DATA_DIR):
    R = tuple(R)
    key = "".join(map(str, R))
    if os.path.exists(os.path.join(root, "P_%s.json.gz" % key)):
        return PData(R, root)
    if os.path.exists(os.path.join(root, "H_%s.json.gz" % key)):
        return HData(R, root, "H")
    return HData(R, root, "G")


@lru_cache(maxsize=None)
def available(root=DATA_DIR):
    if not os.path.isdir(root):
        return []
    return sorted({tuple(int(c) for c in f[2:-8]) for f in os.listdir(root)
                   if f[:2] in ("P_", "H_", "G_") and f.endswith(".json.gz")})


# ---------------------------------------------------------------------------
# small dense linear algebra over F
# ---------------------------------------------------------------------------

def _mv(M, v, F):
    return [sum((M[i][j] * v[j] for j in range(len(v))), F.zero) for i in range(len(M))]


def _inv(M, F):
    from ..algebra.linalg import inverse
    return inverse(M, F.one)


def _mm(X, Y, F):
    return [[sum((X[i][k] * Y[k][j] for k in range(len(Y))), F.zero) for j in range(len(Y[0]))]
            for i in range(len(X))]


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

def _transforms(D, F):
    M = {("anti", "anti"): D["Sbar"], ("par", "anti"): D["V"], ("anti", "par"): D["S"]}
    return M, {k: _inv(m, F) for k, m in M.items()}


def natural_value(fracs, D, F):
    """Reduced invariant of N(T_1 + ... + T_k) from evaluated data D (dict)."""
    comps, tl = _orientations(tuple(Fraction(x) for x in fracs))
    if comps != 1:
        raise ValueError("Montesinos sum is not a knot")
    M, Minv = _transforms(D, F)
    n = len(D["Sbar"])
    e0 = [F.zero] * n
    e0[D.get("vac", 0)] = F.one
    tb, tp = D["Tbar"], D["T"]
    vecs, verts = [], []
    for base, ops, steps in tl:
        basis = "V" if base == "0" else "H"
        v = list(e0)
        for (kind, s), o in zip(ops, steps[:-1]):
            h, vv = _types(o)
            if kind == "H":
                if basis == "V":
                    v, basis = _mv(M[(h, vv)], v, F), "H"
                ev, e = (tb, s * GROUP) if h == "anti" else (tp, -s)
            else:
                if basis == "H":
                    v, basis = _mv(Minv[(h, vv)], v, F), "V"
                ev, e = (tb, s) if vv == "anti" else (tp, -s * GROUP)
            v = [x * y ** e for x, y in zip(v, ev)]
        if basis == "V":
            v = _mv(M[_types(steps[-1])], v, F)
        vecs.append(v)
        verts.append(_types(steps[-1])[0])
    vert = verts[0]
    E = _mv(M[("anti", "anti")] if vert == "anti" else M[("par", "anti")], e0, F)
    dims = D["d_anti"] if vert == "anti" else D["d_par"]
    blocks = D["blocks_anti"] if vert == "anti" else D["blocks_par"]
    tot = F.zero
    for idx in blocks:
        m = len(idx)
        Em = [[E[idx[i][j]] for j in range(m)] for i in range(m)]
        Ei = _inv(Em, F)
        prod = None
        for v in vecs:
            Cm = [[v[idx[i][j]] for j in range(m)] for i in range(m)]
            X = _mm(Cm, Ei, F)
            prod = X if prod is None else _mm(prod, X, F)
        tr = sum((prod[i][i] for i in range(m)), F.zero)
        tot = tot + dims[idx[0][0]] * tr
    return tot / D["dR"]


def value(fracs, R, F, A, q, root=DATA_DIR):
    """H_R of N(sum of tangles) at (A, q); with the sign of the fractions fixed
    by ``chirality`` this is the STANDARD convention.  A representation without
    data is taken from its transpose, H_{R^T}(A, q) = H_R(A, -1/q)."""
    R = P(R)
    if R not in available(root) and conjugate(R) in available(root):
        return natural_value(fracs, load(conjugate(R), root).evaluate(F, A, -1 / q), F)
    return natural_value(fracs, load(R, root).evaluate(F, A, q), F)


def chirality(fracs, reference):
    """Sign s such that N(s*fractions) matches ``reference`` (the STANDARD
    fundamental HOMFLY), or None if both mirrors match (symmetric H_[1])."""
    F = GF(2 ** 61 - 1)
    rng = random.Random(hash(tuple(fracs)) & 0xffff)
    A, q = F.random_element(rng), F.random_element(rng)
    ref = reference.evaluate([A, q], one=F.one)
    hits = [s for s in (1, -1) if value([s * Fraction(x) for x in fracs], (1,), F, A, q) == ref]
    if not hits:
        raise ValueError("no chirality matches the reference")
    return hits[0] if len(hits) == 1 else None


class MontesinosMethod(Method):
    name = "montesinos"

    def __init__(self, root=DATA_DIR):
        self.root = root

    def supports(self, knot, R):
        av = available(self.root)
        return isinstance(knot, MontesinosKnot) and (P(R) in av or conjugate(P(R)) in av)

    def evaluate(self, knot, R, F, A, q):
        return value(knot.fractions(), R, F, A, q, self.root)
