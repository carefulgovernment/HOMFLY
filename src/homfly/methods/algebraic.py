"""Algebraic (arborescent) knots N(t) by tangle calculus with the Montesinos
Racah data (methods.montesinos): every R with |R| <= 6.

A tangle tree (knots.conway) is evaluated bottom-up to the coordinates of the
tangle in the H basis of its own orientation pattern:

  leaf      rational tangle, built by twists from 0 or ∞ (as in montesinos)
  H-sum     C = C_1 E^-1 C_2 E^-1 ... C_k          (H basis, blockwise)
            E = H coordinates of the 0 tangle (identity of the H sum)
  V-sum     W = v_1 F^-1 v_2 F^-1 ... v_k          (V basis, blockwise)
            v_i = V coordinates of the children, F = V coordinates of the
            ∞ tangle (identity of the V stack); C = M W

with M the H<-V change of basis of the node's orientation pattern.  The
first block index belongs to the left (top) pair, the second to the right
(bottom) pair.  The closure is N(t) = sum_blocks d_X Tr(C E^-1) / d_R.  The
orientation of every endpoint of every subtangle follows from the strands of
the whole diagram (``_orient``).
"""
from __future__ import annotations

import random
from fractions import Fraction
from functools import lru_cache

from ..algebra.fields import BadPoint, GF
from ..knots import conway
from ..reps.partitions import P, conjugate
from . import montesinos as MO
from .base import Method


class AlgebraicKnot:
    """N(tree) for a tangle tree of knots.conway (``mirror`` mirrors it)."""

    def __init__(self, tree, mirror=False, notation=None):
        self.tree = conway.mirror(tree) if mirror else tree
        self.notation = notation
        self.mirror = mirror

    @classmethod
    def from_conway(cls, s, mirror=False):
        return cls(conway.parse(s), mirror, s)

    def key(self):
        return _freeze(self.tree)

    def is_knot(self):
        return _orient(self.key())[0] == 1

    def __repr__(self):
        return "AlgebraicKnot(%r%s)" % (self.notation or self.tree, ", mirror" if self.mirror else "")


def _freeze(t):
    if t[0] == "leaf":
        return ("leaf", t[1])
    return (t[0], tuple(_freeze(c) for c in t[1]))


# ---------------------------------------------------------------------------
# orientations
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _orient(tree):
    """(components, nodes) where nodes is the post-order list of
    ("leaf", base, ops, steps) / ("H"|"V", nchildren, (h, v)) with the
    orientation pattern (h, v) of every node."""
    leaves = []          # (base, ops, hist)
    shape = []           # post-order: ("leaf", i) / (kind, n, ends)

    def walk(t):
        if t[0] == "leaf":
            i = len(leaves)
            f = t[1]
            if f is conway.INF:
                st = {"a": (0, 0), "d": (0, 1), "b": (1, 0), "c": (1, 1)}
                leaves.append(("inf", (), [st]))
            else:
                leaves.append(MO._build(f))
            ends = {e: (i, e) for e in "abcd"}
            shape.append(("leaf", i, ends))
            return ends
        ch = [walk(c) for c in t[1]]
        for x, y in zip(ch, ch[1:]):
            if t[0] == "H":
                glue.append((x["b"], y["a"]))
                glue.append((x["c"], y["d"]))
            else:
                glue.append((x["d"], y["a"]))
                glue.append((x["c"], y["b"]))
        if t[0] == "H":
            ends = {"a": ch[0]["a"], "d": ch[0]["d"], "b": ch[-1]["b"], "c": ch[-1]["c"]}
        else:
            ends = {"a": ch[0]["a"], "b": ch[0]["b"], "d": ch[-1]["d"], "c": ch[-1]["c"]}
        shape.append((t[0], len(ch), ends))
        return ends

    glue = []
    root = walk(tree)
    glue += [(root["a"], root["b"]), (root["d"], root["c"])]         # N closure
    ext = {}
    for x, y in glue:
        ext[x], ext[y] = y, x
    inner = []
    for _, _, hist in leaves:
        st = hist[-1]
        inner.append({e1: e2 for e1 in st for e2 in st if e1 != e2 and st[e1][0] == st[e2][0]})
    orient, seen, comps = {}, set(), 0
    for s0 in [(i, e) for i in range(len(leaves)) for e in "abcd"]:
        if s0 in seen:
            continue
        comps += 1
        cur = s0
        while cur not in seen:
            i, e = cur
            seen.add(cur)
            orient[cur] = 1
            e2 = inner[i][e]
            seen.add((i, e2))
            orient[(i, e2)] = 0
            cur = ext[(i, e2)]
    nodes = []
    for kind, x, ends in shape:
        o = {e: orient[ends[e]] for e in "abcd"}
        pat = MO._types(o)
        if kind == "leaf":
            base, ops, hist = leaves[x]
            endo = {hist[-1][e]: orient[(x, e)] for e in "abcd"}
            steps = [{e: endo[st[e]] for e in "abcd"} for st in hist]
            nodes.append(("leaf", base, tuple(ops), steps, pat))
        else:
            nodes.append((kind, x, pat))
    return comps, nodes


# ---------------------------------------------------------------------------
# evaluation (numpy, p < 2^21)
# ---------------------------------------------------------------------------

def _block_product(xs, G, blocks, p):
    """x_1 G^-1 x_2 G^-1 ... x_k blockwise (vectors over the labels)."""
    import numpy as np
    out = np.zeros_like(xs[0])
    by_shape = {}
    for idx in blocks:
        by_shape.setdefault((len(idx), len(idx[0])), []).append(idx)
    for idxs in by_shape.values():
        ix = np.array(idxs)
        Gi = MO._inv_stack(G[ix], p)
        prod = xs[0][ix]
        for x in xs[1:]:
            prod = MO._mm_np(MO._mm_np(prod, Gi, p), x[ix], p)
        out[ix] = prod
    return out


def natural_value_np(tree, D, p, vsum_order=1):
    import numpy as np
    comps, nodes = _orient(tree)
    if comps != 1:
        raise ValueError("algebraic tangle closure is not a knot")
    M = {("anti", "anti"): D["Sbar"], ("par", "anti"): D["V"], ("anti", "par"): D["S"]}
    Minv = {("par", "anti"): D["S"], ("anti", "par"): D["V"]}

    def minv(key):
        if key not in Minv:
            Minv[key] = MO._inv_np(M[key], p)
        return Minv[key]
    n = D["Sbar"].shape[0]
    vac = D["vac"]
    tb, tp = D["Tbar"], D["T"]
    tbi, tpi = MO._inv_vec(tb, p), MO._inv_vec(tp, p)
    blocks = {"anti": D["blocks_anti"], "par": D["blocks_par"]}
    e0 = np.zeros(n, dtype=np.int64)
    e0[vac] = 1

    def pw(ev, evi, e):
        base = ev if e > 0 else evi
        out = np.ones(n, dtype=np.int64)
        for _ in range(abs(e)):
            out = out * base % p
        return out

    def leaf(base, ops, steps):
        basis = "V" if base == "0" else "H"
        v = e0.copy()
        if base != "0":                     # normalised from the 0 tangle
            v = v * inf_norm(MO._types(steps[0])[1]) % p
        for (kind, s), o in zip(ops, steps[:-1]):
            h, vv = MO._types(o)
            if kind == "H":
                if basis == "V":
                    v, basis = MO._mm_np(M[(h, vv)], v[:, None], p)[:, 0], "H"
                ev, evi, e = (tb, tbi, s * MO.GROUP) if h == "anti" else (tp, tpi, -s)
            else:
                if basis == "H":
                    v, basis = MO._mm_np(minv((h, vv)), v[:, None], p)[:, 0], "V"
                ev, evi, e = (tb, tbi, s) if vv == "anti" else (tp, tpi, -s * MO.GROUP)
            v = v * pw(ev, evi, e) % p
        if basis == "V":
            v = MO._mm_np(M[MO._types(steps[-1])], v[:, None], p)[:, 0]
        return v

    def inf_norm(v):
        """The ∞ tangle built by twists from the 0 tangle, 1/(1 - 1), relative
        to e_vac in the H basis: the leaves are normalised from the 0 tangle,
        so the identity of the V stack must be too."""
        if v not in norm:
            o = {"anti": {"a": 1, "b": 0, "c": 1, "d": 0}, "par": {"a": 1, "b": 1, "c": 0, "d": 0}}[v]
            st = {"a": (0, 0), "b": (0, 1), "d": (1, 0), "c": (1, 1)}       # 0 tangle
            hist = [dict(st)]
            for x, y in (("b", "c"), ("c", "d")):                           # H then V twist
                st[x], st[y] = st[y], st[x]
                hist.append(dict(st))
            endo = {hist[-1][e]: o[e] for e in "abcd"}
            steps = [{e: endo[h_[e]] for e in "abcd"} for h_ in hist]
            w = leaf("0", (("H", 1), ("V", -1)), steps)
            if MO._types(steps[-1]) != ("anti", v) or (np.delete(w, vac) % p).any() or not w[vac]:
                raise AssertionError("1/(1-1) is not the ∞ tangle")
            norm[v] = int(w[vac]) % p
        return norm[v]
    norm = {}

    stack = []                     # (H coordinates, pattern)
    for node in nodes:
        if node[0] == "leaf":
            _, base, ops, steps, pat = node
            stack.append((leaf(base, ops, steps), pat))
            continue
        kind, k, (h, v) = node
        ch = stack[-k:]
        del stack[-k:]
        if kind == "H":
            E = M[(h, "anti")][:, vac]
            C = _block_product([c for c, _ in ch], E, blocks[h], p)
        else:
            Fv = minv(("anti", v))[:, vac] * inf_norm(v) % p
            vs = [MO._mm_np(minv((ph, v)), c[:, None], p)[:, 0] for c, (ph, _) in ch]
            if vsum_order < 0:
                vs = vs[::-1]
            W = _block_product(vs, Fv, blocks[v], p)
            C = MO._mm_np(M[(h, v)], W[:, None], p)[:, 0]
        stack.append((C, (h, v)))
    (C, (h, _)), = stack
    E = M[(h, "anti")][:, vac]
    dims = D["d_anti"] if h == "anti" else D["d_par"]
    tot = 0
    for idx in blocks[h]:
        ix = np.array(idx)
        X = MO._mm_np(C[ix], MO._inv_np(E[ix], p), p)
        tot = (tot + int(dims[idx[0][0]]) * int(np.trace(X) % p)) % p
    return tot * pow(int(D["dR"]), -1, p) % p


def value(tree, R, F, A, q, root=MO.DATA_DIR, **kw):
    """H_R(N(tree)) at (A, q) in GF(p), p < 2^21 (STANDARD convention once the
    chirality is fixed by ``chirality``)."""
    R = P(R)
    if R not in MO.available(root) and conjugate(R) in MO.available(root):
        R, q = conjugate(R), -1 / q
    p = getattr(F, "p", None)
    if p is None or p >= MO.NP_BOUND:
        raise ValueError("algebraic: GF(p) with p < 2^21 only")
    d = MO.load(R, root)
    return F(natural_value_np(tree, d.evaluate_np(p, int(A), int(q)), p, **kw))


def chirality(tree, reference, braid=None, amphichiral=False):
    """Sign s (+1: tree, -1: mirror) such that H_[1](N) matches ``reference``.
    A tie (H_[1] mirror symmetric) is broken with H_[2], then H_[2,1], of the
    ``braid`` by cabling in multiplicity spaces; None if still undecided.  For
    an amphichiral knot both are right (+1)."""
    from ..algebra.fields import primes_below
    F = GF(primes_below(MO.SMALL_PRIME, 1)[0])
    rng = random.Random(hash(_freeze(tree)) & 0xffff)
    pts = [(F.random_element(rng), F.random_element(rng)) for _ in range(3)]
    trees = {1: _freeze(tree), -1: _freeze(conway.mirror(tree))}
    hits = [s for s in (1, -1)
            if all(value(trees[s], (1,), F, A, q) == reference.evaluate([A, q], one=F.one) for A, q in pts)]
    if not hits:
        raise ValueError("no chirality matches the reference")
    if len(hits) == 1:
        return hits[0]
    if amphichiral:                 # H_R(K) = H_R(mirror K) for all R
        return 1
    if braid is None:
        return None
    from .cabling_paths import CablingPaths
    cp = CablingPaths()
    A, q = pts[0]
    for R in ((2,), (2, 1)):
        if not cp.supports(braid, R):
            break
        ref = cp.evaluate(braid, R, F, A, q)
        hits = [s for s in (1, -1) if value(trees[s], R, F, A, q) == ref]
        if len(hits) == 1:
            return hits[0]
    return None


class AlgebraicMethod(Method):
    name = "algebraic"
    prime_bound = MO.NP_BOUND

    def __init__(self, root=MO.DATA_DIR):
        self.root = root

    def prime_bound_for(self, R):
        return MO.prime_bound_for(R)

    def supports(self, knot, R):
        av = MO.available(self.root)
        return isinstance(knot, AlgebraicKnot) and (P(R) in av or conjugate(P(R)) in av)

    def evaluate(self, knot, R, F, A, q):
        return value(knot.key(), R, F, A, q, self.root)
