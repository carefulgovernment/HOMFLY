"""Reshetikhin--Turaev evaluation of an m-strand braid in the multiplicity
spaces of R^{⊗m}, fed by a RacahProvider (user tables, eigenvalue hypothesis,
fundamental closed forms, highest-weight method, ...).

    H^{unred}_{R,nat}(beta) = theta_R^{-w} sum_Q dim_q(Q) Tr_{M_Q}(rho(beta)),

M_Q = span of paths R = Y_1 -> Y_2 -> ... -> Y_m = Q, Y_{k+1} ∈ Y_k ⊗ R.
rho(sigma_1) is diagonal (lambda_{Y_2}); rho(sigma_k), k >= 2, acts on Y_k
with block U[Y_{k-1}, R, R -> Y_{k+1}] diag(lambda) U^{-1}.

3-strand: needs only U[R, R, R -> Q]  (the "inclusive" matrices of the tables).
4-strand: additionally U[Y, R, R -> Z] for Y ∈ R ⊗ R  (|Y| = 2|R|).
m-strand: U[Y, R, R -> Z] for all Y ∈ R^{⊗(m-2)}.

Status: implemented generically (labels are opaque, multiplicities allowed).
Validated with the fundamental provider against the Hecke method.
"""
from __future__ import annotations

from collections import defaultdict

from ..algebra.linalg import inverse
from ..conventions import natural_point
from ..knots.braid import Braid
from ..reps.qdim import qdim, theta
from .base import Method


class PathSpace:
    """Path basis of Hom(Q, R^{⊗m}) and the braid-generator action on it."""

    def __init__(self, provider, R, m, Q, F, A, q):
        self.provider, self.R, self.m, self.Q = provider, tuple(R), m, tuple(Q)
        self.F, self.A, self.q = F, A, q
        self.paths = [p for p in self._paths() if p[-1][0] == self.Q]
        self.index = {p: i for i, p in enumerate(self.paths)}
        self.lam = provider.eigenvalues(self.R, F, A, q)
        self._ops = {}

    def _paths(self):
        cur = [((self.R, 0),)]
        for _ in range(self.m - 1):
            cur = [p + (lab,) for p in cur for lab in self.provider.decompose(p[-1][0], self.R)]
        return cur

    def operator(self, k):
        """Sparse representation of rho(sigma_k): list of (i, [(j, coef)]),
        meaning (rho v)_i = sum coef * v_j."""
        if k in self._ops:
            return self._ops[k]
        F = self.F
        rows = [[] for _ in self.paths]
        if k == 1:
            for i, p in enumerate(self.paths):
                rows[i].append((i, self.lam[p[1]]))
        else:
            groups = defaultdict(list)
            for i, p in enumerate(self.paths):
                groups[p[:k - 1] + p[k:]].append(i)
            for key, idxs in groups.items():
                Y = self.paths[idxs[0]][k - 2][0]
                Z = self.paths[idxs[0]][k][0]
                labels, cols, U, Uinv = self.provider.inclusive(Y, self.R, Z, F, self.A, self.q)
                if Uinv is None:
                    Uinv = inverse(U, F.one)
                pos = {self.paths[i][k - 1]: i for i in idxs}
                lam = [self.lam[c] for c in cols]
                n = len(labels)
                B = [[sum((U[a][c] * lam[c] * Uinv[c][b] for c in range(len(cols))), F.zero)
                      for b in range(n)] for a in range(n)]
                for a in range(n):
                    ia = pos[labels[a]]
                    for b in range(n):
                        if B[a][b] != 0:
                            rows[ia].append((pos[labels[b]], B[a][b]))
        self._ops[k] = rows
        return rows

    def apply(self, k, v, inverse_=False):
        if inverse_:
            raise NotImplementedError("use apply_word, which handles inverses")
        return [sum((c * v[j] for j, c in row), self.F.zero) for row in self.operator(k)]

    def _inverse_operator(self, k):
        key = -k
        if key in self._ops:
            return self._ops[key]
        # inverse block-wise via dense inversion on connected blocks
        op = self.operator(k)
        F = self.F
        comp = {}
        blocks = []
        for i in range(len(op)):
            if i in comp:
                continue
            stack, members = [i], []
            comp[i] = len(blocks)
            while stack:
                x = stack.pop()
                members.append(x)
                for j, _ in op[x]:
                    if j not in comp:
                        comp[j] = len(blocks)
                        stack.append(j)
            blocks.append(sorted(members))
        rows = [[] for _ in op]
        for members in blocks:
            loc = {x: t for t, x in enumerate(members)}
            M = [[F.zero] * len(members) for _ in members]
            for x in members:
                for j, c in op[x]:
                    M[loc[x]][loc[j]] = c
            Mi = inverse(M, F.one)
            for x in members:
                for y in members:
                    if Mi[loc[x]][loc[y]] != 0:
                        rows[x].append((y, Mi[loc[x]][loc[y]]))
        self._ops[key] = rows
        return rows

    def apply_word(self, word, v):
        for a in reversed(word):
            op = self.operator(a) if a > 0 else self._inverse_operator(-a)
            v = [sum((c * v[j] for j, c in row), self.F.zero) for row in op]
        return v

    def trace(self, word):
        F = self.F
        tr = F.zero
        for i in range(len(self.paths)):
            v = [F.zero] * len(self.paths)
            v[i] = F.one
            tr = tr + self.apply_word(word, v)[i]
        return tr


def natural_unreduced(provider, braid, R, F, A, q):
    R = tuple(R)
    # all Q reachable
    Qs = set()
    cur = {R}
    for _ in range(braid.strands - 1):
        cur = {lab[0] for Y in cur for lab in provider.decompose(Y, R)}
    Qs = cur
    tot = F.zero
    for Q in sorted(Qs):
        S = PathSpace(provider, R, braid.strands, Q, F, A, q)
        if S.paths:
            tot = tot + qdim(Q, A, q) * S.trace(braid.word)
    return tot * theta(R, A, q) ** (-braid.writhe)


class RTBraid(Method):
    name = "rt-braid"

    def __init__(self, provider, max_strands=6):
        self.provider = provider
        self.max_strands = max_strands

    def supports(self, knot, R):
        return isinstance(knot, Braid) and knot.strands <= self.max_strands

    def evaluate(self, knot, R, F, A, q):
        A, q = natural_point(A, q)
        return natural_unreduced(self.provider, knot, R, F, A, q) / qdim(R, A, q)
