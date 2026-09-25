"""Hecke algebra H_n(q) in Young's (rational) seminormal form.

Generators g_1..g_{n-1} satisfy the braid relations and
(g - q)(g + q^{-1}) = 0.  On the irreducible module S^Q (Q |- n) with basis of
standard Young tableaux T (encoded as row paths, see partitions.standard_tableaux)

    g_k v_T = (q^d / [d]) v_T + b_T v_{s_k T},       d = c_T(k+1) - c_T(k),

where c_T(i) is the content of the box holding i, s_k T swaps k and k+1, and
b_T = 1 if k+1 lies in a lower row than k in T, b_T = [d-1][d+1]/[d]^2
otherwise.  (If s_k T is not standard then d = +-1 and v_T is an eigenvector.)
No square roots appear -- the representation is defined over Q(q), hence
over every GF(p) point where [d] != 0.

This module is the engine of
  * methods.hecke_fundamental  (fundamental HOMFLY of any braid), and
  * methods.cabling            (colored HOMFLY via cabling + idempotents).
"""
from __future__ import annotations

from ..algebra.fields import BadPoint
from ..algebra.quantum import qnum
from ..reps.partitions import standard_tableaux, tableau_contents


class SeminormalModule:
    def __init__(self, Q, q, basis=None):
        self.Q = tuple(Q)
        self.n = sum(self.Q)
        self.q = q
        self.basis = list(basis if basis is not None else standard_tableaux(self.Q))
        self.index = {t: i for i, t in enumerate(self.basis)}
        self._gen = {}
        self.z = q - q ** -1

    def dim(self):
        return len(self.basis)

    def generator(self, k):
        """Data (diag, partner, offdiag) for g_k, 1 <= k < n."""
        if k in self._gen:
            return self._gen[k]
        q = self.q
        qn = {}
        diag, partner, off = [], [], []
        for t in self.basis:
            cs = tableau_contents(t)
            d = cs[k] - cs[k - 1]
            if d not in qn:
                qn[d] = qnum(d, q)
                if qn[d] == 0:
                    raise BadPoint("[%d]_q = 0" % d)
            diag.append(q ** d / qn[d])
            if abs(d) > 1:
                s = list(t)
                s[k - 1], s[k] = s[k], s[k - 1]
                j = self.index[tuple(s)]
                partner.append(j)
                if t[k] > t[k - 1]:  # k+1 in a lower row
                    off.append(q ** 0)
                else:
                    if d - 1 not in qn:
                        qn[d - 1] = qnum(d - 1, q)
                    if d + 1 not in qn:
                        qn[d + 1] = qnum(d + 1, q)
                    off.append(qn[d - 1] * qn[d + 1] / (qn[d] * qn[d]))
            else:
                partner.append(-1)
                off.append(None)
        self._gen[k] = (diag, partner, off)
        return self._gen[k]

    def apply(self, k, v, inverse=False):
        """Return g_k^{+-1} v for a coordinate vector v (list)."""
        diag, partner, off = self.generator(abs(k))
        out = [None] * len(v)
        for i in range(len(v)):
            s = diag[i] * v[i]
            j = partner[i]
            if j >= 0:
                s = s + off[j] * v[j]
            if inverse:
                s = s - self.z * v[i]
            out[i] = s
        return out

    def apply_word(self, word, v):
        """Apply rho(sigma_{w1} sigma_{w2} ...) to v; negative letters are
        inverses.  The rightmost letter acts first."""
        for a in reversed(word):
            v = self.apply(abs(a), v, inverse=a < 0)
        return v

    def trace_word(self, word, support=None):
        """Tr(E rho(word)) where E is the diagonal projector onto ``support``
        (all basis vectors if None)."""
        zero = self.q * 0
        one = zero + 1
        idx = range(len(self.basis)) if support is None else support
        tr = zero
        for i in idx:
            v = [zero] * len(self.basis)
            v[i] = one
            tr = tr + self.apply_word(word, v)[i]
        return tr

    def matrix(self, k):
        """Dense matrix of g_k (for tests / small debugging)."""
        n = len(self.basis)
        zero = self.q * 0
        cols = []
        for i in range(n):
            e = [zero] * n
            e[i] = zero + 1
            cols.append(self.apply(k, e))
        return [[cols[j][i] for j in range(n)] for i in range(n)]
