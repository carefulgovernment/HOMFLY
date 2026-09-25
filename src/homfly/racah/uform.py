"""A-independent inclusive 3-strand Racah matrices U_Q (families G, F of the
release v1.0.0: [4,2], [2,2,1,1], [3,2,1]), stored as numpy archives by
scripts/import_racah_uform.py in data/racah/large/.

For each Q ⊢ 3|R|:  R1 = diag(sign q^k),  R2 = U_Q R1 U_Q⁻¹.  Since nothing but
dim_q(Q) depends on A, the traces t_Q(q) = Tr_Q(word) are computed once per q
value (cached) and

    H^{framed}(A, q) = sum_Q dim_q(Q; A, q) t_Q(q) / dim_q(R).

Arithmetic: residues mod a prime p < 2^21 held in float64, so that matrix
products run through BLAS and stay exact: n (p-1)^2 < 2^53 for blocks up to
n = 2048.  Methods using this module declare ``prime_bound = 2**21``.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from ..algebra.fields import BadPoint

LARGE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                          "data", "racah", "large"))
PRIME_BOUND = 2 ** 21


def rep_key(R):
    return "".join(map(str, R))


class UForm:
    def __init__(self, R, root=LARGE_DIR):
        d = np.load(os.path.join(root, "uform_%s.npz" % rep_key(R)))
        self.R = tuple(R)
        self.family = str(d["family"])
        self.Q = [tuple(int(x) for x in row if x) for row in d["Q"]]
        self.dims = [int(x) for x in d["dims"]]
        self.eig_s = d["eig_s"].astype(np.int64)
        self.eig_k = d["eig_k"].astype(np.int64)
        self.arr = {k: d[k] for k in d.files if k[:2] in ("Un", "Ud", "Vn", "Vd")}
        self.maxpow = int(max(self.arr[k].max() for k in self.arr if k.endswith("_pow")))
        self.offsets = np.concatenate([[0], np.cumsum([n * n for n in self.dims])])
        self.eoff = np.concatenate([[0], np.cumsum(self.dims)])
        self._cache = OrderedDict()

    def _eval(self, name, p, qpow):
        c = self.arr[name + "_coef"] % p
        v = c * qpow[self.arr[name + "_pow"]] % p
        s = np.add.reduceat(v, self.arr[name + "_ptr"][:-1]) % p
        return s

    def matrices(self, p, q):
        """Per block (R1 diag, R1^-1 diag, U, Uinv) as int64 arrays mod p."""
        qpow = np.ones(self.maxpow + 1, dtype=np.int64)
        for k in range(1, self.maxpow + 1):
            qpow[k] = qpow[k - 1] * q % p
        num_u, den_u = self._eval("Un", p, qpow), self._eval("Ud", p, qpow)
        num_v, den_v = self._eval("Vn", p, qpow), self._eval("Vd", p, qpow)
        if (den_u == 0).any() or (den_v == 0).any():
            raise BadPoint("pole of U_Q")
        U = num_u * _inv_vec(den_u, p) % p
        V = num_v * _inv_vec(den_v, p) % p
        qi = pow(q, -1, p)
        eig = np.array([s * pow(q if k >= 0 else qi, abs(int(k)), p) % p
                        for s, k in zip(self.eig_s, self.eig_k)], dtype=np.int64) % p
        eig_inv = _inv_vec(eig, p)
        out = []
        for b, n in enumerate(self.dims):
            o = self.offsets[b]
            e = self.eoff[b]
            out.append((eig[e:e + n], eig_inv[e:e + n],
                        U[o:o + n * n].reshape(n, n), V[o:o + n * n].reshape(n, n)))
        return out

    def traces(self, word, p, q):
        """[t_Q(q)] for the braid word (cached per (word, p, q))."""
        key = (tuple(word), p, q)
        if key in self._cache:
            return self._cache[key]
        res = []
        for d, di, U, V in self.matrices(p, q):
            g = {1: ("d", d), -1: ("d", di),
                 2: ("m", _mm(U * d % p, V, p)), -2: ("m", _mm(U * di % p, V, p))}
            P = None
            for a in word:
                kind, M = g[a]
                if P is None:
                    P = np.diag(M) if kind == "d" else M.copy()
                elif kind == "d":
                    P = P * M[None, :] % p
                else:
                    P = _mm(P, M, p)
            res.append(int(np.trace(P) % p))
        self._cache[key] = res
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return res


def _mm(X, Y, p):
    """Exact modular product via float64 BLAS (requires n (p-1)^2 < 2^53)."""
    n = X.shape[1]
    if n * (p - 1) ** 2 >= 2 ** 53:
        raise ValueError("block too large for exact float64 accumulation")
    Z = X.astype(np.float64) @ Y.astype(np.float64)
    return np.fmod(Z, p).astype(np.int64)


def _inv_vec(x, p):
    return np.array([pow(int(v), -1, p) if v else 0 for v in x], dtype=np.int64)


@lru_cache(maxsize=None)
def load(R, root=LARGE_DIR):
    return UForm(tuple(R), root)


def available(root=LARGE_DIR):
    if not os.path.isdir(root):
        return []
    return [tuple(int(c) for c in f[len("uform_"):-len(".npz")])
            for f in sorted(os.listdir(root)) if f.startswith("uform_") and f.endswith(".npz")]
