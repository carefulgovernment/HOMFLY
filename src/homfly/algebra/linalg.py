"""Tiny dense linear algebra over an arbitrary field (lists of lists).

Adequate for the small blocks appearing in multiplicity spaces.  For large
blocks (4+ strands, |R| >= 4) replace by python-flint ``nmod_mat`` -- the
functions below are the single place to swap.
"""
from __future__ import annotations

from .fields import BadPoint


def identity(n, one):
    z = one * 0
    return [[one if i == j else z for j in range(n)] for i in range(n)]


def matmul(X, Y):
    n, k, m = len(X), len(Y), len(Y[0]) if Y else 0
    out = []
    for i in range(n):
        row = X[i]
        r = []
        for j in range(m):
            s = row[0] * Y[0][j]
            for t in range(1, k):
                s = s + row[t] * Y[t][j]
            r.append(s)
        out.append(r)
    return out


def matvec(X, v):
    return [sum((x * y for x, y in zip(row[1:], v[1:])), row[0] * v[0]) for row in X]


def transpose(X):
    return [list(r) for r in zip(*X)]


def trace(X):
    s = X[0][0]
    for i in range(1, len(X)):
        s = s + X[i][i]
    return s


def diag(vals, zero):
    n = len(vals)
    return [[vals[i] if i == j else zero for j in range(n)] for i in range(n)]


def inverse(X, one):
    n = len(X)
    M = [list(X[i]) + identity(n, one)[i] for i in range(n)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            raise BadPoint("singular matrix")
        M[c], M[piv] = M[piv], M[c]
        inv = one / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def matpow(X, e, one):
    if e < 0:
        X, e = inverse(X, one), -e
    R = identity(len(X), one)
    while e:
        if e & 1:
            R = matmul(R, X)
        X = matmul(X, X)
        e >>= 1
    return R
