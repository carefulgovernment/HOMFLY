"""Exact linear algebra mod p (numpy int64; p < 2^31 so the products fit).  p is rebound by formula2.set_prime."""
import numpy as np

p = 67108859


def rref(M, ncols):
    """reduced row echelon form mod p; returns (R, pivot columns)"""
    M = np.array(M, dtype=np.int64).reshape(-1, ncols) % p
    m = M.shape[0]; piv = []; r = 0
    for col in range(ncols):
        if r >= m: break
        nz = np.flatnonzero(M[r:, col])
        if len(nz) == 0: continue
        pr = r + int(nz[0])
        if pr != r: M[[r, pr]] = M[[pr, r]]
        M[r] = M[r] * pow(int(M[r, col]), p - 2, p) % p
        f = M[:, col].copy(); f[r] = 0
        nzr = np.flatnonzero(f)
        if len(nzr): M[nzr] = (M[nzr] - (f[nzr, None] * M[r][None, :]) % p) % p
        piv.append(col); r += 1
    return M[:r], piv


def nullspace(rows, ncol):
    """basis of the right null space mod p of the matrix with the given rows, as lists"""
    if not len(rows): return [[1 if j == f else 0 for j in range(ncol)] for f in range(ncol)]
    R, piv = rref(rows, ncol)
    ps = set(piv); out = []
    for fj in range(ncol):
        if fj in ps: continue
        v = [0] * ncol; v[fj] = 1
        for i, pc in enumerate(piv): v[pc] = int(-R[i, fj]) % p
        out.append(v)
    return out
