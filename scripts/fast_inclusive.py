"""Inclusive Racah matrices U_Q of R x R x R -> Q at many points at once, with
the fast context of fast_mixed_ctx.py (port of the release's
fused_321_inclusive/incl.py: same path model, same bases and gauge).

  B_L[(Y,a,b)] = ((R R)_Y^a R)_Q^b,   B_R = image of the braiding that moves the
  last R to the front,  B_R = B_L U_Q;  the a-basis diagonalises R1.
  R1 = diag(ev) on (Y,a,b),  R2 = U_Q R1 U_Q^{-1}.

These matrices are independent of A (pure Hecke algebra), so every point only
needs its q.  The columns are grouped by Q, and a sink receives
(Q, labels, U (m, m, K), ev (m, K)) as soon as the block is complete.
"""
import os

import numpy as np

from fvn import P
from fast_mixed_ctx import FastCtx, fapply_arr, _inv_mod


def fast_build_inclusive(M, R, ctx, sink, log=None, Qsel=None):
    from racah_num import tableau_path
    from mixedS import r1_eigenbasis
    n = sum(R)
    K = ctx.K
    E0, RR = ((), ()), (tuple(R), ())
    TRp = tableau_path(E0, R, 'V')
    Ys = [Y for (Y, m) in ctx.targets(RR, 'V')]
    G, eig = {}, {}
    for Y in Ys:
        g, signs, lam = r1_eigenbasis(ctx, R, Y)
        m = len(g)
        G[Y] = np.array([[g[i][k].v for k in range(m)] for i in range(m)], dtype=np.int64)  # new a_k = sum_i G[i][k] a_i
        eig[Y] = np.stack([(s * lam).v for s in signs])
    blocks = {}
    for Y in Ys:
        for (Q, mq) in ctx.targets(Y, 'V'):
            blocks.setdefault(Q, []).append(Y)
    width = max(1, int(os.environ.get("FAST_COL_ELEMS", "480")) // K)
    for qi, Q in enumerate(sorted(blocks)):
        if Qsel is not None and not Qsel(Q):
            continue
        labels = []
        for Y in blocks[Q]:
            ma = len(ctx.basis(RR, Y, 'V')[0])
            mb = len(ctx.basis(Y, Q, 'V')[0])
            labels += [(Y, a, b) for a in range(ma) for b in range(mb)]
        idx = {l: k for k, l in enumerate(labels)}
        m = len(labels)
        U = np.zeros((m, m, K), dtype=np.int64)
        for c0 in range(0, m, width):
            cols = labels[c0:c0 + width]
            w = {}
            for g, (Y, a, b) in enumerate(cols):
                bvec = ctx.basis(Y, Q, 'V')[0][b]
                for p, x in bvec.items():
                    key = ((E0, RR, Y) + p[1:], (0, a) + (None,) * n)
                    arr = w.get(key)
                    if arr is None:
                        arr = w[key] = np.zeros((len(cols), K), dtype=np.int64)
                    arr[g] = (arr[g] + x.v) % P
            keys = list(w)
            A = np.stack([w[k] for k in keys])
            del w
            kinds = ('FV', 'FV') + ('V',) * n
            for t in range(n):
                keys, A, kinds = fapply_arr(ctx, keys, A, kinds, t + 1)
                keys, A, kinds = fapply_arr(ctx, keys, A, kinds, t)
            for (st, lb), arr in zip(keys, A):
                assert st[:n + 1] == TRp and st[n + 2] == Q
                r = idx[(st[n + 1], lb[n], lb[n + 1])]
                U[r, c0:c0 + len(cols)] = (U[r, c0:c0 + len(cols)] + arr) % P
        # R1 eigenbasis on both sides: U_new = Gt^-1 U Gt (Gt block diagonal in Y, acts on a)
        Gt = np.zeros((m, m, K), dtype=np.int64)
        for (Y, a, b), r in idx.items():
            for a2 in range(G[Y].shape[0]):
                Gt[idx[(Y, a2, b)], r] = G[Y][a2, a]
        Un = matmul(matinv(Gt), matmul(U, Gt))
        ev = np.stack([eig[Y][a] for (Y, a, b) in labels])
        sink(Q, labels, Un, ev)
        if log:
            log("Q %d/%d %s dim %d  bases %d fcross %d" % (qi + 1, len(blocks), Q, m, ctx.stats["bases"],
                                                           ctx.stats["fcross"]))


# ----------------------------------------------------------------------
# dense linear algebra mod P on batches (m, m, K): exact via float64 BLAS
# on 11-bit limbs
# ----------------------------------------------------------------------

def _limbs(X):
    X = X % P
    return [(X >> (11 * i)) & 0x7FF for i in range(3)]


def matmul(A, B):
    """(m, l, K) x (l, n, K) -> (m, n, K) mod P"""
    K = A.shape[2]
    out = np.empty((A.shape[0], B.shape[1], K), dtype=np.int64)
    for k in range(K):
        out[:, :, k] = matmul1(A[:, :, k], B[:, :, k])
    return out


def matmul1(A, B):
    la, lb = _limbs(A), _limbs(B)
    acc = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            c = (la[i].astype(np.float64) @ lb[j].astype(np.float64))      # < 2^22 * l, exact
            acc = (acc + (np.rint(c).astype(np.int64) % P) * pow(2, 11 * (i + j), P)) % P
    return acc


def matinv(A):
    K = A.shape[2]
    out = np.empty_like(A)
    for k in range(K):
        out[:, :, k] = matinv1(A[:, :, k])
    return out


def matinv1(A):
    m = A.shape[0]
    X = np.concatenate([A % P, np.eye(m, dtype=np.int64)], axis=1)
    for c in range(m):
        r = c + int(np.flatnonzero(X[c:, c])[0])
        if r != c:
            X[[c, r]] = X[[r, c]]
        X[c] = X[c] * _inv_mod(np.array([X[c, c]]))[0] % P
        f = X[:, c].copy()
        f[c] = 0
        nz = np.flatnonzero(f)
        if len(nz):
            X[nz] = (X[nz] - (f[nz, None] * X[c][None, :]) % P) % P
    return X[:, m:]


def word_trace(U, ev, word):
    """Tr of prod over word (1, 2, -1, -2) of R1^{+-1}, R2^{+-1}; U (m,m,K), ev (m,K) -> (K,)"""
    m, _, K = U.shape
    out = np.empty(K, dtype=np.int64)
    for k in range(K):
        u = U[:, :, k]
        ui = matinv1(u)
        e = ev[:, k] % P
        ei = _inv_mod(e)
        X = np.eye(m, dtype=np.int64)
        for g in word:
            if abs(g) == 1:
                X = X * (e if g > 0 else ei)[None, :] % P                       # X . R1^{+-1}
            else:
                d = e if g > 0 else ei
                X = matmul1(matmul1(X, u) * d[None, :] % P, ui)                # X . U D U^-1
        out[k] = int(np.trace(X) % P) if m < 1000 else int(sum(int(x) for x in np.diag(X)) % P)
    return out
