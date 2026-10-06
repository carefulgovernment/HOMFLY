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
from fast_mixed_ctx import FastCtx, fapply_plan, apply_plan, _inv_mod


def fast_build_inclusive(M, R, ctx, sink, log=None, Qsel=None):
    from racah_num import tableau_path
    from mixedS import r1_eigenbasis
    n = sum(R)
    K = ctx.K
    E0, RR = ((), ()), (tuple(R), ())
    TRp = tableau_path(E0, R, 'V')
    Ys = [Y for (Y, m) in ctx.targets(RR, 'V')]
    G, Ginv, eig = {}, {}, {}
    for Y in Ys:
        g, signs, lam = r1_eigenbasis(ctx, R, Y)
        m = len(g)
        G[Y] = np.array([[g[i][k].v for k in range(m)] for i in range(m)], dtype=np.int64)  # (m, m, K): new a_k = sum_i G[i][k] a_i
        eig[Y] = np.stack([(s * lam).v for s in signs])
        Ginv[Y] = FastCtx._matinv(G[Y])
    blocks = {}
    for Y in Ys:
        for (Q, mq) in ctx.targets(Y, 'V'):
            blocks.setdefault(Q, []).append(Y)
    # columns per batch: the moved arrays are (keys, columns, K)
    budget = int(float(os.environ.get("FAST_BATCH_GB", "0.5")) * 2 ** 30)
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
        # initial keys (union over the columns) and the column entries
        k0, entries = {}, []
        for c, (Y, a, b) in enumerate(labels):
            bvec = ctx.basis(Y, Q, 'V')[0][b]
            for p, x in bvec.items():
                key = ((E0, RR, Y) + p[1:], (0, a) + (None,) * n)
                r = k0.get(key)
                if r is None:
                    r = k0[key] = len(k0)
                entries.append((r, c, x.v))
        keys = list(k0)
        er = np.array([e[0] for e in entries], dtype=np.int64)
        ec = np.array([e[1] for e in entries], dtype=np.int64)
        ev_ = np.stack([e[2] for e in entries])
        # the 2n moves, structure only; keys that vanish on a random combination
        # of all columns vanish on every column (w.h.p.) and are dropped
        rng = np.random.default_rng(len(keys) + 7 * m)
        rc = rng.integers(1, P, size=m)
        Ar = np.zeros((len(keys), 1, K), dtype=np.int64)
        np.add.at(Ar, (er, 0), ev_ * rc[ec][:, None] % P)
        Ar %= P
        plans, sizes = [], []
        kinds = ('FV', 'FV') + ('V',) * n
        for t in range(n):
            for pos in (t + 1, t):
                keys, plan, kinds = fapply_plan(ctx, keys, kinds, pos)
                Ar = apply_plan(plan, Ar, len(keys))
                keep = np.flatnonzero(Ar.any(axis=(1, 2)))
                if len(keep) < len(keys) and plan is not None:
                    src, starts, udst, C = plan
                    cnt = np.diff(np.r_[starts, len(src)])
                    dst = np.repeat(udst, cnt)
                    newi = np.full(len(keys), -1, dtype=np.int64)
                    newi[keep] = np.arange(len(keep))
                    sel = newi[dst] >= 0
                    src, dst, C = src[sel], newi[dst[sel]], C[sel]
                    starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]]) if len(dst) else dst
                    plan = (src, starts, dst[starts], C) if len(dst) else None
                    keys = [keys[r] for r in keep]
                    Ar = Ar[keep]
                plans.append(plan)
                sizes.append(len(keys))
        rows = np.full(len(keys), -1, dtype=np.int64)
        for r, (st, lb) in enumerate(keys):
            if st[:n + 1] == TRp and st[n + 2] == Q:      # other keys carry zero (no pruning)
                rows[r] = idx[(st[n + 1], lb[n], lb[n + 1])]
        live = np.flatnonzero(rows >= 0)
        dead = np.flatnonzero(rows < 0)
        width = max(1, min(m, budget // (8 * K * max(sizes + [len(k0)]))))
        U = np.zeros((m, m, K), dtype=np.int64)
        for c0 in range(0, m, width):
            c1 = min(m, c0 + width)
            sel = (ec >= c0) & (ec < c1)
            A = np.zeros((len(k0), c1 - c0, K), dtype=np.int64)
            np.add.at(A, (er[sel], ec[sel] - c0), ev_[sel])
            A %= P
            for plan, sz in zip(plans, sizes):
                A = apply_plan(plan, A, sz)
            assert not A[dead].any(), "nonzero outside the tableau path"
            Ub = np.zeros((m, c1 - c0, K), dtype=np.int64)
            np.add.at(Ub, rows[live], A[live])
            U[:, c0:c1] = Ub % P
        del plans
        # R1 eigenbasis on both sides: U_new = Gt^-1 U Gt, Gt block diagonal (acts on a for fixed Y, b)
        groups = {}
        for (Y, a, b), r in idx.items():
            groups.setdefault((Y, b), []).append(r)        # rows in order a = 0, 1, ...
        for (Y, b), rs in groups.items():                  # right: columns (in place per group)
            g = G[Y]
            old = U[:, rs].copy()                          # (m, len, K)
            for a, r in enumerate(rs):
                acc = np.zeros((m, K), dtype=np.int64)
                for a2 in range(len(rs)):
                    acc = (acc + old[:, a2] * g[a2, a] % P) % P
                U[:, r] = acc
        for (Y, b), rs in groups.items():                  # left: rows, with G^-1
            gi = Ginv[Y]
            old = U[rs].copy()                             # (len, m, K)
            for a, r in enumerate(rs):
                acc = np.zeros((m, K), dtype=np.int64)
                for a2 in range(len(rs)):
                    acc = (acc + old[a2] * gi[a, a2][None, :] % P) % P
                U[r] = acc
        Un = U
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


def matmul(A, B):  # noqa (kept for tests)
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
    """Tr of the product over word (1, 2, -1, -2) of R1^{+-1}, R2^{+-1}, R1 = diag(ev),
    R2 = U R1 U^-1; U (m,m,K), ev (m,K) -> (K,).  Conjugated by U: R2 -> D,
    R1 -> W = U^-1 D U (FLINT nmod_mat)."""
    import flint
    m, _, K = U.shape
    out = np.empty(K, dtype=np.int64)
    for k in range(K):
        e = ev[:, k] % P
        ei = _inv_mod(e)
        u = flint.nmod_mat(m, m, (U[:, :, k] % P).ravel().tolist(), P)
        need = {g for g in word if abs(g) == 1}
        W = {}
        for g in need:
            d = e if g > 0 else ei
            W[g] = u.solve(flint.nmod_mat(m, m, ((U[:, :, k] % P) * d[:, None] % P).ravel().tolist(), P))
        X = None
        for g in word:
            if abs(g) == 1:
                X = W[g] if X is None else X * W[g]
            else:
                d = e if g > 0 else ei
                if X is None:
                    X = flint.nmod_mat(m, m, np.diag(d).ravel().tolist(), P)
                else:
                    Xa = np.array([int(x) for x in X.entries()], dtype=np.int64).reshape(m, m)
                    X = flint.nmod_mat(m, m, (Xa * d[None, :] % P).ravel().tolist(), P)
        out[k] = int(sum(int(X[i, i]) for i in range(m)) % P)
    return out


def borromean_trace(U, ev):
    """Tr((R1 R2^-1)^3) = Tr((W D^-1)^3), W = U^-1 D U: one solve, one product."""
    import flint
    m, _, K = U.shape
    out = np.empty(K, dtype=np.int64)
    for k in range(K):
        e = ev[:, k] % P
        ei = _inv_mod(e)
        Uk = U[:, :, k] % P
        W = flint.nmod_mat(m, m, Uk.ravel().tolist(), P).solve(
            flint.nmod_mat(m, m, (Uk * e[:, None] % P).ravel().tolist(), P))
        Y = np.array([int(x) for x in W.entries()], dtype=np.int64).reshape(m, m) * ei[None, :] % P
        Yf = flint.nmod_mat(m, m, Y.ravel().tolist(), P)
        Y2 = np.array([int(x) for x in (Yf * Yf).entries()], dtype=np.int64).reshape(m, m)
        out[k] = int((Y2 * Y.T % P).sum() % P)
    return out
