"""Fast drop-in replacement for the release engine's ``racah_fused.Ctx``
(code/mixed_S_gtpath), used to evaluate the mixed Racah matrix S of large
representations (|R| = 8) at many points.

Same generic-A mixed path model, same local R-matrix rules (ratmodel.cross_r),
same canonical bases (e_T applied to paths in lexicographic order, first
independent ones kept; pivots of the echelon form), but

  * skew paths are enumerated layer by layer with pruning (the release
    enumerates all paths of the segment and filters the end state);
  * a segment's generators are sparse transition tables built once, and the
    idempotent e_T acts on a whole batch of path vectors at once;
  * fcross (moving a single V step past a fused block) acts on numpy arrays;
  * every numeric value is an array over the K evaluation points.

Interface (as used by mixedS.build_mixed / r1_eigenbasis / sbar_vacuum_column):
  basis(start, end, kind) -> (vecs: [dict path -> FN], piv, mat)
  coords(start, end, kind, w) -> [FN]
  targets(start, kind) -> [(end, mult)]
  fcross_left / fcross_right -> {(mu', k'): FN}
"""
import numpy as np

from bip import steps
from fvn import FN, P
from ratmodel import cross_r


def _fv(x, K):
    return x.v if isinstance(x, FN) else np.full(K, int(x) % P, dtype=np.int64)


def _inv_mod(a):
    r = np.ones_like(a)
    b = a % P
    e = P - 2
    while e:
        if e & 1:
            r = r * b % P
        b = b * b % P
        e >>= 1
    return r


def _cells(R):
    return [(r, c) for r, length in enumerate(R) for c in range(length)]


class FastCtx:
    def __init__(self, M, R):
        self.M, self.R, self.n = M, tuple(R), sum(R)
        self.K = len(M.q.v)
        self.zero = M.one * 0
        self.fb = {}          # (start, end, kind) -> (vecs dicts | None, piv, matinv (m,m,K), m)
        self.fc = {}
        self._vec_cache = {}  # (start, end, kind) -> (paths, idx, V (N, m, K))
        self._gen = {}        # (paths key) -> generator tables
        self.stats = {"bases": 0, "fcross": 0}

    # ------------------------------------------------------------------
    # skew paths
    # ------------------------------------------------------------------
    def _layers(self, start, kind, n):
        lay = [{start}]
        for _ in range(n):
            nxt = set()
            for s in lay[-1]:
                for m, _ in steps(s, kind):
                    nxt.add(m)
            lay.append(nxt)
        return lay

    def targets(self, start, kind):
        ends = sorted(self._layers(start, kind, self.n)[-1])
        out = []
        for e in ends:
            m = self._basis_arrays(start, e, kind)[2].shape[1]
            if m:
                out.append((e, m))
        return out

    def _paths(self, start, end, kind, n=None):
        n = self.n if n is None else n
        lay = self._layers(start, kind, n)
        if end not in lay[-1]:
            return []
        good = [None] * (n + 1)
        good[n] = {end}
        for t in range(n - 1, -1, -1):
            good[t] = {s for s in lay[t] if any(m in good[t + 1] for m, _ in steps(s, kind))}
        out = []

        def rec(p, t):
            if t == n:
                out.append(tuple(p))
                return
            for m, _ in steps(p[-1], kind):
                if m in good[t + 1]:
                    p.append(m)
                    rec(p, t + 1)
                    p.pop()
        rec([start], 0)
        out.sort()
        return out

    # ------------------------------------------------------------------
    # generator action on arrays of path vectors
    # ------------------------------------------------------------------
    def _apply_gen(self, paths, idx, V, kinds, i, inv=False, new_paths=None):
        """V: (N, ..., K) over `paths`; returns (paths', idx', V') after the
        generator at position i (kinds swapped)."""
        k1, k2 = kinds[i], kinds[i + 1]
        M, K = self.M, self.K
        src, dst, coef = [], [], []
        nidx = {} if new_paths is None else None
        npaths = [] if new_paths is None else new_paths
        if new_paths is not None:
            nidx = {p: j for j, p in enumerate(new_paths)}
        for r, p in enumerate(paths):
            for mp, x in cross_r(M, p[i], p[i + 1], p[i + 2], k1, k2, inv).items():
                q_ = p[:i + 1] + (mp,) + p[i + 2:]
                j = nidx.get(q_)
                if j is None:
                    if new_paths is not None:
                        continue      # outside the target space (zero by construction)
                    j = nidx[q_] = len(npaths)
                    npaths.append(q_)
                src.append(r)
                dst.append(j)
                coef.append(_fv(x, K))
        out = np.zeros((len(npaths),) + V.shape[1:], dtype=np.int64)
        if src:
            src = np.array(src)
            dst = np.array(dst)
            C = np.stack(coef)                                 # (nnz, K)
            order = np.argsort(dst, kind="stable")
            src, dst, C = src[order], dst[order], C[order]
            starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
            C = C.reshape((len(src),) + (1,) * (V.ndim - 2) + (K,))
            out[dst[starts]] = np.add.reduceat(V[src] * C % P, starts, axis=0) % P
        nk = kinds[:i] + (k2, k1) + kinds[i + 2:]
        return npaths, nidx, out, nk

    def _block_gens(self, paths, idx, kind):
        key = (paths[0][0], paths[0][-1], kind, len(paths))
        if key in self._gen:
            return self._gen[key]
        M, K = self.M, self.K
        tabs = []
        for i in range(self.n - 1):
            src, dst, coef = [], [], []
            for r, p in enumerate(paths):
                for mp, x in cross_r(M, p[i], p[i + 1], p[i + 2], kind, kind, False).items():
                    q_ = p[:i + 1] + (mp,) + p[i + 2:]
                    j = idx.get(q_)
                    if j is None:
                        continue
                    src.append(r)
                    dst.append(j)
                    coef.append(_fv(x, K))
            src = np.array(src, dtype=np.int64)
            dst = np.array(dst, dtype=np.int64)
            C = np.stack(coef) if coef else np.zeros((0, K), dtype=np.int64)
            order = np.argsort(dst, kind="stable")
            src, dst, C = src[order], dst[order], C[order]
            starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]]) if len(dst) else np.zeros(0, np.int64)
            tabs.append((src, dst, C, starts, dst[starts] if len(dst) else dst))
        self._gen[key] = tabs
        return tabs

    @staticmethod
    def _gen_apply(tab, V, N):
        src, dst, C, starts, udst = tab
        out = np.zeros((N,) + V.shape[1:], dtype=np.int64)
        if len(src):
            Cb = C.reshape((len(src),) + (1,) * (V.ndim - 2) + (C.shape[-1],))
            out[udst] = np.add.reduceat(V[src] * Cb % P, starts, axis=0) % P
        return out

    def _idempotent(self, tabs, V, N):
        """e_T (row reading tableau of R) on a batch V (N, B, K)."""
        q = self.M.q
        cells = _cells(self.R)
        shape = []
        for j in range(1, len(cells) + 1):
            r, c = cells[j - 1]
            if j >= 2:
                t = c - r
                others = []
                for rr in range(len(shape) + 1):
                    cur = shape[rr] if rr < len(shape) else 0
                    if rr == 0 or shape[rr - 1] > cur:
                        cc = cur - rr
                        if cc != t:
                            others.append(cc)
                for cc in others:
                    if not V.any():
                        break
                    # L_j: generators j-2, ..., 0, 0, ..., j-2 (block positions)
                    Lv = V
                    for i in list(range(j - 2, -1, -1)) + list(range(0, j - 1)):
                        Lv = self._gen_apply(tabs[i], Lv, N)
                    mc = (-(q ** (2 * cc))).v
                    fac = (1 / (q ** (2 * t) - q ** (2 * cc))).v
                    V = (Lv + V * mc % P) % P * fac % P
            if r == len(shape):
                shape.append(1)
            else:
                shape[r] += 1
        return V

    # ------------------------------------------------------------------
    # fused bases
    # ------------------------------------------------------------------
    RECURSIVE = True

    def _basis_arrays(self, start, end, kind):
        key = (start, end, kind)
        if key in self._vec_cache:
            return self._vec_cache[key]
        if self.RECURSIVE:
            if self._expected(start, end, kind) == 0:
                res = ([], {}, np.zeros((0, 0, self.K), dtype=np.int64), [],
                       np.zeros((0, 0, self.K), dtype=np.int64))
                self._vec_cache[key] = res
                return res
            self._all_bases(start, kind, only=end)
            return self._vec_cache[key]
        self.stats["bases"] += 1
        paths = self._paths(start, end, kind)
        N, K = len(paths), self.K
        if N == 0:
            res = (paths, {}, np.zeros((0, 0, K), dtype=np.int64), [], None)
            self._vec_cache[key] = res
            return res
        idx = {p: r for r, p in enumerate(paths)}
        want = self._expected(start, end, kind)
        if want == 0:
            res = (paths, idx, np.zeros((N, 0, K), dtype=np.int64), [], np.zeros((0, 0, K), dtype=np.int64))
            self._vec_cache[key] = res
            return res
        tabs = self._block_gens(paths, idx, kind)
        # candidates: unit vectors of the paths in order, in batches
        kept, piv_rows = [], []          # kept: list of (N, K) arrays
        ech = []                         # echelon rows at point 0: (pivot, row vector over N) mod P
        B = want + 2
        start_i = 0
        while start_i < N:
            batch = list(range(start_i, min(N, start_i + B)))
            V = np.zeros((N, len(batch), K), dtype=np.int64)
            V[batch, np.arange(len(batch)), :] = 1
            W = self._idempotent(tabs, V, N)
            for b in range(len(batch)):
                w = W[:, b, :]
                nz = np.flatnonzero(w.any(axis=1))
                if len(nz) == 0:
                    continue
                # independence test (exact, at all points via point 0 + generic)
                x = w[:, 0].copy()
                for pk, row in ech:
                    c = x[pk]
                    if c:
                        x = (x - c * row) % P
                nzx = np.flatnonzero(x)
                if len(nzx) == 0:
                    continue
                pk = int(nzx[0])
                x = x * pow(int(x[pk]), P - 2, P) % P
                ech.append((pk, x))
                kept.append(w)
                if len(kept) == want:
                    break
            start_i += B
            if len(kept) >= want:
                break
        m = len(kept)
        Vk = np.stack(kept, axis=1) if m else np.zeros((N, 0, K), dtype=np.int64)
        # pivots as in racah_rat.pivots: echelon over the kept vectors, min key
        piv = []
        E = []
        for i in range(m):
            w = Vk[:, i, :].copy()
            for pk, row in E:
                c = w[pk]
                if c.any():
                    w = (w - c * row) % P
            nz = np.flatnonzero(w.any(axis=1))
            pk = int(nz[0])
            ci = _inv_mod(w[pk])
            E.append((pk, w * ci % P))
            piv.append(pk)
        mat = Vk[piv, :, :] if m else np.zeros((0, 0, K), dtype=np.int64)   # mat[k][i] = v_i(piv_k)
        minv = self._matinv(mat) if m else mat
        res = (paths, idx, Vk, piv, minv)
        self._vec_cache[key] = res
        return res

    # ------------------------------------------------------------------
    # recursive construction of all fused bases from one start state
    # ------------------------------------------------------------------
    def _all_bases(self, start, kind, only=None):
        """image of e_T built step by step (Young seminormal recursion): after j
        steps a basis of e_{T_j}(paths of length j) per end state; extend by one
        step and project only with the L_j factors.  Fills _vec_cache for every
        end with nonzero multiplicity."""
        n, K, q = self.n, self.K, self.M.q
        lay = self._layers(start, kind, n)
        ends = [e for e in lay[n] if self._expected(start, e, kind) > 0] if only is None else [only]
        good = [None] * (n + 1)
        good[n] = set(ends)
        for t in range(n - 1, -1, -1):
            good[t] = {x for x in lay[t] if any(m in good[t + 1] for m, _ in steps(x, kind))}
        cells = _cells(self.R)
        # U: end state -> (paths list, idx, vectors (N, d, K))
        U = {start: ([(start,)], {(start,): 0}, np.ones((1, 1, K), dtype=np.int64))}
        shape = []
        for j in range(1, n + 1):
            r, c = cells[j - 1]
            t = c - r
            others = []
            if j >= 2:
                for rr in range(len(shape) + 1):
                    cur = shape[rr] if rr < len(shape) else 0
                    if rr == 0 or shape[rr - 1] > cur:
                        cc = cur - rr
                        if cc != t:
                            others.append(cc)
            # extend
            ext = {}
            for s0, (pl, pidx, V) in U.items():
                for s1, _ in steps(s0, kind):
                    if s1 not in good[j]:
                        continue
                    ext.setdefault(s1, []).append((pl, V))
            newU = {}
            for s1, parts in ext.items():
                # all paths of length j to s1 (generators leave the supports)
                paths = self._paths(start, s1, kind, n=j)
                idx = {p: k for k, p in enumerate(paths)}
                d = sum(V.shape[1] for _, V in parts)
                W = np.zeros((len(paths), d, K), dtype=np.int64)
                col = 0
                for pl, V in parts:
                    rows = [idx[p + (s1,)] for p in pl]
                    W[rows, col:col + V.shape[1]] = V
                    col += V.shape[1]
                if others:
                    tabs = self._prefix_gens(paths, idx, kind, j)
                    for cc in others:
                        Lv = W
                        for i in list(range(j - 2, -1, -1)) + list(range(0, j - 1)):
                            Lv = self._gen_apply(tabs[i], Lv, len(paths))
                        mc = (-(q ** (2 * cc))).v
                        fac = (1 / (q ** (2 * t) - q ** (2 * cc))).v
                        W = (Lv + W * mc % P) % P * fac % P
                # basis of the span (rank at point 0)
                keep, ech = [], []
                for b in range(W.shape[1]):
                    x = W[:, b, 0].copy()
                    for pk, row in ech:
                        cx = x[pk]
                        if cx:
                            x = (x - cx * row) % P
                    nz = np.flatnonzero(x)
                    if len(nz) == 0:
                        continue
                    pk = int(nz[0])
                    ech.append((pk, x * pow(int(x[pk]), P - 2, P) % P))
                    keep.append(b)
                if keep:
                    newU[s1] = (paths, idx, W[:, keep, :])
            U = newU
            if r == len(shape):
                shape.append(1)
            else:
                shape[r] += 1
        for e in ends:
            key = (start, e, kind)
            if key in self._vec_cache:
                continue
            if e not in U:
                self._vec_cache[key] = ([], {}, np.zeros((0, 0, K), dtype=np.int64), [],
                                        np.zeros((0, 0, K), dtype=np.int64))
                continue
            paths, idx, Vk = U[e]
            m = Vk.shape[1]
            assert m == self._expected(start, e, kind), (start, e, m)
            piv, E = [], []
            for i in range(m):
                w = Vk[:, i, :].copy()
                for pk, row in E:
                    cw = w[pk]
                    if cw.any():
                        w = (w - cw * row) % P
                nz = np.flatnonzero(w.any(axis=1))
                pk = int(nz[0])
                E.append((pk, w * _inv_mod(w[pk]) % P))
                piv.append(pk)
            mat = Vk[piv, :, :]
            self._vec_cache[key] = (paths, idx, Vk, piv, self._matinv(mat))
            self.stats["bases"] += 1

    def _prefix_gens(self, paths, idx, kind, j):
        M, K = self.M, self.K
        tabs = []
        for i in range(j - 1):
            src, dst, coef = [], [], []
            for r, p in enumerate(paths):
                for mp, x in cross_r(M, p[i], p[i + 1], p[i + 2], kind, kind, False).items():
                    q_ = p[:i + 1] + (mp,) + p[i + 2:]
                    jj = idx.get(q_)
                    if jj is None:
                        continue
                    src.append(r)
                    dst.append(jj)
                    coef.append(_fv(x, K))
            src = np.array(src, dtype=np.int64)
            dst = np.array(dst, dtype=np.int64)
            C = np.stack(coef) if coef else np.zeros((0, K), dtype=np.int64)
            order = np.argsort(dst, kind="stable")
            src, dst, C = src[order], dst[order], C[order]
            starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]]) if len(dst) else np.zeros(0, np.int64)
            tabs.append((src, dst, C, starts, dst[starts] if len(dst) else dst))
        return tabs

    def _expected(self, start, end, kind):
        from racah_rat import expected_mult
        key = ("mult", start, end, kind)
        if key not in self.fb:
            self.fb[key] = expected_mult(start, end, self.R, kind)
        return self.fb[key]

    @staticmethod
    def _matinv(A):
        """inverse of (m, m, K) batch mod P (Gauss-Jordan, pivots at point 0; generic)."""
        m = A.shape[0]
        K = A.shape[2]
        X = np.concatenate([A % P, np.broadcast_to(np.eye(m, dtype=np.int64)[:, :, None], (m, m, K))], axis=1).copy()
        for c in range(m):
            r = next(rr for rr in range(c, m) if X[rr, c].all())
            if r != c:
                X[[c, r]] = X[[r, c]]
            X[c] = X[c] * _inv_mod(X[c, c]) % P
            for rr in range(m):
                if rr != c:
                    f = X[rr, c].copy()
                    if f.any():
                        X[rr] = (X[rr] - f * X[c]) % P
        return X[:, m:, :]

    def basis(self, start, end, kind):
        paths, idx, Vk, piv, minv = self._basis_arrays(start, end, kind)
        key = ("dicts", start, end, kind)
        if key not in self.fb:
            vecs = []
            for i in range(Vk.shape[1]):
                col = Vk[:, i, :]
                nz = np.flatnonzero(col.any(axis=1))
                vecs.append({paths[r]: FN(col[r].copy()) for r in nz})
            self.fb[key] = (vecs, [paths[r] for r in piv], None)
        return self.fb[key]

    def _coords_arr(self, start, end, kind, W, pathlist):
        """coordinates of vectors W (len(pathlist), B, K) given over pathlist."""
        paths, idx, Vk, piv, minv = self._basis_arrays(start, end, kind)
        m = Vk.shape[1]
        B = W.shape[1]
        if m == 0:
            assert not W.any(), "vector outside fused space"
            return np.zeros((0, B, self.K), dtype=np.int64)
        pos = {p: r for r, p in enumerate(pathlist)}
        rhs = np.zeros((m, B, self.K), dtype=np.int64)
        for k, r in enumerate(piv):
            j = pos.get(paths[r])
            if j is not None:
                rhs[k] = W[j]
        # coords = minv @ rhs per point
        out = np.zeros((m, B, self.K), dtype=np.int64)
        for a in range(m):
            acc = np.zeros((B, self.K), dtype=np.int64)
            for b in range(m):
                acc = (acc + minv[a, b][None, :] * rhs[b] % P) % P
            out[a] = acc
        return out

    def coords(self, start, end, kind, w):
        paths, idx, Vk, piv, minv = self._basis_arrays(start, end, kind)
        m = Vk.shape[1]
        if m == 0:
            assert not w, "vector outside fused space"
            return []
        plist = list(w)
        W = np.stack([_fv(w[p], self.K) for p in plist])[:, None, :]
        c = self._coords_arr(start, end, kind, W, plist)
        return [FN(c[a, 0].copy()) for a in range(m)]

    # ------------------------------------------------------------------
    # fused block <-> single V step
    # ------------------------------------------------------------------
    def _fcross(self, side, lam, mu, nu, K, inv):
        """all basis vectors k at once; returns list over k of {(mu', k'): FN}."""
        n = self.n
        if side == "L":
            paths, idx, Vk, piv, minv = self._basis_arrays(lam, mu, K)
            cur = [p + (nu,) for p in paths]
            kinds = (K,) * n + ("V",)
            order = range(n - 1, -1, -1)
        else:
            paths, idx, Vk, piv, minv = self._basis_arrays(mu, nu, K)
            cur = [(lam,) + p for p in paths]
            kinds = ("V",) + (K,) * n
            order = range(0, n)
        V = Vk.copy()                      # (N, m, K)
        cidx = {p: r for r, p in enumerate(cur)}
        for i in order:
            cur, cidx, V, kinds = self._apply_gen(cur, cidx, V, kinds, i, inv)
            keep = np.flatnonzero(V.any(axis=(1, 2)))
            if len(keep) < len(cur):
                cur = [cur[r] for r in keep]
                V = V[keep]
                cidx = {p: r for r, p in enumerate(cur)}
        m = V.shape[1]
        groups = {}
        for r, p in enumerate(cur):
            g = p[1] if side == "L" else p[n]
            groups.setdefault(g, []).append(r)
        outs = [dict() for _ in range(m)]
        for mp, rows in groups.items():
            sub = V[rows]                                  # (len, m, K)
            plist = [cur[r][1:] if side == "L" else cur[r][:n + 1] for r in rows]
            if side == "L":
                cs = self._coords_arr(mp, nu, K, sub, plist)
            else:
                cs = self._coords_arr(lam, mp, K, sub, plist)
            for kk in range(cs.shape[0]):
                for k in range(m):
                    c = cs[kk, k]
                    if c.any():
                        outs[k][(mp, kk)] = FN(c.copy())
        return outs

    def fcross_left(self, lam, mu, nu, K, k, inv=False):
        key = ("L", lam, mu, nu, K, inv)
        if key not in self.fc:
            self.stats["fcross"] += 1
            self.fc[key] = self._fcross("L", lam, mu, nu, K, inv)
        return self.fc[key][k]

    def fcross_right(self, lam, mu, nu, K, k, inv=False):
        key = ("R", lam, mu, nu, K, inv)
        if key not in self.fc:
            self.stats["fcross"] += 1
            self.fc[key] = self._fcross("R", lam, mu, nu, K, inv)
        return self.fc[key][k]


# ----------------------------------------------------------------------
# build_mixed / gauge_S with all columns at once (numpy)
# ----------------------------------------------------------------------

def fapply_vec(ctx, vec, kinds, i):
    """racah_fused.fapply on {(states, labels): array (C, K)}."""
    k1, k2 = kinds[i], kinds[i + 1]
    out = {}
    for (st, lb), c in vec.items():
        lam, mu, nu = st[i], st[i + 1], st[i + 2]
        if k1 != 'V' and k2 == 'V':
            res = ctx.fcross_left(lam, mu, nu, k1[1], lb[i])
            mk = lambda mp, kk: (st[:i + 1] + (mp,) + st[i + 2:], lb[:i] + (None, kk) + lb[i + 2:])
        elif k1 == 'V' and k2 != 'V':
            res = ctx.fcross_right(lam, mu, nu, k2[1], lb[i + 1])
            mk = lambda mp, kk: (st[:i + 1] + (mp,) + st[i + 2:], lb[:i] + (kk, None) + lb[i + 2:])
        else:
            raise ValueError(kinds)
        for (mp, kk), x in res.items():
            key = mk(mp, kk)
            t = c * x.v % P
            w = out.get(key)
            out[key] = t if w is None else (w + t) % P
    nk = kinds[:i] + (k2, k1) + kinds[i + 2:]
    return {k: v for k, v in out.items() if v.any()}, nk


def fast_build_mixed(M, R, ctx, log=None):
    from racah_num import tableau_path
    from mixedS import r1_eigenbasis
    n = sum(R)
    K = ctx.K
    E0, RR = ((), ()), (tuple(R), ())
    TRp = tableau_path(E0, R, 'V')
    Xs = [X for (X, m) in ctx.targets(RR, 'W')]
    xlab = []
    for X in Xs:
        m = len(ctx.basis(RR, X, 'W')[0])
        assert m == len(ctx.basis(X, RR, 'V')[0])
        xlab += [(X, i, j) for i in range(m) for j in range(m)]
    Qs = [Q for (Q, m) in ctx.targets(RR, 'V')]
    qlab = []
    for Q in Qs:
        ma = len(ctx.basis(RR, Q, 'V')[0])
        mb = len(ctx.basis(Q, RR, 'W')[0])
        assert ma == mb
        qlab += [(Q, a, b) for a in range(ma) for b in range(mb)]
    N = len(qlab)
    assert N == len(xlab)
    qidx = {l: k for k, l in enumerate(qlab)}
    C = len(xlab)
    S = np.zeros((N, C, K), dtype=np.int64)
    # columns grouped by X: one group shares its keys, so arrays stay small
    groups = {}
    for c, (X, i, j) in enumerate(xlab):
        groups.setdefault(X, []).append((c, i, j))
    for gi, (X, cols) in enumerate(groups.items()):
        Cg = len(cols)
        w = {}
        for g, (c, i, j) in enumerate(cols):
            bvec = ctx.basis(X, RR, 'V')[0][j]
            for p, x in bvec.items():
                key = ((E0, RR, X) + p[1:], (0, i) + (None,) * n)
                arr = w.get(key)
                if arr is None:
                    arr = w[key] = np.zeros((Cg, K), dtype=np.int64)
                arr[g] = (arr[g] + x.v) % P
        kinds = ('FV', 'FW') + ('V',) * n
        for t in range(n):
            w, kinds = fapply_vec(ctx, w, kinds, t + 1)
            w, kinds = fapply_vec(ctx, w, kinds, t)
        assert kinds == ('V',) * n + ('FV', 'FW')
        idxs = [c for c, _, _ in cols]
        for (st, lb), arr in w.items():
            assert st[:n + 1] == TRp and st[n + 2] == RR
            r = qidx[(st[n + 1], lb[n], lb[n + 1])]
            S[r, idxs] = (S[r, idxs] + arr) % P
        if log:
            log("X %d/%d %s: %d columns, bases %d fcross %d" % (gi + 1, len(groups), X, Cg,
                                                              ctx.stats["bases"], ctx.stats["fcross"]))
    # R1 eigenbasis on the a index: S_new[(Q,a,b)] = sum_a2 Ginv[a][a2] S[(Q,a2,b)]
    ev = np.zeros((N, K), dtype=np.int64)
    S2 = S.copy()
    for Q in Qs:
        G, signs, lam = r1_eigenbasis(ctx, R, Q)
        m = len(G)
        Gm = np.array([[G[a2][a].v for a in range(m)] for a2 in range(m)], dtype=np.int64)  # Gm[a2][a]
        Ginv = FastCtx._matinv(Gm)                                                         # (m,m,K)
        for b in range(m):
            rows = [qidx[(Q, a2, b)] for a2 in range(m)]
            for a in range(m):
                acc = np.zeros((C, K), dtype=np.int64)
                for a2 in range(m):
                    acc = (acc + Ginv[a, a2][None, :] * S[rows[a2]] % P) % P
                S2[qidx[(Q, a, b)]] = acc
        for (Q2, a, b) in qlab:
            if Q2 == Q:
                ev[qidx[(Q, a, b)]] = (signs[a] * lam).v
    return qlab, xlab, S2, ev


def fast_gauge_S(qlab, xlab, S, col0, K):
    """mixedS.gauge_S on numpy arrays S (N, N, K)."""
    E0 = ((), ())
    N = len(xlab)
    xidx = {l: k for k, l in enumerate(xlab)}
    s00inv = _inv_mod(col0[(E0, 0, 0)].v)
    blocks = {}
    for (X, i, j) in xlab:
        blocks[X] = max(blocks.get(X, 0), i + 1)
    zero = np.zeros(K, dtype=np.int64)
    S = S.copy()
    for X, m in blocks.items():
        V = np.array([[(col0[(X, i, j)].v if (X, i, j) in col0 else zero) * s00inv % P for j in range(m)]
                      for i in range(m)], dtype=np.int64)          # V[i][j]
        W = V.transpose(1, 0, 2)                                      # W[j][k] = V[k][j]
        cols = {(i, j): xidx[(X, i, j)] for i in range(m) for j in range(m)}
        old = {ij: S[:, c].copy() for ij, c in cols.items()}
        for i in range(m):
            for k in range(m):
                acc = np.zeros((N, K), dtype=np.int64)
                for j in range(m):
                    acc = (acc + old[(i, j)] * W[j, k][None, :] % P) % P
                S[:, cols[(i, k)]] = acc
    qidx = {l: k for k, l in enumerate(qlab)}
    iv = xidx[(E0, 0, 0)]
    qb = {}
    for (Q, a, b) in qlab:
        qb[Q] = max(qb.get(Q, 0), a + 1)
    for Q, m in qb.items():
        U = np.array([[S[qidx[(Q, a, b)], iv] for b in range(m)] for a in range(m)], dtype=np.int64)
        Ui = FastCtx._matinv(U)
        rows = {(a, b): qidx[(Q, a, b)] for a in range(m) for b in range(m)}
        old = {ab: S[r].copy() for ab, r in rows.items()}
        for a in range(m):
            for k in range(m):
                acc = np.zeros((N, K), dtype=np.int64)
                for b in range(m):
                    acc = (acc + old[(a, b)] * Ui[b, k][None, :] % P) % P
                S[rows[(a, k)]] = acc
    return S
