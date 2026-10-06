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
import os

import numpy as np

from bip import steps
from fvn import FN, P
from ratmodel import cross_r


def _load_kernels():
    """compile fast_kernels.c for this prime (cached); None if no compiler."""
    import ctypes, hashlib, os, subprocess, tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    srcf = os.path.join(here, "fast_kernels.c")
    try:
        code = open(srcf, "rb").read()
        tag = hashlib.sha1(code + str(P).encode()).hexdigest()[:12]
        d = os.path.join(tempfile.gettempdir(), "fast_kernels_cache")
        os.makedirs(d, exist_ok=True)
        so = os.path.join(d, "fk_%s.so" % tag)
        if not os.path.exists(so):
            tmp = so + ".%d" % os.getpid()
            subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                                   "-DPMOD=%dULL" % P, srcf, "-o", tmp])
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
    except Exception:
        return None
    p = ctypes.c_void_p
    i = ctypes.c_int64
    lib.gather_mul_acc.argtypes = [p, i, i, p, p, i, i, p, p, p]
    lib.small_matmul.argtypes = [p, p, i, i, i, p]
    lib.mix_cols.argtypes = [p, p, i, i, i, i, p]
    lib.modpow.argtypes = [p, i, ctypes.c_uint64, p]
    return lib


_K = None if os.environ.get("FAST_NO_C") else _load_kernels()


def _ptr(a):
    return a.ctypes.data


if _K is not None:
    import fvn as _fvn

    def _c_modpow(a, e):
        a = np.ascontiguousarray(a, dtype=np.int64)
        out = np.empty_like(a)
        _K.modpow(_ptr(a), a.size, int(e), _ptr(out))
        return out

    _fvn._modpow = _c_modpow


def mix_cols(W, G):
    """W (Np, d, K) times the point independent matrix G (d, c) -> (Np, c, K)."""
    Np, d, K = W.shape
    nc = G.shape[1]
    if _K is None:
        out = np.zeros((Np, nc, K), dtype=np.int64)
        for a in range(d):
            out = (out + W[:, a:a + 1, :] * G[a][None, :, None] % P) % P
        return out
    W = np.ascontiguousarray(W, dtype=np.int64)
    G = np.ascontiguousarray(G, dtype=np.int64)
    out = np.empty((Np, nc, K), dtype=np.int64)
    _K.mix_cols(_ptr(W), _ptr(G), Np, d, nc, K, _ptr(out))
    return out


def gather_mul_acc(V, src, starts, udst, C, out):
    """out[udst[g]] = sum_{e in group g} V[src[e]] * C[e] (broadcast over the
    middle axes of V), groups = runs of equal destinations starting at starts."""
    if not len(src):
        return out
    if _K is None:
        Cb = C.reshape((len(src),) + (1,) * (V.ndim - 2) + (C.shape[-1],))
        out[udst] = np.add.reduceat(V[src] * Cb % P, starts, axis=0) % P
        return out
    V = np.ascontiguousarray(V, dtype=np.int64)
    C = np.ascontiguousarray(C, dtype=np.int64)
    src = np.ascontiguousarray(src, dtype=np.int64)
    starts = np.ascontiguousarray(starts, dtype=np.int64)
    udst = np.ascontiguousarray(udst, dtype=np.int64)
    assert out.flags.c_contiguous and out.dtype == np.int64
    K = V.shape[-1]
    B = V[0].size // K if V.shape[0] else 0
    _K.gather_mul_acc(_ptr(V), B, K, _ptr(src), _ptr(starts), len(starts), len(src),
                      _ptr(udst), _ptr(C), _ptr(out))
    return out


def small_matmul(Mx, X):
    """out[a] = sum_b Mx[a, b] * X[b] per point; Mx (m, m, K), X (m, ..., K)."""
    m = Mx.shape[0]
    K = Mx.shape[2]
    if _K is None:
        out = np.zeros(X.shape, dtype=np.int64)
        sh = (1,) * (X.ndim - 2) + (K,)
        for a in range(m):
            acc = np.zeros(X.shape[1:], dtype=np.int64)
            for b in range(m):
                acc = (acc + Mx[a, b].reshape(sh) * X[b] % P) % P
            out[a] = acc
        return out
    Mx = np.ascontiguousarray(Mx, dtype=np.int64)
    X = np.ascontiguousarray(X, dtype=np.int64)
    out = np.empty(X.shape, dtype=np.int64)
    B = X[0].size // K if m else 0
    _K.small_matmul(_ptr(Mx), _ptr(X), m, B, K, _ptr(out))
    return out


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


class _BasisStore:
    """fused bases: small part (pivot paths, inverse pivot matrix) kept forever,
    big part (paths, index, vectors as uint32) in an LRU cache with a byte
    budget; an evicted basis is recomputed on demand (deterministically)."""

    def __init__(self, budget):
        from collections import OrderedDict
        self.small, self.big, self.budget, self.bytes = {}, OrderedDict(), budget, 0

    def __contains__(self, key):
        return key in self.small and key in self.big

    def has_small(self, key):
        return key in self.small

    def __setitem__(self, key, val):
        paths, idx, Vk, piv, minv = val
        self.small[key] = (Vk.shape[1], [paths[r] for r in piv], minv, Vk.shape[0])
        if key in self.big:
            self.bytes -= self.big.pop(key)[3]
        V32 = Vk.astype(np.uint32)
        nb = V32.nbytes + 200 * len(paths)
        self.big[key] = (paths, idx, V32, nb)
        self.bytes += nb
        while self.bytes > self.budget and len(self.big) > 1:
            _, v = self.big.popitem(last=False)
            self.bytes -= v[3]

    def __getitem__(self, key):
        paths, idx, V32, _ = self.big[key]
        self.big.move_to_end(key)
        m, ppaths, minv, N = self.small[key]
        piv = [idx[p] for p in ppaths] if idx else []
        return paths, idx, V32.astype(np.int64), piv, minv


def _cells(R):
    return [(r, c) for r, length in enumerate(R) for c in range(length)]


class FastCtx:
    def __init__(self, M, R):
        self.M, self.R, self.n = M, tuple(R), sum(R)
        self.K = len(M.q.v)
        self.zero = M.one * 0
        self.fb = {}          # (start, end, kind) -> (vecs dicts | None, piv, matinv (m,m,K), m)
        from collections import OrderedDict as _OD
        self.fc = _OD()       # fcross results, LRU by bytes (recomputation is deterministic)
        self._fcbytes = 0
        self._fcbudget = int(float(__import__("os").environ.get("FAST_FC_GB", "1.0")) * 2 ** 30)
        from collections import OrderedDict
        import os as _os
        self._umemo, self._ubytes = OrderedDict(), 0
        self._ubudget = int(float(_os.environ.get("FAST_UMEMO_GB", "1.0")) * 2 ** 30)
        self._rng = np.random.default_rng(12345)
        self._paths_cache = {}
        self._vec_cache = _BasisStore(budget=int(float(__import__("os").environ.get("FAST_CACHE_GB", "1.5")) * 2 ** 30))
        self._gen = {}        # (paths key) -> generator tables
        self._xc = {}         # local cross_r results -> ((mp, coef row id), ...)
        self._carr, self._cn = np.zeros((1024, self.K), dtype=np.int64), 0
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
        ck = (start, end, kind, n)
        if ck in self._paths_cache:
            return self._paths_cache[ck]
        res = self._paths_uncached(start, end, kind, n)
        if len(self._paths_cache) > 20000:
            self._paths_cache.clear()
        self._paths_cache[ck] = res
        return res

    def _paths_uncached(self, start, end, kind, n):
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
    def _cross(self, a, b, c, k1, k2, inv):
        """cross_r memoised; coefficients as row ids of the table _coef."""
        key = (a, b, c, k1, k2, inv)
        r = self._xc.get(key)
        if r is None:
            r = []
            for mp, x in cross_r(self.M, a, b, c, k1, k2, inv).items():
                if self._cn == len(self._carr):
                    self._carr = np.concatenate([self._carr, np.zeros_like(self._carr)])
                self._carr[self._cn] = _fv(x, self.K)
                r.append((mp, self._cn))
                self._cn += 1
            r = self._xc[key] = tuple(r)
        return r

    def _coef(self, ids):
        return self._carr[np.asarray(ids, dtype=np.int64)] if len(ids) else np.zeros((0, self.K), dtype=np.int64)

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
            for mp, x in self._cross(p[i], p[i + 1], p[i + 2], k1, k2, inv):
                q_ = p[:i + 1] + (mp,) + p[i + 2:]
                j = nidx.get(q_)
                if j is None:
                    if new_paths is not None:
                        continue      # outside the target space (zero by construction)
                    j = nidx[q_] = len(npaths)
                    npaths.append(q_)
                src.append(r)
                dst.append(j)
                coef.append(x)
        out = np.zeros((len(npaths),) + V.shape[1:], dtype=np.int64)
        if src:
            src = np.array(src)
            dst = np.array(dst)
            C = self._coef(coef)                               # (nnz, K)
            order = np.argsort(dst, kind="stable")
            src, dst, C = src[order], dst[order], C[order]
            starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
            gather_mul_acc(V, src, starts, dst[starts], C, out)
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
                for mp, x in self._cross(p[i], p[i + 1], p[i + 2], kind, kind, False):
                    q_ = p[:i + 1] + (mp,) + p[i + 2:]
                    j = idx.get(q_)
                    if j is None:
                        continue
                    src.append(r)
                    dst.append(j)
                    coef.append(x)
            src = np.array(src, dtype=np.int64)
            dst = np.array(dst, dtype=np.int64)
            C = self._coef(coef)
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
        return gather_mul_acc(V, src, starts, udst, C, out)

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
    def _shape_j(self, j):
        cells = _cells(self.R)[:j]
        shape = []
        for r, c in cells:
            if r == len(shape):
                shape.append(1)
            else:
                shape[r] += 1
        return tuple(shape)

    def _U(self, start, kind, j, s):
        """basis (paths, idx, V (N, d, K)) of e_{T_j}(paths of length j from start
        to s), memoised (LRU by bytes); built from the U(j-1, predecessors)."""
        key = (start, kind, j, s)
        hit = self._umemo.get(key)
        if hit is not None:
            self._umemo.move_to_end(key)
            return hit
        K, q = self.K, self.M.q
        if j == 0:
            res = ([(start,)], {(start,): 0}, np.ones((1, 1, K), dtype=np.int64))
            self._umemo_put(key, res)
            return res
        paths = self._paths(start, s, kind, n=j)
        idx = {p: k for k, p in enumerate(paths)}
        preds = sorted({p[-2] for p in paths})
        parts = []
        for s0 in preds:
            pl, _, V = self._U(start, kind, j - 1, s0)
            if V.shape[1]:
                parts.append((pl, V))
        d = sum(V.shape[1] for _, V in parts)
        want = self._expected(start, s, kind, self._shape_j(j))
        if d == 0 or want == 0:
            res = (paths, idx, np.zeros((len(paths), 0, K), dtype=np.int64))
            self._umemo_put(key, res)
            return res
        W = np.zeros((len(paths), d, K), dtype=np.int64)
        col = 0
        for pl, V in parts:
            rows = [idx[p + (s,)] for p in pl]
            W[rows, col:col + V.shape[1]] = V
            col += V.shape[1]
        if d > want + 2:                       # project only random combinations
            # seeded by the key: a recomputation after LRU eviction must give
            # the same basis as the one earlier results were expressed in
            import hashlib
            seed = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little")
            G = np.random.default_rng(seed).integers(1, P, size=(d, want + 2)).astype(np.int64)
            W = mix_cols(W, G)
        r, c = _cells(self.R)[j - 1]
        t = c - r
        shape = list(self._shape_j(j - 1))
        others = []
        if j >= 2:
            for rr in range(len(shape) + 1):
                cur = shape[rr] if rr < len(shape) else 0
                if rr == 0 or shape[rr - 1] > cur:
                    cc = cur - rr
                    if cc != t:
                        others.append(cc)
        if others:
            tabs = self._prefix_gens(paths, idx, kind, j)
            for cc in others:
                Lv = W
                for i in list(range(j - 2, -1, -1)) + list(range(0, j - 1)):
                    Lv = self._gen_apply(tabs[i], Lv, len(paths))
                mc = (-(q ** (2 * cc))).v
                fac = (1 / (q ** (2 * t) - q ** (2 * cc))).v
                W = (Lv + W * mc % P) % P * fac % P
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
            if len(keep) == want:
                break
        assert len(keep) == want, (start, s, j, len(keep), want)
        res = (paths, idx, W[:, keep, :])
        self._umemo_put(key, res)
        return res

    def _umemo_put(self, key, res):
        nb = res[2].nbytes + 200 * len(res[0])
        self._umemo[key] = res
        self._ubytes += nb
        while self._ubytes > self._ubudget and len(self._umemo) > 1:
            _, v = self._umemo.popitem(last=False)
            self._ubytes -= v[2].nbytes + 200 * len(v[0])

    def _all_bases(self, start, kind, only=None):
        n, K = self.n, self.K
        e = only
        key = (start, e, kind)
        paths, idx, Vk = self._U(start, kind, n, e)
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
                for mp, x in self._cross(p[i], p[i + 1], p[i + 2], kind, kind, False):
                    q_ = p[:i + 1] + (mp,) + p[i + 2:]
                    jj = idx.get(q_)
                    if jj is None:
                        continue
                    src.append(r)
                    dst.append(jj)
                    coef.append(x)
            src = np.array(src, dtype=np.int64)
            dst = np.array(dst, dtype=np.int64)
            C = self._coef(coef)
            order = np.argsort(dst, kind="stable")
            src, dst, C = src[order], dst[order], C[order]
            starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]]) if len(dst) else np.zeros(0, np.int64)
            tabs.append((src, dst, C, starts, dst[starts] if len(dst) else dst))
        return tabs

    def _expected(self, start, end, kind, shape=None):
        from racah_rat import expected_mult
        shape = self.R if shape is None else tuple(shape)
        key = ("mult", start, end, kind, shape)
        if key not in self.fb:
            self.fb[key] = expected_mult(start, end, shape, kind) if shape else int(start == end)
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
        key = (start, end, kind)
        if not self._vec_cache.has_small(key):
            self._basis_arrays(start, end, kind)
        m, ppaths, minv, _ = self._vec_cache.small[key]
        B = W.shape[1]
        if m == 0:
            assert not W.any(), "vector outside fused space"
            return np.zeros((0, B, self.K), dtype=np.int64)
        pos = {p: r for r, p in enumerate(pathlist)}
        rhs = np.zeros((m, B, self.K), dtype=np.int64)
        for k, pp in enumerate(ppaths):
            j = pos.get(pp)
            if j is not None:
                rhs[k] = W[j]
        return small_matmul(minv, rhs)        # coords = minv @ rhs per point

    def coords(self, start, end, kind, w):
        key = (start, end, kind)
        if not self._vec_cache.has_small(key):
            self._basis_arrays(start, end, kind)
        m = self._vec_cache.small[key][0]
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
        labs = [[] for _ in range(m)]
        blks = [[] for _ in range(m)]
        for mp, rows in groups.items():
            sub = V[rows]                                  # (len, m, K)
            plist = [cur[r][1:] if side == "L" else cur[r][:n + 1] for r in rows]
            if side == "L":
                cs = self._coords_arr(mp, nu, K, sub, plist)
            else:
                cs = self._coords_arr(lam, mp, K, sub, plist)
            nz = cs.any(axis=2)                                # (m', m)
            for k in range(m):
                kks = np.flatnonzero(nz[:, k])
                if len(kks):
                    labs[k].extend((mp, int(kk)) for kk in kks)
                    blks[k].append(cs[kks, k])
        z = np.zeros((0, self.K), dtype=np.int64)
        return [(tuple(labs[k]), np.concatenate(blks[k]) if blks[k] else z) for k in range(m)]

    def fcross_arr(self, side, lam, mu, nu, K, k, inv=False):
        """((mu', k'), ...), coefficient rows (len, K) of moving a V step past
        basis vector k of a fused block (side L: block first)."""
        key = (side, lam, mu, nu, K, inv)
        res = self.fc.get(key)
        if res is None:
            self.stats["fcross"] += 1
            res = self.fc[key] = self._fcross(side, lam, mu, nu, K, inv)
            self._fcbytes += sum(C.nbytes + 100 * len(l) for l, C in res)
            while self._fcbytes > self._fcbudget and len(self.fc) > 1:
                _, old = self.fc.popitem(last=False)
                self._fcbytes -= sum(C.nbytes + 100 * len(l) for l, C in old)
        else:
            self.fc.move_to_end(key)
        return res[k]

    def fcross_left(self, lam, mu, nu, K, k, inv=False):
        labs, C = self.fcross_arr("L", lam, mu, nu, K, k, inv)
        return {l: FN(C[r].copy()) for r, l in enumerate(labs)}

    def fcross_right(self, lam, mu, nu, K, k, inv=False):
        labs, C = self.fcross_arr("R", lam, mu, nu, K, k, inv)
        return {l: FN(C[r].copy()) for r, l in enumerate(labs)}


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


def fapply_arr(ctx, keys, A, kinds, i):
    """fapply on arrays: keys [(states, labels)], A (len(keys), C, K)."""
    k1, k2 = kinds[i], kinds[i + 1]
    K = A.shape[-1]
    src, dst, coef = [], [], []
    nidx, nkeys = {}, []
    for r, (st, lb) in enumerate(keys):
        lam, mu, nu = st[i], st[i + 1], st[i + 2]
        if k1 != 'V' and k2 == 'V':
            labs, Cr = ctx.fcross_arr("L", lam, mu, nu, k1[1], lb[i])
            left = True
        elif k1 == 'V' and k2 != 'V':
            labs, Cr = ctx.fcross_arr("R", lam, mu, nu, k2[1], lb[i + 1])
            left = False
        else:
            raise ValueError(kinds)
        if not labs:
            continue
        pre_s, post_s = st[:i + 1], st[i + 2:]
        pre_l, post_l = lb[:i], lb[i + 2:]
        for mp, kk in labs:
            key = (pre_s + (mp,) + post_s, pre_l + ((None, kk) if left else (kk, None)) + post_l)
            j = nidx.get(key)
            if j is None:
                j = nidx[key] = len(nkeys)
                nkeys.append(key)
            dst.append(j)
        src.extend([r] * len(labs))
        coef.append(Cr)
    out = np.zeros((len(nkeys),) + A.shape[1:], dtype=np.int64)
    if src:
        src = np.array(src)
        dst = np.array(dst)
        C = np.concatenate(coef)
        order = np.argsort(dst, kind="stable")
        src, dst, C = src[order], dst[order], C[order]
        starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
        gather_mul_acc(A, src, starts, dst[starts], C, out)
    keep = np.flatnonzero(out.any(axis=(1, 2)))
    nk = kinds[:i] + (k2, k1) + kinds[i + 2:]
    return [nkeys[r] for r in keep], out[keep], nk


def fapply_plan(ctx, keys, kinds, i):
    """the structure of fapply_arr without values: (new keys, (src, starts, udst, C uint32), new kinds);
    apply it to any A over `keys` with apply_plan (no pruning of zero rows)."""
    k1, k2 = kinds[i], kinds[i + 1]
    src, dst, coef = [], [], []
    nidx, nkeys = {}, []
    for r, (st, lb) in enumerate(keys):
        lam, mu, nu = st[i], st[i + 1], st[i + 2]
        if k1 != 'V' and k2 == 'V':
            labs, Cr = ctx.fcross_arr("L", lam, mu, nu, k1[1], lb[i])
            left = True
        elif k1 == 'V' and k2 != 'V':
            labs, Cr = ctx.fcross_arr("R", lam, mu, nu, k2[1], lb[i + 1])
            left = False
        else:
            raise ValueError(kinds)
        if not labs:
            continue
        pre_s, post_s = st[:i + 1], st[i + 2:]
        pre_l, post_l = lb[:i], lb[i + 2:]
        for mp, kk in labs:
            key = (pre_s + (mp,) + post_s, pre_l + ((None, kk) if left else (kk, None)) + post_l)
            j = nidx.get(key)
            if j is None:
                j = nidx[key] = len(nkeys)
                nkeys.append(key)
            dst.append(j)
        src.extend([r] * len(labs))
        coef.append(Cr)
    nk = kinds[:i] + (k2, k1) + kinds[i + 2:]
    if not src:
        return nkeys, None, nk
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    C = np.concatenate(coef)
    order = np.argsort(dst, kind="stable")
    src, dst, C = src[order], dst[order], C[order]
    starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
    return nkeys, (src, starts, dst[starts], C.astype(np.uint32)), nk


def apply_plan(plan, A, nnew):
    out = np.zeros((nnew,) + A.shape[1:], dtype=np.int64)
    if plan is not None:
        src, starts, udst, C = plan
        gather_mul_acc(A, src, starts, udst, C, out)
    return out


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
    # at most FAST_COL_ELEMS / K columns at once: the moved arrays are (keys, columns, K)
    import os
    width = max(1, int(os.environ.get("FAST_COL_ELEMS", "480")) // K)
    batches = [(X, cols[a:a + width]) for X, cols in groups.items() for a in range(0, len(cols), width)]
    for gi, (X, cols) in enumerate(batches):
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
        keys = list(w)
        A = np.stack([w[k] for k in keys])
        del w
        for t in range(n):
            keys, A, kinds = fapply_arr(ctx, keys, A, kinds, t + 1)
            keys, A, kinds = fapply_arr(ctx, keys, A, kinds, t)
        assert kinds == ('V',) * n + ('FV', 'FW')
        idxs = [c for c, _, _ in cols]
        for (st, lb), arr in zip(keys, A):
            assert st[:n + 1] == TRp and st[n + 2] == RR
            r = qidx[(st[n + 1], lb[n], lb[n + 1])]
            S[r, idxs] = (S[r, idxs] + arr) % P
        if log:
            log("X %d/%d %s: %d columns, bases %d fcross %d" % (gi + 1, len(batches), X, Cg,
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
