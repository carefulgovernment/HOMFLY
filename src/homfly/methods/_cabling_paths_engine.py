# Vendored verbatim from the "colored-homfly-cabling" skill (homfly_cabling.py):
# cabling restricted to the multiplicity spaces W_Q in a path basis.
# Wrapped by methods/cabling_paths.py; do not edit here.
#!/usr/bin/env python3
"""
Colored HOMFLY-PT polynomials H_R(K; A, q) of knots via the cabling procedure.

Method (exact, no Racah matrices needed)
----------------------------------------
* K = closure of a braid beta in B_m.  Colour R, |R| = r.
* The r-cable of beta lives in the Hecke algebra H_{mr}; the colour is imposed by
  inserting the primitive idempotent e_T (T = SYT of shape R) on every group of
  r strands.  By Schur-Weyl duality
        Hbar_R(K) = theta_R^{-w} * sum_{Q |- mr} S*_Q(A,q) * Tr_{W_Q} (beta_cable)
  where W_Q = e_T^{(1)}...e_T^{(m)} rho_Q is the multiplicity space of Q in R^{(x)m}
  (dim W_Q = LR multiplicity) and rho_Q is the Hecke irrep (Young seminormal form).
* Key optimisation: the seminormal basis of rho_Q restricted to W_Q factorises along
  chains  0 = l_0 < l_1 = R < l_2 < ... < l_m = Q  (r boxes per step).  Each projector
  e^{(j)} lives in the r-box skew representation of l_j/l_{j-1}; each band crossing
  b_j lives in the 2r-box skew representation of l_{j+1}/l_{j-1}.  So every heavy
  object is a *small* skew-shape Hecke representation; the big rho_Q (dimension up to
  millions) is never built.  W_Q gets a path basis (chain + multiplicity labels) and
  the band crossings become block-sparse w_Q x w_Q matrices.
* Arithmetic: everything is done modulo a 31-bit prime at a numeric value q0.
  The A-dependence is kept exactly (S*_Q are polynomials in A^2 for fixed q0),
  the q-dependence is recovered by sparse-aware Laurent interpolation
  (Newton interpolation + scan for the z^{-L} denominator), using:
    - parity detection (polynomial in q^2 up to a monomial)   -> halves #points
    - the symmetry H_R(A,q) = H_{R^T}(A,-1/q) for self-conjugate R -> halves again
  Integers are recovered by symmetric lift; a second prime verifies (CRT if needed).

Conventions:  Hecke generator (g-q)(g+q^{-1}) = 0,  A = q^N,
  S*_Q = prod_{boxes} (A q^{c} - A^{-1} q^{-c}) / (q^{h} - q^{-h}),
  H_R normalised by the unknot:  H_R(unknot) = 1,  topological framing.
  Fundamental H_[1] = Knot Atlas HOMFLYPT[a, z] with a = A, z = q - 1/q (checked on 9_34).
"""
import argparse
import json
import random
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# primes & modular helpers
# ----------------------------------------------------------------------------

def _is_prime(n):
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_below(N, count):
    out, n = [], N - 1
    while len(out) < count:
        if _is_prime(n):
            out.append(n)
        n -= 1
    return out


# primes < 2^21: matrix products are then single exact float64 BLAS calls
# (k p^2 < 2^53 for inner dimensions k < 2048); more primes are combined by
# CRT when the coefficients need them
PRIMES = primes_below(2 ** 21, 24)


def powmod_arr(a, e, p):
    a = np.asarray(a, dtype=np.int64) % p
    res = np.ones_like(a)
    while e:
        if e & 1:
            res = res * a % p
        a = a * a % p
        e >>= 1
    return res


def inv_arr(a, p):
    return powmod_arr(a, p - 2, p)


_CH = 11                        # 11-bit chunks -> exact float64 products
_KMAX = 1 << 11                 # inner-dimension block for exactness


def matmul_mod(A, B, p):
    """(A @ B) mod p for int64 matrices with entries in [0,p), p < 2^31, exact via float64 BLAS."""
    A = np.ascontiguousarray(A, dtype=np.int64)
    B = np.ascontiguousarray(B, dtype=np.int64)
    n, k = A.shape
    m = B.shape[1]
    if k * (p - 1) ** 2 < 2 ** 53:      # one exact float64 product (primes < 2^21)
        return np.fmod(A.astype(np.float64) @ B.astype(np.float64), p).astype(np.int64)
    out = np.zeros((n, m), dtype=np.int64)
    mask = (1 << _CH) - 1
    for s in range(0, k, _KMAX):
        As = A[:, s:s + _KMAX]
        Bs = B[s:s + _KMAX].astype(np.float64)
        acc = np.zeros((n, m), dtype=np.int64)
        for c in range(3):
            Ac = ((As >> (_CH * c)) & mask).astype(np.float64)
            if not Ac.any():
                continue
            R = np.fmod(Ac @ Bs, p).astype(np.int64)
            acc = (acc + R * ((1 << (_CH * c)) % p)) % p
        out = (out + acc) % p
    return out


class Singular(Exception):
    pass


def inv_mod(M, p):
    n = M.shape[0]
    A = np.concatenate([np.asarray(M, dtype=np.int64) % p, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        nz = np.nonzero(A[c:, c])[0]
        if len(nz) == 0:
            raise Singular()
        r = c + nz[0]
        if r != c:
            A[[c, r]] = A[[r, c]]
        A[c] = A[c] * pow(int(A[c, c]), p - 2, p) % p
        f = A[:, c].copy()
        f[c] = 0
        nzr = np.nonzero(f)[0]
        if len(nzr):
            A[nzr] = (A[nzr] - (f[nzr, None] * A[c][None, :]) % p) % p
    return A[:, n:]


def rref_pivots(M, p):
    """pivot columns of the row-reduced form of M (mod p)"""
    A = np.asarray(M, dtype=np.int64).copy() % p
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        k = r + nz[0]
        if k != r:
            A[[r, k]] = A[[k, r]]
        A[r] = A[r] * pow(int(A[r, c]), p - 2, p) % p
        f = A[:, c].copy()
        f[r] = 0
        nzr = np.nonzero(f)[0]
        if len(nzr):
            A[nzr] = (A[nzr] - (f[nzr, None] * A[r][None, :]) % p) % p
        piv.append(c)
        r += 1
    return piv

# ----------------------------------------------------------------------------
# partitions, skew standard tableaux, LR coefficients
# ----------------------------------------------------------------------------

def conj(la):
    return tuple(sum(1 for x in la if x > j) for j in range(la[0])) if la else ()


def contents(la):
    return [j - i for i, row in enumerate(la) for j in range(row)]


def kappa(la):
    return sum(contents(la))


def hooks(la):
    lc = conj(la)
    return [la[i] - j + lc[j] - i - 1 for i, row in enumerate(la) for j in range(row)]


def _addable_rows(shape, outer=None):
    res = []
    for i in range(len(shape) + 1):
        L = shape[i] if i < len(shape) else 0
        if i > 0 and shape[i - 1] <= L:
            continue
        if outer is not None and (i >= len(outer) or outer[i] <= L):
            continue
        res.append(i)
    return res


def _add(shape, i):
    s = list(shape)
    if i == len(s):
        s.append(1)
    else:
        s[i] += 1
    return tuple(s)


def skew_tableaux(lam, mu):
    """all standard fillings of mu/lam as tuples of boxes (row, col) in order of entries"""
    k = sum(mu) - sum(lam)
    out = []

    def rec(shape, boxes):
        if len(boxes) == k:
            out.append(tuple(boxes))
            return
        for i in _addable_rows(shape, mu):
            L = shape[i] if i < len(shape) else 0
            boxes.append((i, L))
            rec(_add(shape, i), boxes)
            boxes.pop()
    rec(tuple(lam), [])
    return out


def shapes_adding(lam, k):
    """all partitions obtained from lam by adding k boxes"""
    cur = {tuple(lam)}
    for _ in range(k):
        cur = {_add(s, i) for s in cur for i in _addable_rows(s)}
    return sorted(cur, reverse=True)


def contains(mu, lam):
    return len(lam) <= len(mu) and all(lam[i] <= mu[i] for i in range(len(lam)))


def lr_coeff(lam, R, mu):
    """Littlewood-Richardson coefficient c^{mu}_{lam,R}"""
    if sum(mu) != sum(lam) + sum(R) or not contains(mu, lam):
        return 0
    boxes = []   # reading order: rows top->bottom, right->left
    for i in range(len(mu)):
        lo = lam[i] if i < len(lam) else 0
        for j in range(mu[i] - 1, lo - 1, -1):
            boxes.append((i, j))
    fill = {}
    cnt = [0] * (len(R) + 1)
    total = [0]

    def rec(t):
        if t == len(boxes):
            total[0] += 1
            return
        i, j = boxes[t]
        for v in range(1, len(R) + 1):
            if cnt[v] >= R[v - 1]:
                continue
            if v > 1 and cnt[v] + 1 > cnt[v - 1]:
                continue                        # lattice word
            if (i, j + 1) in fill and fill[(i, j + 1)] < v:
                continue                        # rows weakly increasing
            if (i - 1, j) in fill and fill[(i - 1, j)] >= v:
                continue                        # columns strictly increasing
            fill[(i, j)] = v
            cnt[v] += 1
            rec(t + 1)
            cnt[v] -= 1
            del fill[(i, j)]
    rec(0)
    return total[0]

# ----------------------------------------------------------------------------
# skew seminormal representation of the Hecke algebra
# ----------------------------------------------------------------------------

class SkewRep:
    """Young seminormal representation of H_k on standard tableaux of a skew shape."""

    def __init__(self, tabs):
        self.tabs = tabs
        self.f = len(tabs)
        k = len(tabs[0]) if tabs else 0
        self.k = k
        index = {t: n for n, t in enumerate(tabs)}
        self.d, self.partner = {}, {}
        for i in range(1, k):
            d = np.zeros(self.f, dtype=np.int64)
            pa = np.arange(self.f, dtype=np.int64)
            for n, t in enumerate(tabs):
                (r1, c1), (r2, c2) = t[i - 1], t[i]
                dd = (c2 - r2) - (c1 - r1)
                d[n] = dd
                if abs(dd) >= 2:
                    s = list(t)
                    s[i - 1], s[i] = s[i], s[i - 1]
                    pa[n] = index[tuple(s)]
            self.d[i], self.partner[i] = d, pa
        self.tabs = None          # free the tableaux: only (d, partner) tables are needed later

    def numeric(self, num):
        """per-generator (diag, off, partner) arrays for the numeric context num"""
        ops = {}
        for i in range(1, self.k):
            d = self.d[i]
            diag = num.a_of[d + num.off]
            off = num.b_of[d + num.off]
            ops[i] = (diag, off, self.partner[i])
        return ops


def apply_word(ops, word, X, num):
    """rho(word) @ X ; word = list of signed local generator indices (left to right)"""
    p = num.p
    X = X.copy()
    tmp = np.empty_like(X)
    for g in reversed(word):
        diag, off, pa = ops[abs(g)]
        if g < 0:
            diag = (diag - num.qmqi) % p
        np.take(X, pa, axis=0, out=tmp)
        tmp *= off[:, None]
        X *= diag[:, None]
        X += tmp
        np.remainder(X, p, out=X)
    return X


def batched_inv_mod(Ms, p):
    """inverse of a stack (G, d, d) of matrices mod p (Gauss-Jordan without pivoting, fallback per matrix)"""
    G, d, _ = Ms.shape
    A = np.concatenate([Ms % p, np.broadcast_to(np.eye(d, dtype=np.int64), (G, d, d))], axis=2).copy()
    for c in range(d):
        piv = A[:, c, c]
        if np.any(piv == 0):
            return np.stack([inv_mod(M, p) for M in Ms])
        A[:, c] = A[:, c] * inv_arr(piv, p)[:, None] % p
        f = A[:, :, c].copy()
        f[:, c] = 0
        A = (A - (f[:, :, None] * A[:, c][:, None, :]) % p) % p
    return A[:, :, d:]


class Numeric:
    """numeric data at q = q0 mod p"""

    def __init__(self, q0, p, dmax):
        self.p, self.q = p, q0 % p
        qi = pow(q0, p - 2, p)
        self.qmqi = (q0 - qi) % p
        self.off = dmax + 1
        size = 2 * dmax + 3
        self.a_of = np.zeros(size, dtype=np.int64)
        self.b_of = np.zeros(size, dtype=np.int64)
        inv_qmqi = pow(self.qmqi, p - 2, p)

        def qn(d):
            return (pow(q0, d % (p - 1), p) - pow(q0, (-d) % (p - 1), p)) * inv_qmqi % p
        for d in range(-dmax - 1, dmax + 2):
            if d == 0:
                continue
            qd = qn(d)
            a = pow(q0, d % (p - 1), p) * pow(qd, p - 2, p) % p
            self.a_of[d + self.off] = a
            if abs(d) >= 2:
                if d < 0:
                    self.b_of[d + self.off] = 1
                else:
                    self.b_of[d + self.off] = qn(d - 1) * qn(d + 1) % p * pow(qd * qd % p, p - 2, p) % p

    def qpow(self, e):
        return pow(self.q, e % (self.p - 1), self.p)

# ----------------------------------------------------------------------------
# the engine
# ----------------------------------------------------------------------------

def _make_buckets(spaces, k, maxcells=4_000_000):
    """spaces: key -> (SkewRep, width).  Stack spaces with similar width; generators act row-locally."""
    # sort by width; greedily chunk while padding waste stays <= 10% and size <= maxcells
    order = sorted(spaces, key=lambda kk: spaces[kk][1])
    chunks, cur, exact, f = [], [], 0, 0
    for key in order:
        rep, wd = spaces[key]
        if cur and ((f + rep.f) * wd > 1.1 * (exact + rep.f * wd) or (f + rep.f) * wd > maxcells):
            chunks.append(cur)
            cur, exact, f = [], 0, 0
        cur.append(key)
        exact += rep.f * wd
        f += rep.f
    if cur:
        chunks.append(cur)
    out = []
    for chunk in chunks:
        roffs, acc = [], 0
        for kk in chunk:
            roffs.append(acc)
            acc += spaces[kk][0].f
        dd, pp = {}, {}
        for i in range(1, k):
            dd[i] = np.concatenate([spaces[kk][0].d[i] for kk in chunk])
            pp[i] = np.concatenate([spaces[kk][0].partner[i] + ro for kk, ro in zip(chunk, roffs)])
        wdt = max(spaces[kk][1] for kk in chunk)
        out.append(dict(keys=chunk, roffs=roffs, f=acc, width=wdt, d=dd, partner=pp))
    return out


def band_word(r):
    w = []
    for i in range(r):
        w += list(range(r - i, 2 * r - i))
    return w


class CablingEngine:
    def __init__(self, braid, m, R, verbose=False):
        self.braid = [int(x) for x in braid]
        self.m, self.R = m, tuple(R)
        self.r = r = sum(R)
        self.n = n = m * r
        self.verbose = verbose
        assert all(1 <= abs(g) < m for g in self.braid), "braid letters must be in 1..m-1"
        # permutation / components and writhe
        perm = list(range(m))
        for g in self.braid:
            i = abs(g) - 1
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
        self.writhe = sum(1 if g > 0 else -1 for g in self.braid)
        comp, seen = 0, [False] * m
        for s in range(m):
            if not seen[s]:
                comp += 1
                x = s
                while not seen[x]:
                    seen[x] = True
                    x = perm[x]
        self.components = comp
        if comp != 1:
            raise ValueError("braid closure is a link with %d components; this code handles knots "
                             "(framing correction uses the total writhe)" % comp)
        # the fixed SYT T of shape R (row reading); local Jucys-Murphy data
        Tboxes = [(i, j) for i, row in enumerate(R) for j in range(row)]
        self.T_factors = []    # (k, [other contents], content_T(k))
        shape = ()
        for k, (i, j) in enumerate(Tboxes, start=1):
            if k >= 2:
                others = []
                for ii in _addable_rows(shape):
                    L = shape[ii] if ii < len(shape) else 0
                    c = L - ii
                    if c != j - i:
                        others.append(c)
                self.T_factors.append((k, others, j - i))
            shape = _add(shape, i)
        t0 = time.time()
        # levels of the chain
        levels = [[()]]
        self.c = {}           # (lam, mu) -> LR multiplicity (>0 only)
        for lev in range(1, m + 1):
            nxt = set()
            for lam in levels[-1]:
                for mu in shapes_adding(lam, r):
                    cc = lr_coeff(lam, R, mu)
                    if cc:
                        self.c[(lam, mu)] = cc
                        nxt.add(mu)
            levels.append(sorted(nxt, reverse=True))
        self.levels = levels
        self.Qs = levels[m]
        # r-step skew reps (all pairs lam <= kappa at consecutive levels, needed for band spaces)
        self.rstep = {}
        for (lam, mu) in self.c:
            self.rstep[(lam, mu)] = SkewRep(skew_tableaux(lam, mu))
        # 2r-step band spaces for the bands that occur
        used = sorted({abs(g) for g in self.braid})
        self.bands_needed = used
        self.signs_needed = {j: set() for j in used}
        for g in self.braid:
            self.signs_needed[abs(g)].add(1 if g > 0 else -1)
        pairs = set()
        for j in used:
            for lam in levels[j - 1]:
                for nu in levels[j + 1]:
                    ks = [ka for ka in levels[j] if (lam, ka) in self.c and (ka, nu) in self.c]
                    if ks:
                        pairs.add((lam, nu))
        self.bspace = {}
        for (lam, nu) in pairs:
            kappas = [ka for ka in shapes_adding(lam, r) if contains(nu, ka)]
            tabs, rowoff = [], {}
            for ka in kappas:
                t1 = skew_tableaux(lam, ka)
                t2 = skew_tableaux(ka, nu)
                if not t1 or not t2:
                    continue
                rowoff[ka] = (len(tabs), len(t1), len(t2))
                for a in t1:
                    for b in t2:
                        tabs.append(a + b)
            valid = [ka for ka in kappas if (lam, ka) in self.c and (ka, nu) in self.c]
            loc, cnt = {}, 0
            for ka in valid:
                for a in range(self.c[(lam, ka)]):
                    for b in range(self.c[(ka, nu)]):
                        loc[(ka, a, b)] = cnt
                        cnt += 1
            self.bspace[(lam, nu)] = dict(rep=SkewRep(tabs), rowoff=rowoff, valid=valid, loc=loc, dim=cnt)
        # bucket skew spaces by width (power of two) for stacked numpy processing
        self.buckets = _make_buckets({k: (b['rep'], b['dim']) for k, b in self.bspace.items()}, 2 * r)
        self.rbuckets = _make_buckets({k: (rep, rep.f) for k, rep in self.rstep.items()}, r)
        for rep in [b['rep'] for b in self.bspace.values()] + list(self.rstep.values()):
            rep.d = rep.partner = None          # now held (concatenated) by the buckets
        self.bw = band_word(r)
        self.bw_inv = [-g for g in reversed(self.bw)]
        # path spaces for every Q
        self.paths = {}
        for Q in self.Qs:
            plist = []

            def rec(lev, chain):
                if lev == m:
                    if chain[-1] == Q:
                        plist.append(tuple(chain))
                    return
                for mu in levels[lev + 1]:
                    if (chain[-1], mu) in self.c and (lev + 1 < m or mu == Q):
                        rec(lev + 1, chain + [mu])
            rec(0, [()])
            basis = []
            for ch in plist:
                ranges = [range(self.c[(ch[j - 1], ch[j])]) for j in range(1, m + 1)]
                for labs in _product(ranges):
                    basis.append((ch, labs))
            w = len(basis)
            groups = {}
            for j in used:
                gr = {}
                for idx, (ch, labs) in enumerate(basis):
                    key = (ch[:j] + ch[j + 1:], labs[:j - 1] + labs[j + 1:])
                    bl = self.bspace[(ch[j - 1], ch[j + 1])]
                    li = bl['loc'][(ch[j], labs[j - 1], labs[j])]
                    gr.setdefault(key, [(ch[j - 1], ch[j + 1]), []])[1].append((li, idx))
                byb = {}
                for key, (bkey, lst) in gr.items():
                    lst.sort()
                    assert [a for a, _ in lst] == list(range(len(lst)))
                    byb.setdefault(bkey, []).append([b for _, b in lst])
                # block-sparse structure: for each local block, a (G, b) array of column indices
                groups[j] = [(bkey, np.array(v, dtype=np.int64)) for bkey, v in byb.items()]
            self.paths[Q] = dict(w=w, groups=groups)
        self.dmax = 2 * n + 2        # axial distances in skew shapes can reach ~2n
        # A-exponent bookkeeping
        self.K = (m - 1) * r + 1
        if verbose:
            ws = {Q: self.paths[Q]['w'] for Q in self.Qs}
            print(f"# setup {time.time() - t0:.2f}s: {len(self.Qs)} Q's, sum w_Q = {sum(ws.values())}, "
                  f"max w_Q = {max(ws.values())}, band spaces = {len(self.bspace)} "
                  f"(max dim {max(b['rep'].f for b in self.bspace.values())})", file=sys.stderr)

    # -- framing: theta_R = A^{r} q^{t}, fixed by the kink test (see tests)
    def framing_q_exponent(self):
        return 2 * kappa(self.R)      # theta_R = A^{|R|} q^{2 kappa_R}  (checked on kinks)

    def a_exponents(self):
        r, m = self.r, self.m
        return [-self.writhe * r - (m - 1) * r + 2 * k for k in range(self.K)]

    def evaluate(self, q0, p):
        """vector (length K) of A-coefficients of the normalised H_R at q = q0 (mod p)"""
        num = Numeric(q0, p, self.dmax)
        r = self.r
        # projector images for r-steps (spaces stacked in buckets; E_T via local Jucys-Murphy)
        V, PIV = {}, {}
        for bk in self.rbuckets:
            X = np.zeros((bk['f'], bk['width']), dtype=np.int64)
            for key, roff in zip(bk['keys'], bk['roffs']):
                f = self.rstep[key].f
                X[roff + np.arange(f), np.arange(f)] = 1
            ops = {}
            for i in range(1, r):
                d = bk['d'][i]
                ops[i] = (num.a_of[d + num.off], num.b_of[d + num.off], bk['partner'][i])
            for (k, others, ct) in self.T_factors:
                for c in others:
                    LX = self._apply_L(ops, k, X, num)
                    den = (num.qpow(2 * ct) - num.qpow(2 * c)) % p
                    X = (LX - num.qpow(2 * c) * X) % p * pow(den, p - 2, p) % p
            for key, roff in zip(bk['keys'], bk['roffs']):
                f = self.rstep[key].f
                E = X[roff:roff + f, :f]
                cc = self.c[key]
                if cc == f:
                    V[key], PIV[key] = np.eye(f, dtype=np.int64), list(range(f))
                    continue
                cols = rref_pivots(E, p)
                if len(cols) != cc:
                    raise Singular()
                Vm = E[:, cols]
                piv = rref_pivots(Vm.T.copy(), p)
                V[key], PIV[key] = matmul_mod(Vm, inv_mod(Vm[piv], p), p), piv
        # band blocks: all 2r-box skew spaces of similar size are processed together
        # (generators act row-locally, so spaces can be stacked -> few large numpy ops)
        Bblk = {key: {} for key in self.bspace}
        signs = sorted({s for ss in self.signs_needed.values() for s in ss})
        for bk in self.buckets:
            X = np.zeros((bk['f'], bk['width']), dtype=np.int64)
            rowsel = []
            for key, roff in zip(bk['keys'], bk['roffs']):
                bs = self.bspace[key]
                lam, nu = key
                col = 0
                rows = []
                for ka in bs['valid']:
                    off, f1, f2 = bs['rowoff'][ka]
                    K1 = np.kron(V[(lam, ka)], V[(ka, nu)]) % p
                    X[roff + off:roff + off + f1 * f2, col:col + K1.shape[1]] = K1
                    col += K1.shape[1]
                    pa, pb = PIV[(lam, ka)], PIV[(ka, nu)]
                    rows.append((roff + off + np.add.outer(np.array(pa) * f2, np.array(pb))).ravel())
                rowsel.append((key, np.concatenate(rows), bs['dim']))
            ops = {}
            for i in range(1, 2 * r):
                d = bk['d'][i]
                ops[i] = (num.a_of[d + num.off], num.b_of[d + num.off], bk['partner'][i])
            Y = apply_word(ops, self.bw, X, num)
            for key, rows, dim in rowsel:
                Bblk[key][1] = Y[rows, :dim]
        if -1 in signs:                       # b^{-1} on the reduced space = inverse of the reduced block
            bydim = {}
            for key in self.bspace:
                bydim.setdefault(self.bspace[key]['dim'], []).append(key)
            for dim, keys in bydim.items():
                inv = batched_inv_mod(np.stack([Bblk[k][1] for k in keys]), p)
                for k, Mi in zip(keys, inv):
                    Bblk[k][-1] = Mi
        # traces per Q, weighted by S*_Q(x = A^2)
        U = np.zeros(self.n + 1, dtype=np.int64)
        for Q in self.Qs:
            info = self.paths[Q]
            tr = self._trace_word(info, Bblk, p)
            if tr:
                U = (U + tr * self._Spoly(Q, num)) % p
        # divide by S*_R
        SR = self._Spoly(self.R, num)
        quo = _polydiv_exact(U, SR, p)
        t = self.framing_q_exponent()
        fac = num.qpow(-self.writhe * t)
        return quo * fac % p

    def _trace_word(self, info, Bblk, p):
        """Tr_{W_Q} of the band braid: M <- M @ C_g applied block-wise (cost w * sum b^2 per crossing)"""
        w = info['w']
        if w <= 160:                      # small: dense matrices are cheaper than Python-level blocks
            C = {}
            for g in set(self.braid):
                j, s = abs(g), (1 if g > 0 else -1)
                D = np.zeros((w, w), dtype=np.int64)
                for bkey, idx in info['groups'][j]:
                    B = Bblk[bkey][s]
                    for row in idx:
                        D[np.ix_(row, row)] = B
                C[g] = D
            M = C[self.braid[0]]
            for g in self.braid[1:-1]:
                M = matmul_mod(M, C[g], p)
            if len(self.braid) == 1:
                return int(np.trace(M) % p)
            return int(np.sum((M * C[self.braid[-1]].T) % p) % p)
        M = np.eye(w, dtype=np.int64)
        if (p - 1) ** 2 * max(idx.shape[1] for j in info['groups'] for _, idx in info['groups'][j]) < 2 ** 53:
            # all blocks of one size at once: a batched exact float64 product
            prep = {}
            for g in set(self.braid):
                j, s = abs(g), (1 if g > 0 else -1)
                by_b = {}
                for bkey, idx in info['groups'][j]:
                    G, b = idx.shape
                    cols, Bs = by_b.setdefault(b, ([], []))
                    cols.append(idx)
                    Bs.append(np.broadcast_to(Bblk[bkey][s].astype(np.float64), (G, b, b)))
                prep[g] = [(np.concatenate(c), np.concatenate(B)) for c, B in by_b.values()]
            for g in self.braid:
                for cols, Bs in prep[g]:
                    sub = M[:, cols].transpose(1, 0, 2).astype(np.float64)        # (G, w, b)
                    R = np.fmod(np.matmul(sub, Bs), p).astype(np.int64)
                    M[:, cols] = R.transpose(1, 0, 2)
            return int(np.trace(M) % p)
        for g in self.braid:
            j, s = abs(g), (1 if g > 0 else -1)
            for bkey, idx in info['groups'][j]:
                G, b = idx.shape
                sub = M[:, idx.ravel()].reshape(w * G, b)
                M[:, idx.ravel()] = matmul_mod(sub, Bblk[bkey][s], p).reshape(w, G * b)
        return int(np.trace(M) % p)

    def _apply_L(self, ops, k, X, num):
        if k == 1:
            return X
        X = apply_word(ops, [k - 1], X, num)
        X = self._apply_L(ops, k - 1, X, num)
        return apply_word(ops, [k - 1], X, num)

    def _Spoly(self, Q, num):
        """coefficients (in x = A^2) of A^{|Q|} S*_Q(A, q0) mod p, length n+1"""
        p = num.p
        poly = np.zeros(self.n + 1, dtype=np.int64)
        poly[0] = 1
        den = 1
        for c, h in zip(contents(Q), hooks(Q)):
            a1, a0 = num.qpow(c), (-num.qpow(-c)) % p
            new = (poly * a0) % p
            new[1:] = (new[1:] + poly[:-1] * a1) % p
            poly = new
            den = den * ((num.qpow(h) - num.qpow(-h)) % p) % p
        return poly * pow(den, p - 2, p) % p


def _product(ranges):
    if not ranges:
        yield ()
        return
    for a in ranges[0]:
        for rest in _product(ranges[1:]):
            yield (a,) + rest


def _polydiv_exact(U, S, p):
    """U / S for coefficient arrays (low degree first); asserts zero remainder"""
    U = [int(x) % p for x in U]
    S = [int(x) % p for x in S]
    while S and S[-1] == 0:
        S.pop()
    while U and U[-1] == 0:
        U.pop()
    ds = len(S) - 1
    if len(U) - 1 < ds:
        if any(U):
            raise ArithmeticError("non-zero remainder")
        return np.zeros(0, dtype=np.int64)
    inv_lead = pow(S[-1], p - 2, p)
    quo = [0] * (len(U) - ds)
    for i in range(len(U) - 1, ds - 1, -1):
        c = U[i] * inv_lead % p
        quo[i - ds] = c
        if c:
            for j in range(ds + 1):
                U[i - ds + j] = (U[i - ds + j] - c * S[j]) % p
    if any(U[:ds]):
        raise ArithmeticError("non-zero remainder in division by S*_R")
    return np.array(quo, dtype=np.int64)

# ----------------------------------------------------------------------------
# Laurent-polynomial interpolation mod p
# ----------------------------------------------------------------------------

def _interp_monomial(zs, vals, p):
    """monomial coefficients (M x K, low degree first) of the interpolants through (zs, vals)"""
    zs = np.array(zs, dtype=np.int64) % p
    M = len(zs)
    coef = np.array(vals, dtype=np.int64) % p                  # (M, K)
    for j in range(1, M):
        den = inv_arr((zs[j:] - zs[:-j]) % p, p)
        coef[j:] = ((coef[j:] - coef[j - 1:-1]) % p) * den[:, None] % p
    K = coef.shape[1]
    poly = np.zeros((M, K), dtype=np.int64)
    poly[0] = coef[M - 1]
    deg = 0
    for j in range(M - 2, -1, -1):
        new = np.zeros_like(poly)
        new[1:deg + 2] = poly[:deg + 1]
        new[:deg + 1] = (new[:deg + 1] - zs[j] * poly[:deg + 1]) % p
        new[0] = (new[0] + coef[j]) % p
        poly = new % p
        deg += 1
    return poly


def _nodepoly(zs, p):
    P = [1]
    for z in zs:
        Q = [0] * (len(P) + 1)
        for i, c in enumerate(P):
            Q[i + 1] = (Q[i + 1] + c) % p
            Q[i] = (Q[i] - c * z) % p
        P = Q
    return np.array(P, dtype=np.int64)        # monic, length M+1


def laurent_reconstruct(zs, vals, p, margin=3):
    """Find, for every column k, a Laurent polynomial z^{-L} N(z) matching the data with
    deg N <= M-1-margin (the minimal L).  Returns list of (L, N coeffs) or None if not yet."""
    M = len(zs)
    G = _interp_monomial(zs, vals, p)            # (M, K)
    Pi = _nodepoly(zs, p)                         # (M+1,)
    K = G.shape[1]
    res = [None] * K
    cur = G.copy()
    limit = M - 1 - margin
    for L in range(0, limit + 1):
        nzmask = cur != 0
        degs = np.where(nzmask.any(axis=0), M - 1 - np.argmax(nzmask[::-1], axis=0), -1)
        for k in range(K):
            if res[k] is None and degs[k] <= limit - L:
                res[k] = (L, cur[:, k].copy())
        if all(x is not None for x in res):
            return res
        # cur <- z * cur mod Pi
        top = cur[M - 1].copy()
        sh = np.zeros_like(cur)
        sh[1:] = cur[:-1]
        cur = (sh - (Pi[:M, None] * top[None, :]) % p) % p
    return None


def _crt(residues, primes):
    x, Mod = 0, 1
    for r, p in zip(residues, primes):
        t = ((r - x) * pow(Mod, -1, p)) % p
        x += Mod * t
        Mod *= p
    return x - Mod if x > Mod // 2 else x

# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

class ColoredHOMFLY:
    def __init__(self, braid, m, R, verbose=False, seed=12345):
        self.eng = CablingEngine(braid, m, R, verbose=verbose)
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.aexp = self.eng.a_exponents()
        self.nevals = 0
        self.teval = 0.0

    def _rand_q(self, p):
        n = 2 * self.eng.n + 4
        while True:
            q = self.rng.randrange(2, p - 1)
            if all(pow(q, 2 * h, p) != 1 for h in range(1, n + 1)):
                return q

    def _eval(self, q, p):
        t = time.time()
        while True:
            try:
                v = self.eng.evaluate(q, p)
                break
            except Singular:
                q = self._rand_q(p)
        self.nevals += 1
        self.teval += time.time() - t
        out = np.zeros(self.eng.K, dtype=np.int64)
        out[:len(v)] = v[:self.eng.K]
        return q, out

    def _detect(self, p):
        """parity (per A-coefficient: f(-q) = s f(q)) and conjugation symmetry f(-1/q) = f(q)"""
        pts = []
        signs = None
        for _ in range(2):
            q, v = self._eval(self._rand_q(p), p)
            _, vm = self._eval((-q) % p, p)
            pts += [(q, v), ((-q) % p, vm)]
            s = []
            for a, b in zip(v, vm):
                if a == 0 and b == 0:
                    s.append(0)
                elif a == b:
                    s.append(1)
                elif (a + b) % p == 0:
                    s.append(-1)
                else:
                    s.append(None)
            if signs is None:
                signs = s
            else:
                signs = [x if (x == y or y == 0) else (y if x == 0 else None) for x, y in zip(signs, s)]
        parity = all(x is not None for x in signs)
        sym = False
        if self.eng.R == conj(self.eng.R):
            q1, v1 = pts[0]
            qi = (-pow(q1, p - 2, p)) % p
            _, v2 = self._eval(qi, p)
            pts.append((qi, v2))
            sym = bool(np.all(v1 == v2))
        e = [1 if x == -1 else 0 for x in signs] if parity else [0] * len(signs)
        return parity, e, sym, pts

    def _interpolate_prime(self, p, mode=None):
        if mode is None:
            mode = self._detect(p)
        parity, e, sym, pts = mode
        e = np.array(e, dtype=np.int64)
        zs, vals, seen = [], [], set()

        def add(q, v):
            z = q * q % p if parity else q
            if z in seen:
                return
            seen.add(z)
            qe = powmod_arr(np.full(len(e), q), 1, p)
            div = np.where(e == 1, inv_arr(qe, p), 1)
            zs.append(z)
            vals.append(v * div % p)

        def add_all(q, v):
            add(q, v)
            if sym:
                add((-pow(q, p - 2, p)) % p, v)
        for q, v in pts:
            add_all(q, v)
        nextcheck = 8
        while True:
            while len(zs) < nextcheck:
                q, v = self._eval(self._rand_q(p), p)
                add_all(q, v)
            res = laurent_reconstruct(zs, np.array(vals), p)
            if res is not None:
                break
            nextcheck = int(len(zs) * 1.15) + 2
            if self.verbose:
                print(f"#   p={p}: {len(zs)} points, not yet ({self.nevals} evals, {self.teval:.1f}s)",
                      file=sys.stderr)
        # assemble {(a_exp, q_exp): residue}
        out = {}
        step = 2 if parity else 1
        for k, (L, N) in enumerate(res):
            for d, c in enumerate(N):
                if c:
                    out[(self.aexp[k], step * (d - L) + int(e[k]))] = int(c)
        if self.verbose:
            print(f"#   p={p}: reconstructed from {len(zs)} points ({self.nevals} evals, {self.teval:.1f}s)",
                  file=sys.stderr)
        return out, mode

    def _check(self, poly, p, trials=2):
        for _ in range(trials):
            q, v = self._eval(self._rand_q(p), p)
            pred = np.zeros(self.eng.K, dtype=np.int64)
            for (a, b), c in poly.items():
                k = self.aexp.index(a)
                pred[k] = (pred[k] + (c % p) * pow(q, b % (p - 1), p)) % p
            if not np.all(pred == v):
                return False
        return True

    def compute(self):
        t0 = time.time()
        used = []
        resid = []
        for i, p in enumerate(PRIMES):
            r, _ = self._interpolate_prime(p)
            used.append(p)
            resid.append(r)
            keys = set().union(*[set(x) for x in resid])
            poly = {}
            for kk in keys:
                c = _crt([x.get(kk, 0) for x in resid], used)
                if c:
                    poly[kk] = c
            if self._check(poly, PRIMES[i + 1]):
                self.poly = poly
                self.time = time.time() - t0
                return poly
            if self.verbose:
                print("# verification with next prime failed -> adding a prime (CRT)", file=sys.stderr)
        raise RuntimeError("did not converge")


# ----------------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------------

def to_string(poly, var=("A", "q")):
    if not poly:
        return "0"
    terms = []
    for (a, b) in sorted(poly, key=lambda t: (-t[0], -t[1])):
        c = poly[(a, b)]
        mon = []
        if a:
            mon.append(f"{var[0]}^{a}" if a != 1 else var[0])
        if b:
            mon.append(f"{var[1]}^{b}" if b != 1 else var[1])
        m = "*".join(mon)
        if not m:
            terms.append(f"{c:+d}")
        elif c == 1:
            terms.append("+" + m)
        elif c == -1:
            terms.append("-" + m)
        else:
            terms.append(f"{c:+d}*{m}")
    s = " ".join(terms)
    return s[1:] if s.startswith("+") else s


def to_mathematica(poly):
    """Mathematica-ready expression in variables A, q"""
    return to_string(poly).replace(" ", "")


def transpose_poly(poly):
    """H_{R^T}(A, q) = H_R(A, -1/q)"""
    return {(a, -b): (c if b % 2 == 0 else -c) for (a, b), c in poly.items()}


def evaluate_poly(poly, A, q):
    return sum(c * A ** a * q ** b for (a, b), c in poly.items())


KNOTS = {   # minimal braid words (Knot Atlas conventions)
    "3_1": (2, [1, 1, 1]),
    "4_1": (3, [1, -2, 1, -2]),
    "5_1": (2, [1, 1, 1, 1, 1]),
    "7_1": (2, [1] * 7),
    "8_19": (3, [1, 1, 1, 2, 1, 1, 1, 2]),
    "9_34": (4, [-1, 2, -1, 2, -3, 2, -1, 2, -3]),
}


def parse_rep(s):
    s = s.strip().strip("[]()")
    return tuple(int(x) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser(description="Colored HOMFLY-PT via cabling")
    ap.add_argument("--knot", help="knot name from the built-in table, e.g. 9_34")
    ap.add_argument("--braid", help="braid word, comma separated signed generators, e.g. -1,2,-1,2")
    ap.add_argument("--strands", type=int, help="number of strands (default: max|g|+1)")
    ap.add_argument("--rep", required=True, help="Young diagram, e.g. 2,1")
    ap.add_argument("--json", help="write result to this JSON file")
    ap.add_argument("--transpose", action="store_true",
                    help="also print H for the transposed diagram R^T (free: q -> -1/q)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.knot:
        m, braid = KNOTS[a.knot]
    else:
        braid = [int(x) for x in a.braid.split(",")]
        m = a.strands or max(abs(g) for g in braid) + 1
    R = parse_rep(a.rep)
    H = ColoredHOMFLY(braid, m, R, verbose=a.verbose)
    poly = H.compute()
    print(to_string(poly))
    if a.transpose:
        print(f"# H_{list(conj(R))}:")
        print(to_string(transpose_poly(poly)))
    print(f"# {H.nevals} evaluations, {H.time:.2f}s total", file=sys.stderr)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(knot=a.knot, braid=braid, strands=m, rep=list(R),
                           conventions="A=q^N, (g-q)(g+1/q)=0, H(unknot)=1, topological framing",
                           terms=[[aa, bb, c] for (aa, bb), c in sorted(poly.items())],
                           string=to_string(poly)), f, indent=1)


if __name__ == "__main__":
    main()
