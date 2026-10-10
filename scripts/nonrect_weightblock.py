"""Weight-block evaluation of the (1,1)-tangle invariant of V_R (vertex model of nonrect_supergroup.py).

Same module (ns.Module: U_q gl(N|M) on V_R, inductive coordinates), but
  * the Cartan generators H_i are carried through the induction (H_{t+1} = Rb (H_t x 1 + 1 x h_i) Lb) and V_R is
    re-expressed in a weight basis (simultaneous eigenbasis, orthonormalised inside each weight space);
  * R_{V_R,V_R} is then weight-conserving and the pivotal element mu is diagonal;
  * the braid closure on strands 2..b is evaluated block by block in total weight W: only the input vectors
    e_top (x) e_c (e_top = highest-weight vector, 1-dim weight space) are propagated, inside the block of W, and the
    crossings act as dense matrices on the pair-weight blocks of V_R (x) V_R (gather / matmul / scatter).
The scalar is <e_top (x) e_c | B (1 x mu^{(x) b-1}) | e_top (x) e_c> summed over c, divided by twist^writhe, i.e. the
same quantity as ns.Module.knot_value (row 0), but with cost  sum_W n_W m_W <n_pair>  instead of k^(2b+1).
A second vector (another basis vector of a 1-dim weight space, if present) gives the scalar check.

usage: python3 scripts/nonrect_weightblock.py test     (compares with ns.Module.knot_value and data)
"""
import os
import sys
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_supergroup as ns  # noqa: E402


class WModule:
    def __init__(self, par, R, q, mod=None):
        self.mod = mod = mod if mod is not None else ns.Module(par, R, q)
        self.k = k = mod.k
        if k == 0:
            return
        d = len(par)
        # Cartan generators along the induction
        H = [np.diag([1.0 + 0j if a == i else 0j for a in range(d)]) for i in range(d)]
        h1 = [h.copy() for h in H]
        for (Lb, Rb, ks) in mod.chain:
            H = [Rb @ (ns.kr(Hi, np.eye(d)) + ns.kr(np.eye(ks), hi)) @ Lb for Hi, hi in zip(H, h1)]
        rng = np.random.default_rng(7)
        cvec = rng.uniform(1, 2, d) * np.array([1, 7.31, 51.7, 377.3, 2711.1, 19927.0][:d])
        Ht = sum(c * Hi for c, Hi in zip(cvec, H))
        w, W = np.linalg.eig(Ht)
        # weights of eigenvectors
        Wi = np.linalg.inv(W)
        wts = np.array([[(Wi @ Hi @ W)[a, a].real for Hi in H] for a in range(k)])
        self.wt_err = np.abs(wts - np.round(wts)).max()
        wts = np.round(wts).astype(int)
        keys = sorted(set(map(tuple, wts)), reverse=True)
        cols, wlist = [], []
        for key in keys:
            idx = [a for a in range(k) if tuple(wts[a]) == key]
            Qm, _ = np.linalg.qr(W[:, idx])
            cols.append(Qm)
            wlist += [key] * len(idx)
        Wb = np.hstack(cols)
        Wbi = np.linalg.inv(Wb)
        self.cond = np.linalg.cond(Wb)
        self.wts = np.array(wlist)
        # check block-diagonality of H in new basis
        self.H_dev = max(np.abs(Wbi @ Hi @ Wb - np.diag(self.wts[:, i])).max() for i, Hi in enumerate(H))
        mu = Wbi @ mod.mu @ Wb
        self.mu_dev = np.abs(mu - np.diag(np.diag(mu))).max()
        self.mud = np.diag(mu).copy()
        WW, WWi = np.kron(Wb, Wb), np.kron(Wbi, Wbi)
        C = WWi @ mod.C @ WW
        Ci = WWi @ mod.Ci @ WW
        self.twist = mod.twist
        # pair-weight blocks of V (x) V
        pw = {}
        for a in range(k):
            for b in range(k):
                pw.setdefault(tuple(self.wts[a] + self.wts[b]), []).append(a * k + b)
        self.pairblocks = {}
        leak = 0.0
        for key, idx in pw.items():
            idx = np.array(idx)
            self.pairblocks[key] = (idx, C[np.ix_(idx, idx)].copy(), Ci[np.ix_(idx, idx)].copy())
            mask = np.ones(k * k, bool)
            mask[idx] = False
            leak = max(leak, np.abs(C[np.ix_(mask, idx)]).max() if mask.any() else 0.0)
        self.C_leak = leak
        self.top = 0                         # highest weight (keys sorted descending lexicographically)
        one_dim = [a for a in range(k) if (self.wts == self.wts[a]).all(axis=1).sum() == 1]
        self.alt = [a for a in one_dim if a != self.top][-1] if len(one_dim) > 1 else None
        self.rel = self.wts - self.wts[self.top]
        self.Cfull, self.Cifull = C, Ci


# ================================================================================ block engine
OFF, BASE = 64, 256


def wcode(w):
    """integer code of (arrays of) relative weights"""
    w = np.asarray(w)
    return ((w + OFF) * (BASE ** np.arange(w.shape[-1]))).sum(axis=-1)


def lin(states, k):
    L = states.shape[1]
    return states @ (k ** np.arange(L - 1, -1, -1)).astype(np.int64) if L else np.zeros(len(states), np.int64)


class Structure:
    """weight-graded tensor powers of a module with relative weights `rel` (k x d ints)"""

    def __init__(self, rel):
        self.rel = np.asarray(rel)
        self.k = len(rel)
        self.codes1 = wcode(self.rel)
        self._loc = {}

    def all_states(self, L):
        k = self.k
        st = np.indices((k,) * L).reshape(L, -1).T.astype(np.int64) if L else np.zeros((1, 0), np.int64)
        return st

    def local(self, L):
        """L-strand states grouped by weight code: dict code -> sorted lin indices"""
        if L not in self._loc:
            st = self.all_states(L)
            codes = wcode(self.rel[st].sum(axis=1)) if L else np.array([wcode(np.zeros(self.rel.shape[1], int))])
            li = lin(st, self.k)
            order = np.lexsort((li, codes))
            codes_s, li_s = codes[order], li[order]
            cut = np.flatnonzero(np.diff(codes_s)) + 1
            self._loc[L] = {int(c[0]): l for c, l in zip(np.split(codes_s, cut), np.split(li_s, cut))}
        return self._loc[L]

    def blocks(self, n):
        """n-strand states grouped by total relative weight: dict code -> states array"""
        st = self.all_states(n)
        codes = wcode(self.rel[st].sum(axis=1))
        order = np.argsort(codes, kind="stable")
        codes_s = codes[order]
        cut = np.flatnonzero(np.diff(codes_s)) + 1
        return {int(c[0]): st[o] for c, o in zip(np.split(codes_s, cut), np.split(order, cut))}


def op_grids(S, states, p0, L):
    """gather structure of a local op on strands p0..p0+L-1 for the block `states`"""
    k = S.k
    loc = states[:, p0:p0 + L]
    ll = lin(loc, k)
    lc = wcode(S.rel[loc].sum(axis=1))
    other = np.delete(states, list(range(p0, p0 + L)), axis=1)
    ol = lin(other, k)
    out = []
    for c in np.unique(lc):
        ii = np.flatnonzero(lc == c)
        idx = S.local(L)[int(c)]
        pos = np.searchsorted(idx, ll[ii])
        uo, oinv = np.unique(ol[ii], return_inverse=True)
        grid = np.full((len(uo), len(idx)), -1, dtype=np.int64)
        grid[oinv, pos] = ii
        assert (grid >= 0).all()
        out.append((int(c), grid))
    return out


def crossing_blocks(S, wm):
    """local blocks (code -> (M+, M-)) of R_{V,V} in the sorted-lin order of S.local(2)"""
    out = {}
    for c, idx in S.local(2).items():
        out[c] = (wm.Cfull[np.ix_(idx, idx)], wm.Cifull[np.ix_(idx, idx)])
    return out


def propagate(X, plan_ops, mats):
    """apply ops (each: (grids, key)) to X; mats[key][code] = matrix"""
    for grids, key in plan_ops:
        Y = np.empty_like(X)
        M = mats[key]
        for c, grid in grids:
            Y[grid] = np.matmul(M[c], X[grid])
        X = Y
    return X


class Plan:
    """Symbolic evaluation plan for the (1,1)-tangle of the closure of a braid word.

    ops: list of (p0, L, key); key = ('X', +-1) for a crossing, ('T', i) for an eliminated-strand operator.
    Elimination of the last strand: the shortest cyclic window containing every op touching it is replaced by
    T = tr_last(mu_last * window) acting on strands j0..last-1 (requires j0 >= 1, i.e. the open strand 0 untouched).
    """

    def __init__(self, word, S, max_sub=4, verbose=False):
        self.S = S
        b = max(abs(g) for g in word) + 1
        self.word = word
        ops = [(abs(g) - 1, 2, ('X', 1 if g > 0 else -1)) for g in word]
        self.Tdefs = []                    # (nsub, ops of the window re-indexed)
        n = b
        while n > 2:
            last = n - 1
            touch = [i for i, o in enumerate(ops) if o[0] + o[1] - 1 == last]
            best = None
            N = len(ops)
            for s in touch:
                # window starting at s, going forward cyclically, ending at the last touching op
                rel_pos = sorted((t - s) % N for t in touch)
                length = rel_pos[-1] + 1
                win = [ops[(s + t) % N] for t in range(length)]
                j0 = min(o[0] for o in win)
                if best is None or (j0, -length) > (best[0], -best[1]):
                    best = (j0, length, s)
            j0, length, s = best
            nsub = n - j0
            if j0 < 1 or nsub > max_sub:
                break
            rot = ops[s:] + ops[:s]
            win, rest = rot[:length], rot[length:]
            self.Tdefs.append((nsub, [(o[0] - j0, o[1], o[2]) for o in win]))
            ops = [(j0, nsub - 1, ('T', len(self.Tdefs) - 1))] + rest
            n -= 1
            if verbose:
                print("eliminated strand %d: window %d ops, T on strands %d..%d" % (last, length, j0, last - 1))
        self.n, self.ops = n, ops
        self._cache = {}

    def grids(self, nstr, ops, Wcode, states):
        key = (nstr, tuple(ops), Wcode)
        if key not in self._cache:
            self._cache[key] = [(op_grids(self.S, states, p0, L), k) for (p0, L, k) in ops]
        return self._cache[key]

    def build_T(self, i, mats, mud, maxmem):
        nsub, ops = self.Tdefs[i]
        S = self.S
        loc = S.local(nsub - 1)
        T = {c: np.zeros((len(idx), len(idx)), dtype=complex) for c, idx in loc.items()}
        for Wc, states in S.blocks(nsub).items():
            n = len(states)
            pl = self.grids(nsub, ops, ("T", i, Wc), states)
            last = states[:, -1]
            sl = lin(states[:, :-1], S.k)
            scode = wcode(S.rel[states[:, :-1]].sum(axis=1))
            chunk = max(1, int(maxmem // (16 * n * 3)))
            for c0 in range(0, n, chunk):
                c1 = min(n, c0 + chunk)
                X = np.zeros((n, c1 - c0), dtype=complex)
                X[np.arange(c0, c1), np.arange(c1 - c0)] = 1
                X = propagate(X, pl, mats)
                # T[s', s] += mu_c X[(s',c), (s,c)]
                lastc = last[c0:c1]
                for c in np.unique(lastc):
                    rows = np.flatnonzero(last == c)
                    cols = np.flatnonzero(lastc == c)
                    code = int(scode[rows[0]])
                    idx = loc[code]
                    pr = np.searchsorted(idx, sl[rows])
                    pc = np.searchsorted(idx, sl[c0 + cols])
                    T[code][np.ix_(pr, pc)] += mud[c] * X[np.ix_(rows, cols)]
        return T

    def value(self, wm, maxmem=3e8, start=None):
        S = self.S
        assert (wm.rel == S.rel).all()
        mats = {('X', 1): {c: m[0] for c, m in crossing_blocks(S, wm).items()},
                ('X', -1): {c: m[1] for c, m in crossing_blocks(S, wm).items()}}
        for i in range(len(self.Tdefs)):
            mats[('T', i)] = self.build_T(i, mats, wm.mud, maxmem)
        n = self.n
        s0 = wm.top if start is None else start
        tot = 0j
        rest = S.blocks(n - 1)
        full = S.blocks(n)
        for rc, cs in rest.items():
            Wc = int(wcode(S.rel[s0] + S.rel[cs[0]].sum(axis=0)))
            states = full[Wc]
            order = np.argsort(lin(states, S.k))
            ls = lin(states, S.k)[order]
            inp = np.hstack([np.full((len(cs), 1), s0, np.int64), cs])
            inpos = order[np.searchsorted(ls, lin(inp, S.k))]
            pl = self.grids(n, self.ops, ("main", Wc), states)
            m = len(inp)
            chunk = max(1, int(maxmem // (16 * len(states) * 3)))
            for c0 in range(0, m, chunk):
                c1 = min(m, c0 + chunk)
                X = np.zeros((len(states), c1 - c0), dtype=complex)
                X[inpos[c0:c1], np.arange(c1 - c0)] = 1
                X = propagate(X, pl, mats)
                diag = X[inpos[c0:c1], np.arange(c1 - c0)]
                muf = np.prod(wm.mud[inp[c0:c1, 1:]], axis=1) if n > 1 else 1
                tot += (diag * muf).sum()
        wr = sum(1 if g > 0 else -1 for g in self.word)
        return tot / wm.twist ** wr


def best_plan(word, S, max_sub=4):
    """Plan for the word or its reflection sigma_i -> sigma_{b-i} (conjugation by the half twist: same closure),
    whichever has fewer remaining strands, then smaller largest eliminated subsystem."""
    b = max(abs(g) for g in word) + 1
    cands = []
    for w in (list(word), [(1 if g > 0 else -1) * (b - abs(g)) for g in word]):
        P = Plan(w, S, max_sub=max_sub)
        cands.append(((P.n, max([t[0] for t in P.Tdefs] or [0]), sum(len(t[1]) for t in P.Tdefs)), P))
    return min(cands, key=lambda c: c[0])[1]


def test():
    rows = ns.knot_rows()
    q = 1 / np.exp(0.6j)
    for par, R, knots in [("011", (3, 1), ["3_1", "4_1", "5_2", "6_2", "8_19", "10_132", "11n_34", "12n_457"]),
                          ("0111", (4, 1), ["3_1", "4_1", "5_2"]),
                          ("0011", (3, 2), ["3_1"]), ("01", (3, 1, 1), ["3_1", "4_1", "8_19"]),
                          ("011", (3,), ["7_4", "9_42", "11n_67"])]:
        wm = WModule([int(c) for c in par], R, q)
        S = Structure(wm.rel)
        print(par, R, "dim", wm.k, "wt_err %.1e H_dev %.1e mu_dev %.1e C_leak %.1e cond %.1e" % (
            wm.wt_err, wm.H_dev, wm.mu_dev, wm.C_leak, wm.cond))
        N, M = par.count("0"), par.count("1")
        for kn in knots:
            w = ns.braid_word(rows[kn])
            b = max(map(abs, w)) + 1
            v = Plan(w, S, max_sub=4).value(wm)
            v1 = Plan(w, S, max_sub=0).value(wm)
            v2 = Plan(w, S, max_sub=4).value(wm, start=wm.alt)
            v0 = wm.mod.knot_value(w)[0] if wm.k ** (2 * b - 1) < 1e9 else None
            h = ns.data_value(kn, R, (1 / q) ** (N - M), 1 / q)
            print("  %s b=%d plan_n=%d v=%.10f%+.10fi noelim_diff=%.1e old_diff=%s data_diff=%s scal=%.1e" % (
                kn, b, Plan(w, S).n, v.real, v.imag, abs(v - v1), "%.1e" % abs(v - v0) if v0 is not None else "-",
                "%.1e" % abs(v - h) if h is not None else "-", abs(v - v2)))


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
