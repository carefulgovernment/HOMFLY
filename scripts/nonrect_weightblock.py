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
    def __init__(self, par, R, q):
        self.mod = mod = ns.Module(par, R, q)
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

    # ------------------------------------------------------------------------------------------------
    def knot_value(self, word, check_scalar=False, maxmem=4e8):
        k = self.k
        b = max(abs(x) for x in word) + 1
        wr = sum(1 if g > 0 else -1 for g in word)
        starts = [self.top] + ([self.alt] if (check_scalar and self.alt is not None) else [])
        wts = self.wts
        nw = wts.shape[1]
        # enumerate states of strands 2..b by weight
        rest = {}
        for c in product(range(k), repeat=b - 1):
            rest.setdefault(tuple(wts[list(c)].sum(axis=0)), []).append(c)
        # all states of V^{(x)b} by total weight, lazily per needed W
        allw = {}
        for c in product(range(k), repeat=b):
            allw.setdefault(tuple(wts[list(c)].sum(axis=0)), []).append(c)
        res = []
        for s0 in starts:
            tot = 0j
            for wr_key, cs in rest.items():
                Wtot = tuple(np.array(wr_key) + wts[s0])
                states = np.array(allw[Wtot], dtype=np.int64)        # (n, b)
                n = len(states)
                mult = k ** np.arange(b - 1, -1, -1)
                lin = states @ mult
                order = np.argsort(lin)
                lin_sorted = lin[order]

                def find(arr):
                    pos = np.searchsorted(lin_sorted, arr @ mult)
                    return order[pos]
                inp = np.array([(s0,) + c for c in cs], dtype=np.int64)
                inpos = find(inp)
                m = len(inp)
                # crossing gather structure per position p (cached per block)
                ops = {}
                for p in set(abs(g) - 1 for g in word):
                    pk = states[:, p] * k + states[:, p + 1]
                    pwk = wts[states[:, p]] + wts[states[:, p + 1]]
                    groups = []
                    keyarr = [tuple(x) for x in pwk]
                    bykey = {}
                    for i, kk in enumerate(keyarr):
                        bykey.setdefault(kk, []).append(i)
                    for kk, ii in bykey.items():
                        idx, Cb, Cib = self.pairblocks[kk]
                        ii = np.array(ii)
                        npair = len(idx)
                        pos_in_pair = np.searchsorted(idx, pk[ii])   # idx sorted ascending
                        other = np.delete(states[ii], [p, p + 1], axis=1)
                        olin = other @ (k ** np.arange(b - 3, -1, -1)) if b > 2 else np.zeros(len(ii), np.int64)
                        uo, oinv = np.unique(olin, return_inverse=True)
                        grid = np.full((len(uo), npair), -1, dtype=np.int64)
                        grid[oinv, pos_in_pair] = ii
                        assert (grid >= 0).all()
                        groups.append((grid, Cb, Cib))
                    ops[p] = groups
                chunk = max(1, int(maxmem // (16 * n * 3)))
                for c0 in range(0, m, chunk):
                    c1 = min(m, c0 + chunk)
                    X = np.zeros((n, c1 - c0), dtype=complex)
                    X[inpos[c0:c1], np.arange(c1 - c0)] = 1
                    for g in word:
                        p = abs(g) - 1
                        Y = np.empty_like(X)
                        for grid, Cb, Cib in ops[p]:
                            M = Cb if g > 0 else Cib
                            G = X[grid]                       # (no, npair, cols)
                            Y[grid] = np.einsum("ij,ojc->oic", M, G, optimize=True)
                        X = Y
                    # mu on strands 2..b (diagonal), output component = input state
                    rows = inpos[c0:c1]
                    diag = X[rows, np.arange(c1 - c0)]
                    muf = np.prod(self.mud[inp[c0:c1, 1:]], axis=1)
                    tot += (diag * muf).sum()
            res.append(tot / self.twist ** wr)
        return res[0], (abs(res[1] - res[0]) if len(res) > 1 else float("nan"))


def test():
    rows = ns.knot_rows()
    q = 1 / np.exp(0.6j)
    for par, R, knots in [("011", (3, 1), ["3_1", "4_1", "5_2", "6_2"]), ("0111", (4, 1), ["3_1", "4_1"]),
                          ("0011", (3, 2), ["3_1"]), ("01", (3, 1, 1), ["3_1", "4_1", "8_19"]),
                          ("011", (3,), ["7_4", "9_42"])]:
        wm = WModule([int(c) for c in par], R, q)
        print(par, R, "dim", wm.k, "wt_err %.1e H_dev %.1e mu_dev %.1e C_leak %.1e cond %.1e" % (
            wm.wt_err, wm.H_dev, wm.mu_dev, wm.C_leak, wm.cond))
        N, M = par.count("0"), par.count("1")
        for kn in knots:
            w = ns.braid_word(rows[kn])
            v, dv = wm.knot_value(w, check_scalar=True)
            v0, _ = wm.mod.knot_value(w)
            h = ns.data_value(kn, R, (1 / q) ** (N - M), 1 / q)
            print("  %s b=%d new=%.10f%+.10fi old_diff=%.1e data_diff=%s scal=%.1e" % (
                kn, max(map(abs, w)) + 1, v.real, v.imag, abs(v - v0),
                "%.1e" % abs(v - h) if h is not None else "-", dv))


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
