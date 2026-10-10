"""k = 2 families: two-row R = (r1, r2) on A = q^-m beyond the gl(1|m+1) hook (r2 > m+1), vertex model of
U_q gl(2|m+2) on V_R (nonrect_supergroup / nonrect_weightblock), independent of data/homfly.

The module is built along the column-reading tableau (boxes (0,0),(1,0),(0,1),(1,1),...), so that every
intermediate module has dimension <= dim V_R (the row-reading order of ns.Module passes through Sym^{r1}, whose
dimension grows with r1).  V_R does not depend on the standard tableau (image of a primitive idempotent).

usage:
  python3 scripts/nonrect_k2_family.py vals PAR d r2_0 r2_1 theta KNOT1,KNOT2.. > file  (q'=e^{i theta}, R=(r2+d,r2))
  python3 scripts/nonrect_k2_family.py check PAR R1,R2 KNOT...                     (compare with data/homfly)
"""
import cmath
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_supergroup as ns  # noqa: E402
import nonrect_weightblock as wb  # noqa: E402
from nonrect_supergroup import kr, perk_schultz, pivot, addable_contents  # noqa: E402


class ColModule(ns.Module):
    """ns.Module with an arbitrary box order (default: column reading)."""

    def __init__(self, par, R, q, boxes=None):
        self.par, self.R, self.q = par, tuple(R), q
        d = self.d = len(par)
        T = self.T = perk_schultz(par, q)
        I_d = np.eye(d)
        mu1, _ = pivot(T, d)
        mu1 = np.diag(mu1)
        if boxes is None:
            boxes = sorted([(i, j) for i, row in enumerate(R) for j in range(row)], key=lambda b: (b[1], b[0]))
        chain = []
        k, D, E, mu = d, T.copy(), T.copy(), mu1
        assert boxes[0] == (0, 0)
        shape = [1]
        self.worst_proj = 0.0
        for (i, j) in boxes[1:]:
            c = j - i
            L = D @ E
            targets = [q ** (2 * c2) for c2 in addable_contents(shape)]
            w, W = np.linalg.eig(L)
            near = np.array([min(range(len(targets)), key=lambda t: abs(x - targets[t])) for x in w])
            err = max(abs(x - targets[t]) for x, t in zip(w, near)) if len(w) else 0.0
            self.worst_proj = max(self.worst_proj, err)
            sel = np.where(near == addable_contents(shape).index(c))[0]
            kn = len(sel)
            if kn == 0:
                self.k = 0
                return
            Wi = np.linalg.inv(W)
            P = W[:, sel] @ Wi[sel, :]
            U, _, _ = np.linalg.svd(P)
            Lb = U[:, :kn]
            Rb = Lb.conj().T @ P
            Ikt = np.eye(k)
            Dn = kr(Rb, I_d) @ kr(Ikt, T) @ kr(D, I_d) @ kr(I_d, Lb)
            En = kr(I_d, Rb) @ kr(E, I_d) @ kr(Ikt, T) @ kr(Lb, I_d)
            mu = Rb @ kr(mu, mu1) @ Lb
            chain.append((Lb, Rb, k))
            k, D, E = kn, Dn, En
            if i == len(shape):
                shape.append(1)
            else:
                shape[i] += 1
        self.k, self.mu, self.D, self.chain = k, mu, D, chain
        K = k
        F = D
        D4 = D.reshape(K, d, d, K)
        for (Lb, Rb, ks) in chain:
            kn = Lb.shape[1]
            L3 = Lb.reshape(ks, d, kn)
            Y = np.einsum("Bjib,aiz->aBjzb", D4, L3, optimize=True)
            F4 = F.reshape(K, ks, ks, K)
            Y = np.einsum("Ceab,abjzy->Cejzy", F4, Y, optimize=True)
            R3 = Rb.reshape(kn, ks, d)
            F = np.einsum("gej,Cejzy->Cgzy", R3, Y, optimize=True).reshape(K * kn, kn * K)
        self.C = F
        self.Ci = np.linalg.inv(F)
        Cm = np.einsum("iaja->ij", (kr(np.eye(K), mu) @ F).reshape(K, K, K, K))
        self.twist = Cm[0, 0]
        self.twist_dev = np.abs(Cm - Cm[0, 0] * np.eye(K)).max() / abs(Cm[0, 0])


_plans = {}


def values(par, R, words, qd, check_scalar=False, maxsub=3):
    """H_R at q' = qd for several braid words from one module: list of (v, dim, scalar_dev, health)"""
    q = 1 / qd
    mod = ColModule(par, R, q)
    if mod.k == 0:
        return [(0j, 0, float("nan"), 0.0) for _ in words]
    wm = wb.WModule(par, R, q, mod=mod)
    health = max(wm.wt_err, wm.H_dev, wm.mu_dev, wm.C_leak, mod.twist_dev, mod.worst_proj)
    out = []
    for word in words:
        key = (tuple(word), wm.rel.tobytes())
        if key not in _plans:
            _plans[key] = wb.best_plan(word, wb.Structure(wm.rel), max_sub=maxsub)
        P = _plans[key]
        v = P.value(wm)
        dv = (abs(P.value(wm, start=wm.alt) - v) / max(1, abs(v)) if (check_scalar and wm.alt is not None)
              else float("nan"))
        out.append((v, wm.k, dv, health))
    return out


def value(par, R, word, qd, check_scalar=False, maxsub=3):
    return values(par, R, [word], qd, check_scalar, maxsub)[0]


def main():
    rows = ns.knot_rows()
    if sys.argv[1] == "check":
        par = [int(c) for c in sys.argv[2]]
        R = tuple(int(x) for x in sys.argv[3].split(","))
        m = par.count(1) - par.count(0)
        qd = cmath.exp(0.6j)
        for kn in sys.argv[4:]:
            v, k, dv, h = value(par, R, ns.braid_word(rows[kn]), qd, check_scalar=True)
            hd = ns.data_value(kn, R, qd ** (-m), qd)
            print("%s R=%s dim=%d v=%s data_diff=%s scal=%.1e health=%.1e" % (
                kn, R, k, v, "%.1e" % (abs(v - hd) / max(1, abs(hd))) if hd is not None else "-", dv, h))
    elif sys.argv[1] == "vals":
        par = [int(c) for c in sys.argv[2]]
        d, a, b, theta = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
        knots = sys.argv[7].split(",")
        qd = cmath.exp(1j * theta)
        words = [ns.braid_word(rows[kn]) for kn in knots]
        for r2 in range(a, b + 1):
            res = values(par, (r2 + d, r2), words, qd, check_scalar=(r2 % 10 == 0))
            for kn, (v, k, dv, h) in zip(knots, res):
                print("%s\t%d\t%d\t%d\t%.17g\t%.17g\t%.1e\t%.1e" % (kn, r2 + d, r2, k, v.real, v.imag, dv, h),
                      flush=True)


if __name__ == "__main__":
    main()
