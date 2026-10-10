"""U_q gl(N|M) vertex model on covariant modules V_R, for arbitrary Young diagrams R.

Generalises scripts/supergroup_vertex_check.py (Sym^r only) to any R and to large |R|.
Plain complex linear algebra, independent of the Hecke/Racah/interpolation methods behind data/homfly.

  * V = C^{N|M}, Perk-Schultz matrix T on V (x) V, (T - q)(T + 1/q) = 0.
  * V_R is built inductively along the row-reading standard tableau of R: V_{t+1} is the image, inside
    V_t (x) V, of the spectral projector of the Jucys-Murphy element L_{n} = c_{V,V_t} c_{V_t,V} onto the
    eigenvalue q^{2c}, c = content of the added box (with --conj: q^{-2c}).  This is the image of the
    primitive seminormal idempotent E_T, so V_R ~ E_T V^{(x)|R|}.  Everything is kept in these inductive
    coordinates: dim V_t only, never d^|R|, so |R| can be large.
  * Braidings R_{V,V_t}, R_{V_t,V}, R_{V_R,V_R} and the pivotal element mu_R are carried along the
    induction.  The knot value is the scalar of the partial (pivotal) trace over strands 2..b of the braid,
    divided by twist^writhe.

Comparison with the data (as in supergroup_vertex_check.py): vertex model at q  <->  H_R(A = q^{M-N}, 1/q),
i.e. H_R on the line A = q'^{-(M-N)} at q' = 1/q.

usage (check against data):
  python3 scripts/nonrect_supergroup.py check PARITIES R1,R2,.. MAXB [knot ...]
       e.g.  check 011 3,1 4         gl(1|2), V_[3,1], all knots with braid index <= 4
"""
import cmath
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from homfly.algebra.laurent import parse_laurent  # noqa: E402

KNOTINFO = os.path.join(HERE, "..", "src", "homfly", "knots", "data", "knotinfo_upto12.csv")
DATA = os.path.join(HERE, "..", "data", "homfly")


# ----------------------------------------------------------------------------------------------- algebra
def perk_schultz(par, q):
    n = len(par)
    T = np.zeros((n * n, n * n), dtype=complex)
    for i in range(n):
        T[i * n + i, i * n + i] = q if par[i] == 0 else -1 / q
        for j in range(i + 1, n):
            s = (-1) ** (par[i] * par[j])
            T[j * n + i, i * n + j] = s
            T[i * n + j, j * n + i] = s
            T[j * n + i, j * n + i] = q - 1 / q
    return T


def pivot(T, d):
    """diagonal mu with tr_2((1 x mu) T^{+-1}) = kappa^{+-1} Id"""
    A = []
    for which, Mx in enumerate((T, np.linalg.inv(T))):
        M4 = Mx.reshape(d, d, d, d)
        for i in range(d):
            for ip in range(d):
                row = [M4[i, j, ip, j] for j in range(d)] + [0.0, 0.0]
                if i == ip:
                    row[d + which] = -1.0
                A.append(row)
    _, s, Vh = np.linalg.svd(np.array(A, dtype=complex))
    x = Vh[-1].conj()
    mu, ka, kb = x[:d], x[d], x[d + 1]
    sc = 1 / cmath.sqrt(ka * kb)
    return mu * sc, ka * sc


def kr(*ms):
    out = np.eye(1, dtype=complex)
    for m in ms:
        out = np.kron(out, m)
    return out


def path_contents(R):
    return [j - i for i, row in enumerate(R) for j in range(row)]


def addable_contents(shape):
    sh = list(shape)
    out = []
    for i in range(len(sh) + 1):
        ri = sh[i] if i < len(sh) else 0
        if i == 0 or sh[i - 1] > ri:
            out.append(ri - i)
    return out


class Module:
    """V_R for U_q gl(N|M), parities `par`, generic complex q."""

    def __init__(self, par, R, q, conj=False, kmax=None):
        self.par, self.R, self.q = par, tuple(R), q
        d = self.d = len(par)
        T = self.T = perk_schultz(par, q)
        I_d = np.eye(d)
        mu1, _ = pivot(T, d)
        mu1 = np.diag(mu1)
        sgn = -1 if conj else 1
        chain = []                      # (Lb, Rb) for steps 2..n
        k, D, E, mu = d, T.copy(), T.copy(), mu1
        shape = [1]
        self.worst_proj = 0.0
        boxes = [(i, j) for i, row in enumerate(R) for j in range(row)]
        for (i, j) in boxes[1:]:
            c = j - i
            L = D @ E                   # Jucys-Murphy on V_t (x) V
            lam = q ** (2 * sgn * c)
            targets = [q ** (2 * sgn * c2) for c2 in addable_contents(shape)]
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
            P = W[:, sel] @ Wi[sel, :]          # spectral projector; orthonormal basis of its image
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
        if kmax is not None and k > kmax:
            self.C = None
            return
        # R-matrix on V_R (x) V_R: F_s = R_{V_s, V_R}
        K = k
        F = D                            # s = 1
        D4 = D.reshape(K, d, d, K)       # [B', i' ; i, B]
        for (Lb, Rb, ks) in chain:
            kn = Lb.shape[1]
            L3 = Lb.reshape(ks, d, kn)                       # [(a, i), alpha]
            Y = np.einsum("Bjib,aiz->aBjzb", D4, L3, optimize=True)        # V_s V_R V , cols (alpha, beta)
            F4 = F.reshape(K, ks, ks, K)                     # [C, a' ; a, B']
            Y = np.einsum("Ceab,abjzy->Cejzy", F4, Y, optimize=True)       # V_R V_s V
            R3 = Rb.reshape(kn, ks, d)                       # [gamma ; (a', i')]
            F = np.einsum("gej,Cejzy->Cgzy", R3, Y, optimize=True).reshape(K * kn, kn * K)
        self.C = F
        self.Ci = np.linalg.inv(F)
        Cm = np.einsum("iaja->ij", (kr(np.eye(K), mu) @ F).reshape(K, K, K, K))
        self.twist = Cm[0, 0]
        self.twist_dev = np.abs(Cm - Cm[0, 0] * np.eye(K)).max() / abs(Cm[0, 0])

    def knot_value(self, word, chunk=None, check_scalar=False):
        k = self.k
        b = max(abs(x) for x in word) + 1
        chunk = chunk or max(1, int(1.5e7 // k ** b))
        w = sum(1 if g > 0 else -1 for g in word)
        Cs = {1: self.C.reshape(k, k, k, k), -1: self.Ci.reshape(k, k, k, k)}
        rows = [0, 1] if check_scalar else [0]
        res = []
        ncol = k ** (b - 1)
        for r0 in rows:
            tot = 0
            for c0 in range(0, ncol, chunk):
                c1 = min(ncol, c0 + chunk)
                X = np.zeros((k,) + (k,) * (b - 1) + (c1 - c0,), dtype=complex)
                Xf = X.reshape(k, ncol, c1 - c0)
                Xf[r0, np.arange(c0, c1), np.arange(c1 - c0)] = 1
                for g in reversed(word[::-1]):  # leftmost letter acts first (as in supergroup_vertex_check)
                    p = abs(g) - 1
                    X = np.moveaxis(np.tensordot(Cs[1 if g > 0 else -1], X, axes=([2, 3], [p, p + 1])),
                                    [0, 1], [p, p + 1])
                for s in range(1, b):
                    X = np.moveaxis(np.tensordot(self.mu, X, axes=([1], [s])), 0, s)
                Y = X.reshape(k, ncol, c1 - c0)[r0]
                tot += np.trace(Y[c0:c1, :])
            res.append(tot / self.twist ** w)
        return res[0], (abs(res[1] - res[0]) if check_scalar else 0.0)


# ----------------------------------------------------------------------------------------------- data
_cache = {}


def data_poly(knot, R):
    if knot not in _cache:
        d, cur = {}, None
        path = os.path.join(DATA, knot + ".txt")
        if os.path.exists(path):
            for line in open(path):
                if line.startswith("R = "):
                    cur = tuple(int(x) for x in line[4:].strip(" []\n").split(","))
                elif line.startswith("H = ") and cur is not None:
                    d[cur] = line[4:].strip()
        _cache[knot] = d
    s = _cache[knot].get(tuple(R))
    if s is None:
        return None
    if isinstance(s, str):
        s = [(i, j, int(c)) for (i, j), c in parse_laurent(s).terms.items()]
        _cache[knot][tuple(R)] = s
    return s


def data_value(knot, R, A, q):
    t = data_poly(knot, R)
    if t is None:
        return None
    return sum(c * A ** i * q ** j for i, j, c in t)


def knot_rows():
    return {row["name"]: row for row in csv.DictReader(open(KNOTINFO), delimiter="|") if row["braid_notation"]}


def braid_word(row):
    w = eval(row["braid_notation"])
    return w[0] if isinstance(w[0], list) else w


def main():
    if sys.argv[1] != "check":
        raise SystemExit(__doc__)
    par = [int(c) for c in sys.argv[2]]
    R = tuple(int(x) for x in sys.argv[3].split(","))
    maxb = int(sys.argv[4])
    names = sys.argv[5:]
    conj = os.environ.get("CONJ") == "1"
    q = complex(os.environ.get("QV", "0.83+0.31j"))
    N, M = par.count(0), par.count(1)
    rows = knot_rows()
    mod = Module(par, R, q, conj)
    print("gl(%d|%d) R=%s dim=%d eig_err=%.1e twist_dev=%.1e" % (N, M, R, mod.k, mod.worst_proj,
                                                                 getattr(mod, "twist_dev", 0)))
    if mod.k == 0:
        return
    names = names or sorted(rows)
    ok = bad = skip = 0
    worst = 0
    for kn in names:
        row = rows.get(kn)
        if not row or int(row["braid_index"]) > maxb:
            skip += 1
            continue
        h = data_value(kn, R, (1 / q) ** (N - M), 1 / q)
        if h is None:
            skip += 1
            continue
        v, _ = mod.knot_value(braid_word(row))
        rel = abs(v - h) / max(1, abs(h))
        worst = max(worst, rel)
        if rel < 1e-7:
            ok += 1
        else:
            bad += 1
            print("MISMATCH", kn, v, h)
    print("agree %d mismatch %d skipped %d worst %.1e" % (ok, bad, skip, worst))


if __name__ == "__main__":
    main()

