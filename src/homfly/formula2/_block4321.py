"""Closed Formula II at R = [4,3,2,1]: the non-diagonalizable block from its closed Gram matrix (block4321_closed.json),
no two-colour input.  At a point (A,q) mod p:
    V   = block basis by the vanishing / support conditions (rows of the three multiplicity-2 tops + the 12 ghosts),
    Psi = the 15 functionals (twists T(1), T(2), T(-2) and 12 colours) on V,
    G   = closed Gram matrix,   M = Psi^-1 G Psi^-T,
and M is inserted into the strengthened engine.  Check: H_{[4,3,2,1]}(Tw_m), m = +-1..+-4, against the exact database.
Data: data/block4321_closed.json."""
import os, json
from . import _strong as F2
from . import _core as f2core
from ._core import I, inv
from ._fastrows import FastPointVec

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "block4321_closed.json")
p = f2core.p

U = (4, 3, 2, 1)

def contents(R): return [j - i for i in range(len(R)) for j in range(R[i])]
def hooks(R):
    Rc = [sum(1 for x in R if x > j) for j in range(R[0])]
    return [R[i] - j + Rc[j] - i - 1 for i in range(len(R)) for j in range(R[i])]
def br(z): return (z - inv(z)) % p
def Dk(A, q, k): return br(A * pow(q, k % (p - 1), p) % p)
def dim(R, A, q):
    d = 1
    for c, h in zip(contents(R), hooks(R)): d = d * Dk(A, q, c) % p * inv(br(pow(q, h, p))) % p
    return d
def qn(q, n): return br(pow(q, n % (p - 1), p)) * inv(br(q)) % p
def PU(A, q):
    return pow(br(q), 8, p) * pow(qn(q, 3), 4, p) % p * pow(qn(q, 5), 2, p) % p * qn(q, 7) % p * \
        Dk(A, q, -5) * Dk(A, q, -4) % p * Dk(A, q, 4) % p * Dk(A, q, 5) % p
_cyc = {}
def cyclo(n):
    if n not in _cyc:
        num = [-1] + [0] * (n - 1) + [1]
        for d in range(1, n):
            if n % d == 0:
                b = cyclo(d); a = num[:]; qd = [0] * (len(a) - len(b) + 1)
                for i in range(len(a) - len(b), -1, -1):
                    c = a[i + len(b) - 1] // b[-1]; qd[i] = c
                    for jj, bj in enumerate(b): a[i + jj] -= c * bj
                num = qd
        _cyc[n] = num
    return _cyc[n]
def cyc_val(q, n): return sum(c * pow(q, i, p) for i, c in enumerate(cyclo(n))) % p

def parse(name):
    return ("T", int(name[1:])) if name[0] == "T" else ("C", tuple(int(c) for c in name[1:]))

class ClosedBlock:
    def __init__(self, fn=DATA):
        J = json.load(open(fn))
        self.funcs = [parse(x) for x in J["functionals"]]
        self.entries = {}
        for name, r in J["entries"].items():
            a, b = name[2:-1].split(",")
            self.entries[(parse(a), parse(b))] = r
    def gred(self, r, A, q):
        tot = 0
        for a, h in r["h"].items():
            c = sum(v * pow(q, int(e) % (p - 1), p) for e, v in h.items()) % p
            tot = (tot + c * pow(A, int(a) % (p - 1), p)) % p
        for n, m in r["content"].items(): tot = tot * pow(cyc_val(q, int(n)), m, p) % p
        for k in r["D"]: tot = tot * Dk(A, q, k) % p
        return tot * PU(A, q) % p
    def gram(self, A, q):
        n = len(self.funcs); Gm = [[0] * n for _ in range(n)]
        for i, fa in enumerate(self.funcs):
            for j, fb in enumerate(self.funcs):
                if j < i: continue
                r = self.entries.get((fa, fb)) or self.entries.get((fb, fa))
                v = self.gred(r, A, q)
                for fn in (fa, fb):
                    if fn[0] == "C": v = v * inv(dim(fn[1], A, q)) % p
                Gm[i][j] = Gm[j][i] = v
        return Gm

def block_basis(st):
    good, bad = st.top_rows_sp(U)
    fpv = FastPointVec(st.A, st.q, NB=15, p=p)
    seeds = []
    for t in bad:
        B = list(I.support(t)); nds = [tuple(nu) for nu in f2core._NODES_ORIG(t, 15)]
        Ma = [[fpv.psi(x, nu) for x in B] + [(-fpv.psi(t, nu)) % p] for nu in nds]
        v = [v for v in F2._null_np(Ma, len(B) + 1) if v[-1] % p][0]; c0 = (-inv(v[-1])) % p
        e = {t: 1}
        for x, a in zip(B, v[:-1]):
            if a * c0 % p: e[x] = a * c0 % p
        seeds.append(e)
    Gs, _, _ = f2core.ghost_space(st.core, U, smax=13)
    return seeds + [g[0] for g in Gs]

def matinv(M):
    n = len(M); A = [[x % p for x in row] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        pr = next(i for i in range(c, n) if A[i][c])
        A[c], A[pr] = A[pr], A[c]; iv = inv(A[c][c]); A[c] = [x * iv % p for x in A[c]]
        for i in range(n):
            if i != c and A[i][c]:
                f = A[i][c]; A[i] = [(x - f * y) % p for x, y in zip(A[i], A[c])]
    return [row[n:] for row in A]

def closed_M(cb, st, V):
    core = st.core
    Psi = [[core.phi(v, fn[1]) for v in V] if fn[0] == "T" else [core.Phi(v, fn[1]) for v in V] for fn in cb.funcs]
    Pi = matinv(Psi); Gm = cb.gram(st.A, st.q); n = len(V)
    T = [[sum(Pi[i][k] * Gm[k][j] for k in range(n)) % p for j in range(n)] for i in range(n)]
    return [[sum(T[i][k] * Pi[j][k] for k in range(n)) % p for j in range(n)] for i in range(n)]

