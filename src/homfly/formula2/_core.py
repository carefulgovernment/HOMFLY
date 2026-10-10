"""Common core for the Formula II strengthening analysis (mod p, one point (A,q) = seed 3 of legs.py).

    labels / channels, code rows (interpolation.py), naive rows of the [4,3,1]-tops, kernel k431,
    pivots W_c(R) = g_c Phi_c[R],   Phi_c[leg]:  leg 'tw' -> phi_c(1),  leg 'S...' -> sum_X E_cX psi_X(S),
    reconstruction of the true legs c(R) v from the stored residuals of legs_<R>_3.pkl.
"""
import os, pickle, random
from ..methods import interpolation as I
from ..algebra.fields import GF
from ..reps.partitions import P
from ._linalg import nullspace

p = 67108859
F = GF(p)
_NODES_ORIG = I.nodes


def inv(x): return pow(int(x) % p, -1, p)


def nm(c): return str(list(c[1])) if c[0] == "s" else "{%s,%s}" % (list(c[1]), list(c[2]))


def lev(c): return sum(c[1])


def parts(n, mx=None):
    if mx is None: mx = n
    if n == 0: yield (); return
    for k in range(min(n, mx), 0, -1):
        for rest in parts(n - k, k): yield (k,) + rest


def contains(big, small):
    return len(small) <= len(big) and all(s <= b for s, b in zip(small, big))


def leg_colour(key):
    """'S532' -> (5,3,2); 'S10' -> (10,) ; keys with a part >= 10 are written 'S10_1' etc. by newer scripts"""
    s = key[1:]
    if "_" in s: return tuple(int(t) for t in s.split("_"))
    if s == "10": return (10,)
    return tuple(int(ch) for ch in s)


def leg_key(S):
    return "S" + ("_".join(map(str, S)) if max(S) >= 10 and S != (10,) else "".join(map(str, S)))


FAST = os.environ.get("F2_FAST", "1") == "1"      # integer/numpy point evaluation (identical rows, 7-16x faster)


class Core:
    def __init__(self, seed=3, NB=None, A=None, q=None):
        rnd = random.Random(seed)
        self.A = rnd.randrange(2, p - 1); self.q = rnd.randrange(2, p - 1)
        if A is not None: self.A, self.q = A % p, q % p
        self.NB = NB
        if NB is not None:
            I.nodes = lambda c, extra_size=None: _NODES_ORIG(c, NB)
        else:
            I.nodes = _NODES_ORIG
        self.pt = I.Point(F, F(self.A), F(self.q))
        self.fp = None
        if FAST:
            from ._fastrows import FastPointVec
            self.fp = FastPointVec(self.A, self.q, NB=(NB or 13), p=p)
        self._rows = {}; self._psi = {}; self._Lam = {}; self._d = {}
        self.k = {}          # kernel vectors: name -> {label: coeff}

    # --- basic point data
    def Lam(self, x):
        if x not in self._Lam: self._Lam[x] = (self.fp.Lam(x) if self.fp else int(self.pt.Lam(x))) % p
        return self._Lam[x]

    def d(self, c):
        if c not in self._d: self._d[c] = (self.fp.d(c) if self.fp else int(self.pt.d(c))) % p
        return self._d[c]

    def psi(self, x, S):
        key = (x, S)
        if key not in self._psi: self._psi[key] = (self.fp.psi(x, S) if self.fp else int(self.pt.psi(x, P(S)))) % p
        return self._psi[key]

    def code_row(self, c):
        if c not in self._rows:
            src = self.fp.E_row(c) if self.fp else self.pt.E_row(c)
            self._rows[c] = {x: int(v) % p for x, v in src.items()}
        return self._rows[c]

    # --- functionals of a vector v = {label: coeff} (vacuum coefficient = -sum, so that v(empty colour) = 0)
    def phi(self, v, m=1):
        return sum(a * (pow(self.Lam(x), m % (p - 1), p) - 1) for x, a in v.items()) % p

    def Phi(self, v, S):
        return sum(a * self.psi(x, S) for x, a in v.items()) % p

    def pair_leg(self, v, leg):
        return self.phi(v, 1) if leg == "tw" else self.Phi(v, leg_colour(leg))

    # --- kernels
    def kernel(self, U, c=None, norm_label=None):
        """kernel vector(s) of the vanishing system of top channel c of U (default: the single [U,U])"""
        U = P(tuple(U)); c = c or I.single(U); B = I.support(c)
        ker = nullspace([[self.psi(x, tuple(nu)) for x in B] for nu in I.nodes(c)], len(B))
        out = []
        for kv in ker:
            k = {x: a for x, a in zip(B, kv) if a}
            nl = norm_label or I.single(P((1,)))
            if nl in k:
                s = inv(k[nl]); k = {x: a * s % p for x, a in k.items()}
            out.append(k)
        return out


def _boxes(l): return [(i + 1, j + 1) for i, r in enumerate(l) for j in range(r)]


def _conj(l):
    if not l: return ()
    return tuple(sum(1 for x in l if x > j) for j in range(l[0]))


def _kappa(l): return sum(j - i for (i, j) in _boxes(l))


def _idx_in(U, b):
    i, j = b; Uc = _conj(U); return (U[i - 1] - i) + (j - Uc[j - 1])


def _union(a, b):
    import itertools
    return tuple(max(x, y) for x, y in itertools.zip_longest(a, b, fillvalue=0))


def _inter(a, b): return tuple(x for x in (min(x, y) for x, y in zip(a, b)) if x)


def phi_pred_d1(A, q, Z, Zp):
    """closed phi_c(1) of a distance-one pair c = {W+a, W+b}  (HYPOTHESES.tex, eq. (phipair))"""
    D = lambda k: (A * pow(q, k % (p - 1), p) - inv(A * pow(q, k % (p - 1), p))) % p
    qn = lambda n: (pow(q, n % (p - 1), p) - pow(q, (-n) % (p - 1), p)) * inv((q - inv(q)) % p) % p
    eps = (q - inv(q)) % p
    W = _inter(Z, Zp); U = _union(Z, Zp)
    a = [b for b in _boxes(Z) if b not in _boxes(W)][0]; b = [b for b in _boxes(Zp) if b not in _boxes(W)][0]
    val = (-inv(2)) * pow(A, sum(U), p) % p * pow(q, _kappa(U) % (p - 1), p) % p * eps * eps % p * pow(qn((b[1] - b[0]) - (a[1] - a[0])), 2, p) % p
    for beta in _boxes(W): val = val * D(_idx_in(U, beta)) % p
    return val


def naive_431_rows(core, rows):
    """modify rows (dict c -> row) in place: naive rows of the five [4,3,1]-tops; returns t_c and k431"""
    U = P((4, 3, 1))
    k = core.kernel((4, 3, 1))[0]
    phit1 = core.phi(k, 1)
    ZERO4 = I.pair(P((1, 1, 1)), P((3,)))
    ts = {}
    for c in [c for c in I.channels_of(U) if I.rmin(c) == (U,)]:
        r = dict(core.code_row(c))
        if c[0] == "s": t = 0
        elif c == I.pair(P((3, 3)), P((4, 1, 1))): t = (-r.get(ZERO4, 0)) * inv(k[ZERO4]) % p
        else:
            t = (phi_pred_d1(core.A, core.q, c[1], c[2]) - core.phi(r, 1)) * inv(phit1) % p
        for x, a in k.items(): r[x] = (r.get(x, 0) + t * a) % p
        rows[c] = {x: a for x, a in r.items() if a}
        ts[c] = t
    return ts, k


class RepData:
    """data of one representation R: channels, rows (code or modified), pivots"""
    def __init__(self, core, R, rows_override=None):
        self.core = core; self.R = tuple(R)
        self.chans = sorted(I.channels_of(P(self.R)), key=lambda c: (lev(c), nm(c)))
        self.rows = {c: dict(core.code_row(c)) for c in self.chans}
        if rows_override:
            for c, r in rows_override.items():
                if c in self.rows: self.rows[c] = dict(r)
        self.update()

    def update(self):
        core = self.core; R = self.R
        self.FR = {c: core.d(c) * core.Phi(self.rows[c], R) % p for c in self.chans}
        self.ph1 = {c: core.phi(self.rows[c], 1) for c in self.chans}
        self.phm = {c: core.phi(self.rows[c], -1) for c in self.chans}
        self.W = {c: self.FR[c] * inv(self.ph1[c] * self.phm[c]) % p for c in self.chans}

    def formula_leg(self, leg, labels):
        """sum_c E_cX W_c Phi_c[leg] for X in labels"""
        pc = {c: self.core.pair_leg(self.rows[c], leg) for c in self.chans}
        out = {}
        for X in labels:
            out[X] = sum(self.rows[c].get(X, 0) * self.W[c] % p * pc[c] for c in self.chans) % p
        return out


def true_legs(core, R, legsfile, labels=None):
    """true c(R) v for every stored leg, on labels (default: channels of R), from residuals w.r.t. code rows"""
    d = pickle.load(open(legsfile, "rb"))
    assert d["A"] == core.A and d["q"] == core.q
    rd = RepData(core, R)
    labels = labels or rd.chans
    out = {}
    for leg, lv in d["legs"].items():
        labs = [X for X in labels if nm(X) in lv]          # levels below the LMIN of that leg are not stored
        f = rd.formula_leg(leg, labs)
        out[leg] = {X: (lv[nm(X)] + f[X]) % p for X in labs}
    return out, rd


# ---------------------------------------------------------------------------------------------------------------
# residual legs w.r.t. "naive rows + universal 48th term"  and ghost spaces I_U(U)
# ---------------------------------------------------------------------------------------------------------------
def gt431(core):
    """universal coefficient of the 48th term: lambda_431(R) = gt * psit(R)  (from gt431.pkl, same point)"""
    d = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt431.pkl"), "rb"))
    assert d["A"] == core.A and d["q"] == core.q
    return d["gt"]


def residual_legs(R, legsfile=None, NB=None):
    """{leg: {label: residual}} of the true legs of R w.r.t. naive Formula II + gt*psit(R) k k^T; also returns core, k"""
    R = tuple(R); legsfile = legsfile or "legs_%s_3.pkl" % "".join(map(str, R))
    core = Core(3, NB=NB or sum(R) + 3)
    tl, rdc = true_legs(core, R, legsfile)
    rows_n = {}
    ts, k = naive_431_rows(core, rows_n)
    rdn = RepData(core, R, rows_override=rows_n)
    lamR = gt431(core) * core.Phi(k, R) % p
    res = {}
    for leg, tv in tl.items():
        f = rdn.formula_leg(leg, list(tv)); kl = core.pair_leg(k, leg)
        res[leg] = {X: (v - f[X] - lamR * k.get(X, 0) % p * kl) % p for X, v in tv.items()}
    return res, core, k, rdn


def ideal_space(core, U, labels=None, smax=None):
    """basis of I_U(labels) = {f = sum_Y u_Y psi_Y + u0 : f(S) = 0 for S ⊉ U, |S| <= smax}; vectors (dict label->u, u0)"""
    U = tuple(U)
    labels = labels or sorted(I.channels_of(P(U)), key=lambda c: (lev(c), nm(c)))
    smax = smax or sum(U) + 3
    Ss = [S for n in range(1, smax + 1) for S in parts(n) if not contains(S, U)]
    ann = nullspace([[core.psi(Y, S) for Y in labels] + [1] for S in Ss], len(labels) + 1)
    return [({Y: u[i] for i, Y in enumerate(labels) if u[i]}, u[-1]) for u in ann], labels


def ghost_space(core, U, smax=None):
    """ghost part of I_U(U): elements with zero coefficients on the rows of the top channels of U (code rows)"""
    ann, labels = ideal_space(core, U, smax=smax)
    tops = [c for c in labels if I.rmin(c) == (P(tuple(U)),)]
    rows = {c: core.code_row(c) for c in labels}
    def expand(u):
        beta = {}
        for X in sorted(labels, key=lambda c: -lev(c)):
            beta[X] = (u.get(X, 0) - sum(beta[c] * rows[c].get(X, 0) for c in beta if c != X)) % p
        return beta
    E = [expand(u) for u, _ in ann]
    gk = nullspace([[e[c] for e in E] for c in tops], len(ann))
    out = []
    for g in gk:
        v = {}
        for j, (u, u0) in enumerate(ann):
            for Y, a in u.items(): v[Y] = (v.get(Y, 0) + g[j] * a) % p
        u0 = sum(g[j] * ann[j][1] for j in range(len(ann))) % p
        out.append(({Y: a for Y, a in v.items() if a}, u0))
    return out, tops, labels


def fval(core, f, leg):
    """value of f = (u, u0) on a leg: colour S -> sum u psi(S) + u0 ; 'tw' -> sum u (Lam - 1) + u0"""
    u, u0 = f
    return (core.pair_leg(u, leg) + u0) % p


# ---------------------------------------------------------------------------------------------------------------
# general kernel diagrams: naive rows by the zero rule, one cube term per kernel direction
# ---------------------------------------------------------------------------------------------------------------
PAIR_ZERO = None


def naive_rows_rule(core, U, rows=None):
    """kernel diagram U (common 1-dim kernel k_U of its ambiguous top systems).  Rule for the naive rows:
       pair top  -> entry at the pair {[1,1,1],[3]} vanishes;   single top -> no pair entries (entry at {[1,1],[2]} vanishes).
    Returns {top: naive row} for the ambiguous tops, the shifts t relative to the code rows, and k_U (k_[1] = 1)."""
    U = P(tuple(U))
    k = core.kernel(U)[0]
    Z3 = I.pair(P((1, 1, 1)), P((3,))); Z2 = I.pair(P((1, 1)), P((2,)))
    out = {}; ts = {}
    for c in [c for c in I.channels_of(U) if I.rmin(c) == (U,)]:
        B = I.support(c)
        ker = nullspace([[core.psi(x, tuple(nu)) for x in B] for nu in I.nodes(c)], len(B))
        if not ker: continue
        r = dict(core.code_row(c))
        X0 = Z2 if c[0] == "s" else Z3
        t = (-r.get(X0, 0)) * inv(k[X0]) % p
        for x, a in k.items(): r[x] = (r.get(x, 0) + t * a) % p
        out[c] = {x: a for x, a in r.items() if a}; ts[c] = t
    if rows is not None: rows.update(out)
    return out, ts, k


def cube_coeff(core, U, R_legsfile=None):
    """gamma_U from the twist leg at R = U, with naive rows of all kernel diagrams contained in U and their cubes known"""
    raise NotImplementedError
