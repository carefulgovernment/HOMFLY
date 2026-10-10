"""U_q gl(N|M) vertex model, exact mod p, with recursive fusion to any covariant module V_lambda;
towers of H_R on the lines A = q^{-m} for the families R = [r, mu] (docs/notes/nonrect_lines.tex).

V = C^{N|M} with Perk-Schultz T (Hecke: (T-q)(T+1/q)=0).  V_lambda is built along a chain of
Young diagrams lambda^(1) = [1] -> ... -> lambda, V_{lambda+b} = eigenspace of the monodromy
Omega = R_{V,X} R_{X,V} on X (x) V with eigenvalue q^{2 c(b)}.  Braidings of composite objects
are obtained recursively from T.  The (1,1)-tangle value of a braid closure is the scalar of the
partial (pivotal) trace over strands 2..n divided by twist^writhe.
"""
import numpy as np

P = 8388593          # < 2^23: float64 BLAS products of 64-term sums stay exact


def inv(x, p=P):
    return pow(int(x) % p, p - 2, p)


def mm(A, B, p=P):
    """exact matrix product mod p (float64 BLAS on blocks of <= 64 contraction terms)"""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    k = A.shape[1]
    out = np.zeros((A.shape[0], B.shape[1]))
    for s in range(0, k, 64):
        out = np.mod(out + A[:, s:s + 64] @ B[s:s + 64], p)
    return out


def kron(A, B, p=P):
    return np.mod(np.kron(A, B), p)


def rref(M, p=P):
    M = np.array(M, dtype=np.int64) % p
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        i = r + nz[0]
        M[[r, i]] = M[[i, r]]
        M[r] = M[r] * inv(M[r, c]) % p
        for j in range(rows):
            if j != r and M[j, c]:
                M[j] = (M[j] - M[j, c] * M[r]) % p
        piv.append(c)
        r += 1
    return M[:r], piv


def nullspace(M, p=P):
    R, piv = rref(M, p)
    cols = M.shape[1]
    free = [c for c in range(cols) if c not in piv]
    out = []
    for f in free:
        v = np.zeros(cols, dtype=np.int64)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = (-R[i, f]) % p
        out.append(v)
    return np.array(out).T if out else np.zeros((cols, 0), dtype=np.int64)


def rank_factor(Pm, p=P):
    """Pm = C F with C = pivot columns, F = rref rows (F C = 1 for a projector)"""
    F, piv = rref(Pm, p)
    C = np.array(Pm, dtype=np.int64)[:, piv] % p
    return C, F


class Model:
    def __init__(self, par, q, p=P):
        self.par, self.q, self.p = par, q % p, p
        n = len(par)
        self.n = n
        qi = inv(q, p)
        T = np.zeros((n * n, n * n), dtype=np.int64)
        for i in range(n):
            T[i * n + i, i * n + i] = q if par[i] == 0 else (-qi) % p
            for j in range(i + 1, n):
                s = (-1) ** (par[i] * par[j])
                T[j * n + i, i * n + j] = s % p
                T[i * n + j, j * n + i] = s % p
                T[j * n + i, j * n + i] = (q - qi) % p
        self.T = T % p
        self.objs = {}          # name -> dict(d, mu, kind, ...)
        self.Rmemo = {}
        self.objs['V'] = dict(d=n, kind='V')
        self.objs['V']['mu'] = self._pivot()

    # ---- pivotal element of V, normalised by kappa_+ kappa_- = 1 ----
    def _pivot(self):
        p, n, T = self.p, self.n, self.T
        Ti = np.array(sympy_inv(T, p))
        rows = []
        for which, M in enumerate((T, Ti)):
            M4 = M.reshape(n, n, n, n)
            for i in range(n):
                for ip in range(n):
                    row = [int(M4[i, j, ip, j]) for j in range(n)] + [0, 0]
                    if i == ip:
                        row[n + which] = p - 1
                    rows.append(row)
        ns = nullspace(np.array(rows, dtype=np.int64), p)
        assert ns.shape[1] == 1, ns.shape
        x = ns[:, 0]
        mu, ka, kb = x[:n], int(x[n]), int(x[n + 1])
        # scale s: (s ka)(s kb) = 1  ->  s^2 = 1/(ka kb); find s = +-q^k
        target = inv(ka * kb % p, p)
        for k in range(-60, 61):
            s = pow(self.q, k % (p - 1), p)
            if s * s % p == target:
                break
        else:
            raise ValueError("no q-power normalisation")
        self.kappa = s * ka % p
        return np.diag(mu * s % p)

    # ---- objects ----
    def d(self, X):
        return self.objs[X]['d']

    def R(self, X, Y):
        """braiding X (x) Y -> Y (x) X"""
        key = (X, Y)
        if key in self.Rmemo:
            return self.Rmemo[key]
        p = self.p
        oX, oY = self.objs[X], self.objs[Y]
        if oX['kind'] == 'V' and oY['kind'] == 'V':
            out = self.T
        elif oX['kind'] == 'sub':
            A, B = oX['parts']
            dA, dB, dY = self.d(A), self.d(B), self.d(Y)
            # R_{A(x)B, Y} = (R_{A,Y} (x) 1_B)(1_A (x) R_{B,Y})
            RAB = mm(kron(self.R(A, Y), np.eye(dB, dtype=np.int64)), kron(np.eye(dA, dtype=np.int64), self.R(B, Y)))
            out = mm(mm(kron(np.eye(dY, dtype=np.int64), oX['pi']), RAB), kron(oX['iota'], np.eye(dY, dtype=np.int64)))
        else:
            A, B = oY['parts']
            dA, dB, dX = self.d(A), self.d(B), self.d(X)
            # R_{X, A(x)B} = (1_A (x) R_{X,B})(R_{X,A} (x) 1_B)
            RXAB = mm(kron(np.eye(dA, dtype=np.int64), self.R(X, B)), kron(self.R(X, A), np.eye(dB, dtype=np.int64)))
            out = mm(mm(kron(oY['pi'], np.eye(dX, dtype=np.int64)), RXAB), kron(np.eye(dX, dtype=np.int64), oY['iota']))
        out = np.asarray(out, dtype=np.int64) % p
        self.Rmemo[key] = out
        return out

    def add_box(self, X, name, c, others):
        """sub-object of X (x) V on which Omega = R_{V,X} R_{X,V} = q^{2c}; others: contents of other addable boxes"""
        p, q = self.p, self.q
        dX = self.d(X)
        Om = mm(self.R('V', X), self.R(X, 'V'))
        D = dX * self.n
        I = np.eye(D, dtype=np.int64)
        lam = pow(q, (2 * c) % (p - 1), p)
        Pm = I.copy()
        for c2 in others:
            l2 = pow(q, (2 * c2) % (p - 1), p)
            Pm = mm(Pm, (Om - l2 * I) % p) * inv(lam - l2, p) % p
        Pm = np.asarray(Pm, dtype=np.int64) % p
        assert np.all(mm(Pm, Pm) == Pm), "not a projector"
        C, F = rank_factor(Pm, p)
        if C.shape[1] == 0:
            return None
        muXV = kron(self.objs[X]['mu'], self.objs['V']['mu'])
        self.objs[name] = dict(d=C.shape[1], kind='sub', parts=(X, 'V'), iota=C, pi=F,
                               mu=np.asarray(mm(mm(F, muXV), C), dtype=np.int64) % p)
        return name

    def twist(self, X):
        d = self.d(X)
        M = mm(kron(np.eye(d, dtype=np.int64), self.objs[X]['mu']), self.R(X, X))
        C = np.einsum('iaja->ij', M.reshape(d, d, d, d)) % self.p
        assert np.all(C == C[0, 0] * np.eye(d, dtype=np.int64) % self.p), "twist not scalar"
        return int(C[0, 0])


def sympy_inv(M, p=P):
    n = M.shape[0]
    A = np.concatenate([np.array(M, dtype=np.int64) % p, np.eye(n, dtype=np.int64)], axis=1)
    R, piv = rref(A, p)
    assert piv[:n] == list(range(n))
    return R[:, n:]


def contents_addable(lam):
    lam = list(lam)
    out = []
    for i in range(len(lam) + 1):
        prev = lam[i - 1] if i > 0 else None
        cur = lam[i] if i < len(lam) else 0
        if prev is None or cur < prev:
            out.append((i, cur, cur - i))      # row i, new column cur, content
    return out


def build(model, lam):
    """build V_lam along the chain that fills rows top to bottom; returns object name (or None if zero)"""
    chain = []
    for i, r in enumerate(lam):
        for j in range(r):
            chain.append((i, j))
    cur = 'V'
    shape = [1]
    for (i, j) in chain[1:]:
        add = contents_addable(shape)
        c = j - i
        others = [cc for (_, _, cc) in add if cc != c]
        if i < len(shape):
            shape[i] += 1
        else:
            shape.append(1)
        name = 'L' + ','.join(map(str, shape))
        if name not in model.objs:
            if model.add_box(cur, name, c, others) is None:
                return None
        cur = name
    return cur


def knot_value(model, X, word, check=False, chunk=512):
    p = model.p
    d = model.d(X)
    n = max(abs(g) for g in word) + 1
    R = model.R(X, X).astype(np.float64)
    Ri = np.array(sympy_inv(model.R(X, X), p), dtype=np.float64)
    mu = model.objs[X]['mu'].astype(np.int64)
    rest = d ** (n - 1)
    Mu = np.ones((1, 1), dtype=np.int64)
    for _ in range(n - 1):
        Mu = np.kron(Mu, mu) % p

    def run(i0):
        out = np.zeros(d, dtype=np.int64)
        for s0 in range(0, rest, chunk):
            cols = np.arange(s0, min(rest, s0 + chunk))
            V = np.zeros((d, rest, len(cols)))
            V[i0, cols, np.arange(len(cols))] = 1
            V = V.reshape(d ** n, len(cols))
            for g in word:
                k = abs(g) - 1
                M = R if g > 0 else Ri
                T4 = V.reshape(d ** k, d * d, d ** (n - k - 2) * len(cols))
                T4 = np.tensordot(M, T4, axes=([1], [1]))        # (dd, d^k, rest')
                V = np.mod(np.transpose(T4, (1, 0, 2)), p).reshape(d ** n, len(cols))
            B = V.reshape(d, rest, len(cols)).astype(np.int64)     # [i_out, b, a in cols]
            Mc = Mu[cols, :]                                     # [a, b]
            for j in range(d):
                out[j] = (out[j] + ((Mc * B[j].T) % p).sum()) % p
        return [int(x) for x in out]
    c0 = run(0)
    w = sum(1 if g > 0 else -1 for g in word)
    tw = model.twist(X)
    val = c0[0] * pow(inv(tw, p), w % (p - 1), p) % p
    if check:
        c1 = run(1)
        assert c0[1:] == [0] * (d - 1) and c1[1] == c0[0], (c0[:3], c1[:3])
    return val


def weights(model, X):
    o = model.objs[X]
    if 'wt' in o:
        return o['wt']
    if o['kind'] == 'V':
        o['wt'] = [tuple(1 if i == j else 0 for j in range(model.n)) for i in range(model.n)]
        return o['wt']
    A, B = o['parts']
    wa, wb = weights(model, A), weights(model, B)
    dB = model.d(B)
    out = []
    for j in range(o['d']):
        nz = np.nonzero(o['iota'][:, j])[0]
        ws = {tuple(x + y for x, y in zip(wa[i // dB], wb[i % dB])) for i in nz}
        assert len(ws) == 1, "not a weight vector"
        out.append(ws.pop())
    o['wt'] = out
    return out


def knot_value_blocks(model, X, word):
    """(1,1)-tangle value using total-weight blocks of X^{(x)n} (diagonal pivotal element required)"""
    import scipy.sparse as sp
    p = model.p
    d = model.d(X)
    n = max(abs(g) for g in word) + 1
    Rm = model.R(X, X)
    Rim = sympy_inv(Rm, p)
    mud = np.diag(model.objs[X]['mu']).astype(np.int64)
    assert np.count_nonzero(model.objs[X]['mu'] - np.diag(mud)) == 0
    wt = np.array(weights(model, X), dtype=np.int64)
    base = 4 * n * (int(np.abs(wt).max()) + 1) + 1
    code1 = np.zeros(d, dtype=np.int64)
    for c in range(wt.shape[1]):
        code1 = code1 * base + wt[:, c]
    tot = np.zeros(1, dtype=np.int64)
    for _ in range(n):
        tot = (tot[:, None] + code1[None, :]).reshape(-1)
    gens = {}
    for g in set(word):
        k = abs(g) - 1
        M = sp.csr_matrix((Rm if g > 0 else Rim).astype(np.float64))
        gens[g] = sp.kron(sp.kron(sp.identity(d ** k, format='csr'), M), sp.identity(d ** (n - k - 2), format='csr'), format='csr')
    rest = d ** (n - 1)
    Mud = np.ones(1, dtype=np.int64)
    for _ in range(n - 1):
        Mud = np.kron(Mud, mud) % p
    first = tot[:rest]
    res = 0
    for W in np.unique(first):
        rows = np.nonzero(tot == W)[0]
        cols = np.nonzero(first == W)[0]
        pos = np.full(d ** n, -1, dtype=np.int64)
        pos[rows] = np.arange(len(rows))
        ci = pos[cols]
        V = np.zeros((len(rows), len(cols)))
        V[ci, np.arange(len(cols))] = 1
        for g in word:
            G = gens[g][rows][:, rows]
            V = np.mod(G @ V, p)
        diag = V[ci, np.arange(len(cols))].astype(np.int64)
        res = (res + int(((diag * Mud[cols]) % p).sum())) % p
    w = sum(1 if g > 0 else -1 for g in word)
    tw = model.twist(X)
    return res * pow(inv(tw, p), w % (p - 1), p) % p


# ---------------------------------------------------------------------------
# towers: P^mu_m(q, lambda) = H_[r,mu](A = q'^{-m}, q' = 1/q) at lambda = q^r
# ---------------------------------------------------------------------------
def bm(s,p):
    """Berlekamp-Massey: minimal connection polynomial C (C[0]=1) for sequence s mod p"""
    C=[1];B=[1];L=0;m=1;b=1
    for n in range(len(s)):
        d=s[n]
        for i in range(1,L+1): d=(d+C[i]*s[n-i])%p
        if d==0: m+=1; continue
        T=C[:]; coef=d*pow(b,p-2,p)%p
        C=C+[0]*(len(B)+m-len(C))
        for i in range(len(B)): C[i+m]=(C[i+m]-coef*B[i])%p
        if 2*L<=n: L=n+1-L;B=T;b=d;m=1
        else: m+=1
    return C[:L+1],L


def lambda_exponents(s, q, p=P, K=80):
    """s_r (r = r0, r0+1, ...) = sum_k c_k q^{k r}: the exponents k (None if the sequence is too short)"""
    C, L = bm(s, p)
    if 2 * L > len(s) - 2:
        return None, L
    ks = []
    for k in range(-K, K + 1):
        z = pow(q, k % (p - 1), p)
        v = 0
        for c in C:
            v = (v * z + c) % p
        if v == 0:
            ks.append(k)
    return ks, L


def braid_word(knot):
    import csv, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "homfly", "knots", "data", "knotinfo_upto12.csv")
    for row in csv.DictReader(open(path), delimiter="|"):
        if row["name"] == knot:
            w = eval(row["braid_notation"])
            return w[0] if isinstance(w[0], list) else w
    raise KeyError(knot)


def tower(knot, m, mu, rmax, q):
    """lambda-exponents of H_[r,mu] on A = q^{-m}, r = mu_1 .. rmax, via U_q gl(1|m+1)"""
    M = Model([0] + [1] * (m + 1), q)
    w = braid_word(knot)
    s = []
    r0 = max(mu[0] if mu else 1, 1)
    for r in range(r0, rmax + 1):
        X = build(M, tuple([r] + list(mu)))
        s.append(knot_value_blocks(M, X, w))
    ks, L = lambda_exponents(s, q)
    return dict(knot=knot, m=m, mu=list(mu), r=[r0, rmax], dim=M.d(X), L=L, exps=ks)


if __name__ == "__main__":
    import json, random, sys
    if len(sys.argv) < 5:
        print("usage: super_vertex.py KNOT m MU rmax   (MU: comma separated, '' for the symmetric family)")
        print("  e.g. super_vertex.py 11n_34 1 1 30   ->  exps -12..12: e_1 = 6 for the hook family [r,1]")
        sys.exit(1)
    mu = tuple(int(x) for x in sys.argv[3].split(",") if x)
    q = random.Random(11).randrange(2, P - 1)
    print(json.dumps(tower(sys.argv[1], int(sys.argv[2]), mu, int(sys.argv[4]), q)))
