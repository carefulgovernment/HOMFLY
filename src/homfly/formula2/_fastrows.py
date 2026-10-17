"""Fast integer (mod p) re-implementation of the point evaluation of interpolation.Point:
psi_X(nu), d_X, Lambda_X and the rows E_c (vanishing conditions), with numpy elimination.
Same combinatorics (support, nodes, schur_in_power_sums, skew) as homfly.methods.interpolation."""
import numpy as np
from ..methods import interpolation as I
from ..reps.partitions import P, partitions, conjugate, kappa

I64 = np.int64


class FastPoint:
    def __init__(self, A, q, p=67108859):
        self.p = p; self.A = A % p; self.q = q % p
        self.q2 = self.q * self.q % p; self.A2 = self.A * self.A % p
        self._pw = {}; self._s = {}; self._chi = {}; self._d = {}; self._E = {}; self._psi = {}
        self._coef = {}

    def inv(self, x): return pow(x % self.p, self.p - 2, self.p)

    def power(self, nu, k):
        key = (nu, k)
        if key not in self._pw:
            p = self.p
            q2k = pow(self.q2, k % (p - 1), p)
            den = (1 - self.inv(q2k)) % p
            if den == 0: raise ZeroDivisionError("q^{2k} = 1")
            v = (1 - pow(self.A2, (-k) % (p - 1), p)) * self.inv(den) % p
            C = 0
            for i, r in enumerate(nu):
                for j in range(r):
                    C += pow(q2k, (j - i) % (p - 1), p)
            self._pw[key] = (v + (q2k - 1) * C) % p
        return self._pw[key]

    def coefs(self, lam):
        if lam not in self._coef:
            p = self.p
            self._coef[lam] = [(rho, c.numerator * pow(c.denominator, p - 2, p) % p) for rho, c in I.schur_in_power_sums(lam)]
        return self._coef[lam]

    def schur(self, nu, lam, sign):
        key = (nu, lam, sign)
        if key not in self._s:
            if not lam: self._s[key] = 1
            else:
                p = self.p; tot = 0
                for rho, c in self.coefs(lam):
                    t = c
                    for r in rho: t = t * self.power(nu, sign * r) % p
                    tot += t
                self._s[key] = tot % p
        return self._s[key]

    def skew_schur(self, nu, alpha, gamma, sign):
        return sum(c * self.schur(nu, lam, sign) for lam, c in I.skew(alpha, gamma).items()) % self.p

    def chi(self, nu, alpha, beta):
        key = (nu, alpha, beta)
        if key not in self._chi:
            tot = 0
            for k in range(0, min(sum(alpha), sum(beta)) + 1):
                for g in partitions(k):
                    if not I.contains(alpha, g) or not I.contains(beta, conjugate(g)): continue
                    t = self.skew_schur(nu, alpha, g, 1) * self.skew_schur(nu, beta, conjugate(g), -1)
                    tot += t if k % 2 == 0 else -t
            self._chi[key] = tot % self.p
        return self._chi[key]

    def d(self, c):
        if c not in self._d:
            self._d[c] = sum(self.chi((), a, b) for a, b in I.composites(c)) % self.p
        return self._d[c]

    def psi(self, c, nu):
        nu = P(tuple(nu)); key = (c, nu)
        if key not in self._psi:
            d = self.d(c)
            if d == 0: raise ZeroDivisionError("vanishing composite dimension")
            self._psi[key] = (sum(self.chi(nu, a, b) for a, b in I.composites(c)) * self.inv(d) - 1) % self.p
        return self._psi[key]

    def Lam(self, c):
        Z, Zp = c[1], c[2]
        return pow(self.A2, sum(Z), self.p) * pow(self.q2, (kappa(Z) + kappa(Zp)) % (self.p - 1), self.p) % self.p

    def E_row(self, c):
        """{X: E_cX}: E_cc = 1, sum_{X in {c} u B(c)} E_cX psi_X(nu) = 0 for nu in nodes(c); free unknowns -> 0"""
        if c in self._E: return self._E[c]
        p = self.p; B = list(I.support(c)); n = len(B); nodes = list(I.nodes(c))
        if n == 0:
            self._E[c] = {c: 1}; return self._E[c]
        rows = np.array([[self.psi(x, nu) for x in B] + [(-self.psi(c, nu)) % p] for nu in nodes], dtype=I64)
        # Gauss-Jordan mod p over all node rows
        M = rows % p; m = M.shape[0]; piv_cols = []; r = 0
        for col in range(n):
            if r >= m: break
            nz = np.flatnonzero(M[r:, col])
            if len(nz) == 0: continue
            pr = r + int(nz[0])
            if pr != r: M[[r, pr]] = M[[pr, r]]
            M[r] = M[r] * pow(int(M[r, col]), p - 2, p) % p
            f = M[:, col].copy(); f[r] = 0
            nzr = np.flatnonzero(f)
            if len(nzr): M[nzr] = (M[nzr] - (f[nzr, None] * M[r][None, :]) % p) % p
            piv_cols.append(col); r += 1
        sol = [0] * n
        for i, col in enumerate(piv_cols): sol[col] = int(M[i, n])
        row = {c: 1}
        for x, v in zip(B, sol): row[x] = v
        self._E[c] = row
        self._E[("rank", c)] = (len(piv_cols), n)
        return row


class FastPointVec(FastPoint):
    """psi_X evaluated at once on a universe of partitions (all nu with |nu| <= NB) with numpy; rows from those values."""
    def __init__(self, A, q, NB=13, p=67108859):
        super().__init__(A, q, p)
        self.NB = NB
        self.U = [P(nu) for n in range(0, NB + 1) for nu in partitions(n)]
        self.Uidx = {nu: i for i, nu in enumerate(self.U)}
        self._P = {}; self._sv = {}; self._cv = {}; self._pv = {}

    def Pk(self, k):
        if k not in self._P:
            self._P[k] = np.array([self.power(nu, k) for nu in self.U], dtype=I64)
        return self._P[k]

    def schur_vec(self, lam, sign):
        key = (lam, sign)
        if key not in self._sv:
            p = self.p
            if not lam: v = np.ones(len(self.U), dtype=I64)
            else:
                v = np.zeros(len(self.U), dtype=I64)
                for rho, c in self.coefs(lam):
                    t = np.full(len(self.U), c, dtype=I64)
                    for r in rho: t = t * self.Pk(sign * r) % p
                    v = (v + t) % p
            self._sv[key] = v
        return self._sv[key]

    def chi_vec(self, alpha, beta):
        key = (alpha, beta)
        if key not in self._cv:
            p = self.p; tot = np.zeros(len(self.U), dtype=I64)
            for k in range(0, min(sum(alpha), sum(beta)) + 1):
                for g in partitions(k):
                    if not I.contains(alpha, g) or not I.contains(beta, conjugate(g)): continue
                    a = np.zeros(len(self.U), dtype=I64)
                    for lam, c in I.skew(alpha, g).items(): a = (a + c * self.schur_vec(lam, 1)) % p
                    b = np.zeros(len(self.U), dtype=I64)
                    for lam, c in I.skew(beta, conjugate(g)).items(): b = (b + c * self.schur_vec(lam, -1)) % p
                    t = a * b % p
                    tot = (tot + t) % p if k % 2 == 0 else (tot - t) % p
            self._cv[key] = tot
        return self._cv[key]

    def psi_vec(self, c):
        if c not in self._pv:
            p = self.p; d = self.d(c)
            if d == 0: raise ZeroDivisionError("vanishing composite dimension")
            s = np.zeros(len(self.U), dtype=I64)
            for a, b in I.composites(c): s = (s + self.chi_vec(a, b)) % p
            self._pv[c] = (s * self.inv(d) - 1) % p
        return self._pv[c]

    def psi(self, c, nu):
        i = self.Uidx.get(nu if type(nu) is tuple else tuple(nu))      # fast path: nu already a normalized key
        if i is None:
            nu = P(tuple(nu)); i = self.Uidx.get(nu)
            if i is None: return super().psi(c, nu)
        return int(self.psi_vec(c)[i])

    def E_row(self, c):
        if c in self._E: return self._E[c]
        p = self.p; B = list(I.support(c)); n = len(B); nodes = list(I.nodes(c))
        if n == 0:
            self._E[c] = {c: 1}; self._E[("rank", c)] = (0, 0); return self._E[c]
        idx = [self.Uidx[P(nu)] for nu in nodes]
        cols = [self.psi_vec(x)[idx] for x in B] + [(-self.psi_vec(c)[idx]) % p]
        M = np.stack(cols, axis=1) % p; m = M.shape[0]; piv_cols = []; r = 0
        for col in range(n):
            if r >= m: break
            nz = np.flatnonzero(M[r:, col])
            if len(nz) == 0: continue
            pr = r + int(nz[0])
            if pr != r: M[[r, pr]] = M[[pr, r]]
            M[r] = M[r] * pow(int(M[r, col]), p - 2, p) % p
            f = M[:, col].copy(); f[r] = 0
            nzr = np.flatnonzero(f)
            if len(nzr): M[nzr] = (M[nzr] - (f[nzr, None] * M[r][None, :]) % p) % p
            piv_cols.append(col); r += 1
        sol = [0] * n
        for i, col in enumerate(piv_cols): sol[col] = int(M[i, n])
        row = {c: 1}
        for x, v in zip(B, sol): row[x] = v
        self._E[c] = row; self._E[("rank", c)] = (len(piv_cols), n)
        return row
