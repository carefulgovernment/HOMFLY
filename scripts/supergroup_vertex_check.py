"""Independent check that H_[r](A = q^{N-M}, q) is the (1,1)-tangle invariant of U_q gl(N|M) on Sym^r C^{N|M}.

No Hecke seminormal forms, Racah matrices or interpolation: plain float64 linear algebra.

  * V = C^{N|M} with parities `par`; the Perk-Schultz matrix T on V (x) V satisfies the Hecke relation
    (T - q)(T + 1/q) = 0 and the braid relation (checked at start-up).
  * Sym^r V = joint q-eigenspace of T_1..T_{r-1} on V^{(x)r} (image of the q-symmetrizer); the cabled
    crossing of two r-ribbons is restricted to Sym^r V (x) Sym^r V.
  * The pivotal element of V is the diagonal mu with tr_2((1 (x) mu) T^{+-1}) = kappa^{+-1} Id; on Sym^r V it is
    mu^{(x)r} restricted.  The knot value is the scalar of the partial trace over strands 2..n of the braid,
    divided by twist^writhe.

For gl(1|m+1) (par = 0 1...1) and r >= m+1, Sym^r V is the 2^{m+1}-dimensional typical module, so this is
LG^{m+1,1} at alpha = r, and it is compared with the data in data/homfly at A = q^{M-N}, q -> 1/q (the
parity change mirrors the knot).

usage: python3 scripts/supergroup_vertex_check.py PARITIES r q MAX_BRAID_INDEX [knot ...]
       e.g.  supergroup_vertex_check.py 011 3 0.81 5        (gl(1|2), Sym^3, every knot with data)
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from homfly.algebra.laurent import parse_laurent  # noqa: E402

KNOTINFO = os.path.join(HERE, "..", "src", "homfly", "knots", "data", "knotinfo_upto12.csv")
DATA = os.path.join(HERE, "..", "data", "homfly")


def perk_schultz(par, q):
    n = len(par)
    T = np.zeros((n * n, n * n))
    for i in range(n):
        T[i * n + i, i * n + i] = q if par[i] == 0 else -1 / q
        for j in range(i + 1, n):
            s = (-1) ** (par[i] * par[j])
            T[j * n + i, i * n + j] = s                 # e_i e_j -> s e_j e_i
            T[i * n + j, j * n + i] = s                 # e_j e_i -> s e_i e_j + (q - 1/q) e_j e_i
            T[j * n + i, j * n + i] = q - 1 / q
    return T


def op_on(T, d, k, total):
    return np.kron(np.kron(np.eye(d ** k), T), np.eye(d ** (total - k - 2)))


def kron_all(ms):
    out = np.eye(1)
    for m in ms:
        out = np.kron(out, m)
    return out


def null(M):
    _, s, Vt = np.linalg.svd(M)
    return Vt[int((s > 1e-9).sum()):].T


def pivot(T, d):
    """diagonal mu, kappa with tr_2((1 x mu) T) = kappa Id, tr_2((1 x mu) T^-1) = Id / kappa"""
    A = []
    for which, M in enumerate((T, np.linalg.inv(T))):
        M4 = M.reshape(d, d, d, d)
        for i in range(d):
            for ip in range(d):
                row = [M4[i, j, ip, j] for j in range(d)] + [0.0, 0.0]
                if i == ip:
                    row[d + which] = -1.0
                A.append(row)
    x = null(np.array(A))[:, -1]
    mu, ka, kb = x[:d], x[d], x[d + 1]
    sc = 1 / np.sqrt(abs(ka * kb))
    return mu * sc, ka * sc


def fused(par, r, q):
    """(R on Sym^r V (x) Sym^r V, dim Sym^r V, pivotal element on Sym^r V, twist)"""
    d = len(par)
    T = perk_schultz(par, q)
    I = np.eye(d ** r)
    if r > 1:
        Lb = null(np.vstack([op_on(T, d, k, r) - q * I for k in range(r - 1)]))
        Rb = null(np.hstack([op_on(T, d, k, r) - q * I for k in range(r - 1)]).T).T
    else:
        Lb = Rb = I
    Rb = np.linalg.solve(Rb @ Lb, Rb)
    k, n = Lb.shape[1], 2 * r

    def apply(kpos, Mc):
        X = Mc.reshape(d ** kpos, d * d, d ** (n - kpos - 2), Mc.shape[1])
        return np.einsum("ab,ibjc->iajc", T, X).reshape(d ** n, Mc.shape[1])

    Mc = np.kron(Lb, Lb)
    for i in range(r):                  # strand r-1-i of the first ribbon crosses the second ribbon
        for j in range(r):
            Mc = apply(r - 1 - i + j, Mc)
    RW = np.kron(Rb, Rb) @ Mc
    mu1, _ = pivot(T, d)
    muW = Rb @ kron_all([np.diag(mu1)] * r) @ Lb
    C = np.einsum("iaja->ij", (np.kron(np.eye(k), muW) @ RW).reshape(k, k, k, k))
    return RW, k, muW, C[0, 0]


def knot_value(word, RW, k, mu, twist):
    n = max(abs(x) for x in word) + 1
    Ri = np.linalg.inv(RW)
    B = np.eye(k ** n)
    for g in word:
        B = op_on(RW if g > 0 else Ri, k, abs(g) - 1, n) @ B
    M = (np.kron(np.eye(k), kron_all([mu] * (n - 1))) @ B).reshape(k, k ** (n - 1), k, k ** (n - 1))
    c = np.einsum("iaja->ij", M)
    w = sum(1 if g > 0 else -1 for g in word)
    return c[0, 0] / twist ** w, np.abs(c - c[0, 0] * np.eye(k)).max()


def homfly_value(knot, r, A, q):
    path, R = os.path.join(DATA, knot + ".txt"), None
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith("R = "):
            R = line[4:].strip()
        elif line.startswith("H = ") and R == "[%d]" % r:
            return sum(float(c) * A ** i * q ** j for (i, j), c in parse_laurent(line[4:].strip()).terms.items())


def main():
    par = [int(c) for c in sys.argv[1]]
    N, M = par.count(0), par.count(1)
    r, q, maxb = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    T = perk_schultz(par, q)
    d = len(par)
    I = np.eye(d * d)
    assert np.abs((T - q * I) @ (T + I / q)).max() < 1e-12
    T1, T2 = op_on(T, d, 0, 3), op_on(T, d, 1, 3)
    assert np.abs(T1 @ T2 @ T1 - T2 @ T1 @ T2).max() < 1e-12
    rows = {row["name"]: row for row in csv.DictReader(open(KNOTINFO), delimiter="|") if row["braid_notation"]}
    names = sys.argv[5:] or sorted(rows)
    RW, k, mu, twist = fused(par, r, q)
    ok = bad = skip = 0
    worst = 0.0
    for kn in names:
        row = rows.get(kn)
        h = homfly_value(kn, r, q ** (M - N), 1 / q) if row and int(row["braid_index"]) <= maxb else None
        if h is None:
            skip += 1
            continue
        word = eval(row["braid_notation"])
        word = word[0] if isinstance(word[0], list) else word
        v, dev = knot_value(word, RW, k, mu, twist)
        rel = abs(v - h) / max(1.0, abs(h))
        worst = max(worst, rel)
        if rel < 1e-7 and dev < 1e-6 * max(1.0, abs(v)):
            ok += 1
        else:
            bad += 1
            print("MISMATCH %s vertex %.12g data %.12g (scalar dev %.1e)" % (kn, v, h, dev))
    print("gl(%d|%d) Sym^%d (dim %d), q = %g: agree %d, mismatch %d, skipped %d, worst relative difference %.1e"
          % (N, M, r, k, q, ok, bad, skip, worst))


if __name__ == "__main__":
    main()
