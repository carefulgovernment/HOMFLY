"""Exact integer traces T_Q(q) of the Borromean braid from residues modulo two primes.

The full traces are known modulo P1 at enough points (borromean_pts.py), so every T_Q
is known exactly modulo P1: T_Q = r_Q + P1 * K_Q with r_Q the symmetric lift.  For
large channels some coefficients exceed P1/2 and K_Q != 0.  K_Q is supported on the
exponents S_Q where |r_Q| is large (the coefficients decay away from q^0), so it is
fixed by |S_Q| values of T_Q modulo a second prime P2; the remaining points check the
support assumption (a wrong support fails them unless a polynomial of degree ~150
vanishes at random points mod P2).  Channels with S_Q empty are only checked.
Finally T_Q(1) = dim of the multiplicity space (the braid is trivial at q = 1).

usage: python3 borromean_exact.py out.pkl 'P1 pkls glob' 'P2 pkls glob' [threshold_bits]
"""
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from borromean_assemble import load, recon_w, w_to_q  # noqa: E402


def solve_mod(A, b, p):
    """square system mod p (None if singular)"""
    n = len(A)
    M = [list(r) + [y] for r, y in zip(A, b)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] % p), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        inv = pow(M[c][c], p - 2, p)
        M[c] = [x * inv % p for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % p for x, y in zip(M[r], M[c])]
    return [M[r][n] for r in range(n)]


def sym_eval(e, q, p):
    return 1 if e == 0 else (pow(q, e, p) + pow(q, p - 1 - e % (p - 1), p)) % p


def main():
    out, g1, g2 = sys.argv[1:4]
    bits = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    R, p1, v1 = load(sorted(glob.glob(g1)))
    R2, p2, v2 = load(sorted(glob.glob(g2)))
    assert R == R2 and p1 != p2
    dims = {}
    for f in sorted(glob.glob(g1)):
        for Q, (d, _) in pickle.load(open(f, "rb"))["T"].items():
            dims[tuple(Q)] = d
    mult = dims
    if os.environ.get("WBOUND"):                     # {"wbound": {Q: deg_w}, "mult": {Q: dim}}
        W = pickle.load(open(os.environ["WBOUND"], "rb"))
        assert W["mult"] == mult, "channels or multiplicities differ from the expected ones"
    assert set(v1) == set(mult) == set(v2), "channels missing"
    exact, stats = {}, [0, 0, 0]
    inv1 = pow(p1, p2 - 2, p2)
    for Q in sorted(mult):
        byw = {}
        for q, y in v1[Q].items():
            byw[(q * q + pow(q * q, p1 - 2, p1)) % p1] = y
        # spare points: beyond the proven w-degree bound if given, else 4
        spare = 4 if not os.environ.get("WBOUND") else len(byw) - W["wbound"][Q] - 1
        mono = recon_w(sorted(byw.items()), p1, spare)
        assert mono is not None, ("mod P1 reconstruction failed", Q)
        r = w_to_q(mono, p1)
        pts = sorted(v2[Q].items())
        rhs = [(y - sum(c * pow(q, e % (p2 - 1), p2) for e, c in r.items())) * inv1 % p2 for q, y in pts]
        order = sorted((e for e in r if e >= 0), key=lambda e: -abs(r[e]))
        nS = sum(1 for e in order if abs(r[e]) > p1 >> bits)
        while True:
            S = order[:nS]
            k = [0] * len(S)
            if S:
                k = solve_mod([[sym_eval(e, q, p2) for e in S] for q, _ in pts[:nS]], rhs[:nS], p2)
            ok = k is not None and all(
                sum(kk * sym_eval(e, q, p2) for kk, e in zip(k, S)) % p2 == y for (q, _), y in zip(pts[nS:], rhs[nS:]))
            if ok or nS + 3 > len(pts):
                break
            nS += 1                                      # widen the support and retry
        assert ok, ("no consistent lift", Q, nS, len(pts))
        T = dict(r)
        for kk, e in zip(k, S):
            kk = kk if kk <= p2 // 2 else kk - p2
            if kk:
                stats[2] += 1
                for ee in {e, -e}:
                    T[ee] = T.get(ee, 0) + p1 * kk
        T = {e: c for e, c in T.items() if c}
        assert sum(T.values()) == mult[Q], ("T_Q(1) != dim", Q)
        assert all(T.get(-e) == c for e, c in T.items())
        exact[Q] = T
        stats[0] += 1
        stats[1] += bool(S)
    print("%d channels exact; %d with a P2 correction support, %d coefficients corrected; max |c| %d"
          % (stats[0], stats[1], stats[2], max(abs(c) for T in exact.values() for c in T.values())))
    pickle.dump({"R": R, "T": exact}, open(out, "wb"))


if __name__ == "__main__":
    main()
