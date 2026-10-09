"""Differential-expansion (defect) scan of the symmetric colored HOMFLY in data/homfly.

For symmetric R = [r] the differential expansion reads

    H_[r] = 1 + sum_{k=1}^{r} [r k]_q  D_{r+k-1} ... D_r  G_k(A, q),   D_n = A q^n - A^-1 q^-n,

with r-independent G_k.  G_k is extracted recursively from H_[k] (exact division by
D_{2k-1} ... D_k; failure is reported as a broken "plus" structure) and checked on
H_[4] for k = 4.  The "minus" factors are measured: m_k = the largest m with
D_{-1} D_0 ... D_{m-2} dividing G_k.  For defect delta the expected profile is
m_k = floor((k-1)/(delta+1)) + 1, and the conjecture delta = deg Alexander - 1 is
compared with it (deg = half the t-span of Delta(t) = H_[1](A=1, q^2 = t)).

usage: python3 scripts/defect_scan.py [knot ...] > report.tsv     (default: every knot)
"""
import glob
import os
import sys

from flint import fmpz_mpoly_ctx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from homfly.algebra.laurent import parse_laurent  # noqa: E402

CTX = fmpz_mpoly_ctx.get(("A", "q"), "lex")
A_, Q_ = CTX.gens()


class L:
    """Laurent polynomial poly * A^sa * q^sq with poly an fmpz_mpoly"""
    __slots__ = ("p", "sa", "sq")

    def __init__(self, p, sa=0, sq=0):
        self.p, self.sa, self.sq = p, sa, sq

    @staticmethod
    def from_terms(terms):
        if not terms:
            return L(CTX.from_dict({}))
        ma = min(i for i, j in terms)
        mq = min(j for i, j in terms)
        return L(CTX.from_dict({(i - ma, j - mq): int(c) for (i, j), c in terms.items()}), ma, mq)

    def _align(self, o):
        sa, sq = min(self.sa, o.sa), min(self.sq, o.sq)
        return (self.p * A_ ** (self.sa - sa) * Q_ ** (self.sq - sq),
                o.p * A_ ** (o.sa - sa) * Q_ ** (o.sq - sq), sa, sq)

    def __add__(self, o):
        a, b, sa, sq = self._align(o)
        return L(a + b, sa, sq)

    def __sub__(self, o):
        a, b, sa, sq = self._align(o)
        return L(a - b, sa, sq)

    def __mul__(self, o):
        return L(self.p * o.p, self.sa + o.sa, self.sq + o.sq)

    def is_zero(self):
        return self.p == 0

    def divexact(self, o):
        """self / o, None if not a Laurent polynomial"""
        qt, rm = divmod(self.p, o.p)
        if rm != 0:
            return None
        return L(qt, self.sa - o.sa, self.sq - o.sq)


ONE = L(CTX.from_dict({(0, 0): 1}))


def D(n):
    # A q^n - A^-1 q^-n = A^-1 q^-n (A^2 q^2n - 1)  (n >= 0),  = A^-1 q^n (A^2 - q^-2n) (n < 0)
    if n >= 0:
        return L(A_ ** 2 * Q_ ** (2 * n) - 1, -1, -n)
    return L(A_ ** 2 - Q_ ** (-2 * n), -1, n)


def qint(n):
    """[n] = (q^n - q^-n)/(q - q^-1)"""
    return L.from_terms({(0, n - 1 - 2 * i): 1 for i in range(n)})


def qbinom(n, k):
    num, den = ONE, ONE
    for i in range(k):
        num = num * qint(n - i)
        den = den * qint(i + 1)
    return num.divexact(den)


def load(path):
    H, R = {}, None
    for line in open(path):
        if line.startswith("R = "):
            R = tuple(int(x) for x in line[4:].strip(" []\n").split(","))
        elif line.startswith("H = ") and R is not None and len(R) == 1:
            H[R[0]] = L.from_terms(dict(parse_laurent(line[4:].strip()).terms))
    return H


def alexander_degree(H1):
    tq = {}
    for (i, j), c in zip(H1.p.monoms(), H1.p.coeffs()):
        jj = j + H1.sq
        tq[jj] = tq.get(jj, 0) + int(c)
    e = [j for j, c in tq.items() if c]
    return (max(e) - min(e)) // 4


def minus_count(G, cap):
    m = 0
    while m < cap:
        G2 = G.divexact(D(m - 1))
        if G2 is None:
            break
        G, m = G2, m + 1
    return m


def scan(knot, H, rmax=4):
    G = {}
    out = {"knot": knot, "degA": alexander_degree(H[1])}
    plus_ok = True
    for k in range(1, rmax + 1):
        if k not in H:
            break
        N = H[k] - ONE
        for j in range(1, k):
            t = qbinom(k, j)
            for i in range(j):
                t = t * D(k + i)
            N = N - t * G[j]
        den = ONE
        for i in range(k):
            den = den * D(k + i)
        Gk = N.divexact(den)
        if Gk is None:
            plus_ok = False
            out["plus_fail"] = k
            break
        G[k] = Gk
        out["m%d" % k] = minus_count(Gk, k + 2) if not Gk.is_zero() else "inf"
    out["plus_ok"] = plus_ok
    return out


def expected(delta, k):
    return (k - 1) // (delta + 1) + 1


def main():
    files = ([os.path.join("data/homfly", k + ".txt") for k in sys.argv[1:]] if sys.argv[1:]
             else sorted(glob.glob("data/homfly/*.txt")))
    print("knot\tdegAlex\tm1\tm2\tm3\tm4\tplus_ok\tconsistent_with_delta=deg-1\tfitted_delta")
    for f in files:
        knot = os.path.basename(f)[:-4]
        H = load(f)
        if 1 not in H or 2 not in H:
            continue
        r = scan(knot, H)
        ms = [r.get("m%d" % k) for k in range(1, 5)]
        delta = max(r["degA"] - 1, 0)
        cons = all(m is None or m == "inf" or m == expected(delta, k) for k, m in enumerate(ms, 1))
        fits = [d for d in range(0, 8)
                if all(m is None or m == "inf" or m == expected(d, k) for k, m in enumerate(ms, 1))]
        print("%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (knot, r["degA"], *["-" if m is None else m for m in ms],
                                                      r["plus_ok"] if r["plus_ok"] else "fail@%d" % r["plus_fail"],
                                                      cons, ",".join(map(str, fits)) or "none"), flush=True)


if __name__ == "__main__":
    main()
