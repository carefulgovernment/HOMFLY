"""Colored HOMFLY of the Borromean rings (all components colored R) from the
traces T_Q(q) = Tr_Q (R1 R2^-1)^3 at many q (borromean_pts.py output).

1. Every T_Q is a Laurent polynomial in q with T_Q(q) = T_Q(1/q) = T_Q(-q), i.e.
   a polynomial in w = q^2 + q^-2: interpolate in w, keep `spare` points as a
   check, lift the coefficients to integers (symmetric residues).
2. H_R(A, q) = sum_Q dim_q(Q) T_Q(q) / dim_q(R) (invariant under (A, q) ->
   (1/A, 1/q) term by term, so natural and standard conventions agree).
   H * den(q), den = prod_{boxes of R} (q^h - q^-h)^2, is reconstructed as a
   Laurent polynomial by dense interpolation modulo several primes (the exact T_Q
   reduce mod any prime; EXACT_T=pkl from borromean_exact.py) and CRT, checked at
   fresh points and for stability of the CRT lift without the last prime.

usage: python3 borromean_assemble.py R out.json T_*.pkl ...
"""
import glob
import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homfly.algebra.fields import GF  # noqa: E402
from homfly.algebra.laurent import Laurent  # noqa: E402
from homfly.io.export import to_json  # noqa: E402
from homfly.reconstruction.interpolation import dense_bivariate  # noqa: E402

P1, P2 = 2147483647, 2147483629


def conj(R):
    return tuple(sum(1 for r in R if r > i) for i in range(R[0])) if R else ()


def boxes(Q):
    lt = conj(Q)
    return [(j - i, (Q[i] - j - 1) + (lt[j] - i - 1) + 1) for i in range(len(Q)) for j in range(Q[i])]


def recon_w(pts, p, spare):
    """pts [(w, y)] mod p -> monomial coefficients in w (checked on `spare` extra points)"""
    xs = [w for w, _ in pts]
    c = [y for _, y in pts]
    n = len(xs) - spare
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            c[i] = (c[i] - c[i - 1]) * pow(xs[i] - xs[i - j], p - 2, p) % p
    c = c[:n]
    # Newton -> monomial
    mono = [0] * n
    for k in range(n - 1, -1, -1):
        # mono = mono * (w - x_k) + c_k
        new = [0] * n
        for d in range(n - 1):
            if mono[d]:
                new[d + 1] = (new[d + 1] + mono[d]) % p
                new[d] = (new[d] - mono[d] * xs[k]) % p
        new[0] = (new[0] + c[k]) % p
        mono = new
    for w, y in pts[n:]:
        v = 0
        for a in reversed(mono):
            v = (v * w + a) % p
        if v != y:
            return None
    while mono and mono[-1] == 0:
        mono.pop()
    return mono


def w_to_q(mono, p):
    """sum_j a_j (q^2 + q^-2)^j -> {exponent of q: coefficient} with symmetric lift"""
    from math import comb
    out = {}
    for j, a in enumerate(mono):
        if a:
            for k in range(j + 1):
                e = 2 * (j - 2 * k)
                out[e] = (out.get(e, 0) + a * comb(j, k)) % p
    lift = {}
    for e, c in out.items():
        c = c if c <= p // 2 else c - p
        if c:
            lift[e] = c
    return lift


def load(files):
    vals = {}
    p = None
    R = None
    for f in files:
        d = pickle.load(open(f, "rb"))
        p = d["P"] if p is None else p
        assert d["P"] == p
        R = tuple(d["R"])
        for Q, (dim, t) in d["T"].items():
            for q, y in zip(d["qs"], t):
                vals.setdefault(tuple(Q), {})[int(q)] = int(y) % p
    return R, p, vals


def main():
    R = tuple(int(x) for x in sys.argv[1].split(","))
    out = sys.argv[2]
    files = [f for a in sys.argv[3:] for f in sorted(glob.glob(a))]
    if os.environ.get("EXACT_T"):
        # exact integer traces (borromean_exact.py): the T_Q reduce modulo any prime
        d = pickle.load(open(os.environ["EXACT_T"], "rb"))
        assert tuple(d["R"]) == R
        T, p, modp = d["T"], None, False
        print("exact T_Q: %d channels, max |coeff| %d" % (len(T), max(abs(c) for t in T.values() for c in t.values())))
    else:
        R2, p, vals = load(files)
        assert R2 == R
        spare = int(os.environ.get("SPARE", "4"))
        T, maxdeg, maxc = {}, 0, 0
        for Q, d in vals.items():
            byw = {}
            for q, y in d.items():
                w = (q * q + pow(q * q, p - 2, p)) % p
                if w in byw:
                    assert byw[w] == y, ("symmetry T(q) = T(1/q) violated", Q)
                byw[w] = y
            mono = recon_w(sorted(byw.items()), p, spare)
            if mono is None:
                raise SystemExit("not enough points for Q=%s (%d values)" % (Q, len(byw)))
            T[Q] = w_to_q(mono, p)
            maxdeg = max(maxdeg, len(mono) - 1)
            maxc = max([maxc] + [abs(c) for c in T[Q].values()])
        print("T_Q reconstructed: %d channels, max degree in w %d (points %d), max |coeff| %d"
              % (len(T), maxdeg, min(len(v) for v in vals.values()), maxc), flush=True)
        # MODP_ONLY=1: the T_Q are only used modulo the prime they were computed with (their
        # residues are exact even when a coefficient exceeds p/2); H is then reconstructed
        # modulo that prime alone and its coefficients must stay below p/2.
        modp = bool(os.environ.get("MODP_ONLY"))
        assert modp or maxc < p // 2 ** 8 or os.environ.get("NOLIFTCHECK"), "coefficients too close to p/2 for a safe lift"
    den_pow = int(os.environ.get("DEN_POW", "2"))
    den_boxes = [h for c, h in boxes(R)]
    Qdata = [(sorted(Tq.items()), boxes(Q)) for Q, Tq in T.items() if Tq]

    def make_f(F):
        pp = F.p
        cache = {}

        def row(qv):
            q = int(qv)
            if q in cache:
                return cache[q]
            iq = pow(q, pp - 2, pp)
            terms = []
            for tq, bx in Qdata:
                t = sum(c * pow(q if e >= 0 else iq, abs(e), pp) for e, c in tq) % pp
                hd = 1
                for c_, h in bx:
                    hd = hd * (pow(q, h, pp) - pow(iq, h, pp)) % pp
                terms.append((t * pow(hd, pp - 2, pp) % pp, [c_ for c_, h in bx]))
            dR = 1
            for c_, h in boxes(R):
                dR = dR * (pow(q, h, pp) - pow(iq, h, pp)) % pp
            den = 1
            for h in den_boxes:
                den = den * pow(pow(q, h, pp) - pow(iq, h, pp), den_pow, pp) % pp
            cache.clear()
            cache[q] = (terms, dR, den, q, iq)
            return cache[q]

        def f2(a, qv):
            terms, dR, den, q, iq = row(qv)
            a = int(a)
            g = {}
            tot = 0
            for w, cs in terms:
                for c_ in cs:
                    x = g.get(c_)
                    if x is None:
                        y = a * pow(q if c_ >= 0 else iq, abs(c_), pp) % pp
                        x = g[c_] = (y - pow(y, pp - 2, pp)) % pp
                    w = w * x % pp
                tot += w
            numR = 1
            for c_, h in boxes(R):
                y = a * pow(q if c_ >= 0 else iq, abs(c_), pp) % pp
                numR = numR * (y - pow(y, pp - 2, pp)) % pp
            # H * den = sum_Q dim_q(Q) T_Q * den / dim_q(R)
            return F(tot % pp * den % pp * dR % pp * pow(numR, pp - 2, pp) % pp)
        return f2

    from pointwise_montesinos import line_exponents
    res = {}
    primes = [p] if modp else [int(x) for x in os.environ.get("PRIMES", "%d,%d" % (P1, P2)).split(",")]
    for prime in primes:
        F = GF(prime)
        f2 = make_f(F)
        rng = random.Random(7)
        a0 = F(rng.randrange(2, prime - 1))
        qs = [F(rng.randrange(2, prime - 1)) for _ in range(int(os.environ.get("QLINE", "700")))]
        qbox = line_exponents(qs, [f2(a0, x) for x in qs], F, 1, -int(os.environ.get("QLOWER", "400")))
        q0 = F(rng.randrange(2, prime - 1))
        As = [F(rng.randrange(2, prime - 1)) for _ in range(int(os.environ.get("ALINE", "140")))]
        Abox = line_exponents(As, [f2(x, q0) for x in As], F, 1, -int(os.environ.get("ALOWER", "100")))
        print("p=%d box A %s q %s" % (prime, Abox, qbox), flush=True)
        sA = 2 if (Abox[0] % 2 == 0 and Abox[1] % 2 == 0) else 1
        sq = 2 if (qbox[0] % 2 == 0 and qbox[1] % 2 == 0) else 1
        img = dense_bivariate(f2, F, Abox, qbox, steps=(sA, sq), rng=random.Random(3))
        res[prime] = {k: int(v) for k, v in img.items() if int(v)}
        # fresh points
        for _ in range(6):
            a, q = F(rng.randrange(2, prime - 1)), F(rng.randrange(2, prime - 1))
            v = sum((F(c) * a ** i * q ** j for (i, j), c in res[prime].items()), F.zero)
            assert v == f2(a, q), "fresh point check failed mod %d" % prime

    def crt(ps):
        M, terms = 1, {}
        for pr in ps:
            for k in set(terms) | set(res[pr]):
                x, y = terms.get(k, 0), res[pr].get(k, 0)
                terms[k] = x + M * ((y - x) * pow(M, -1, pr) % pr)
            M *= pr
        return {k: (c if c <= M // 2 else c - M) for k, c in terms.items() if c % M}, M
    terms, M = crt(primes)
    if not modp:
        # the lift must already be stable without the last prime (a margin of one prime)
        sub, M1 = crt(primes[:-1])
        assert len(primes) > 1 and sub == terms, "CRT not stable: add primes (PRIMES=...)"
        print("CRT over %d primes stable; max |c| / (M/2) without the last prime: %.3g"
              % (len(primes), 2.0 * max(abs(c) for c in terms.values()) / M1))
    H = Laurent({k: c for k, c in terms.items()}, ("A", "q"))
    den = "*".join("(q^%d - q^-%d)^%d" % (h, h, den_pow) for h in den_boxes)
    json.dump({"link": "L6a4 (Borromean rings)", "R": list(R), "normalisation": "H * %s" % den,
               "primes": primes,
               "numerator": to_json(H), "terms": len(terms),
               "max_coeff": max(abs(c) for c in terms.values())}, open(out, "w"))
    print("numerator: %d terms, max |c| %d; wrote %s" % (len(terms), max(abs(c) for c in terms.values()), out))


if __name__ == "__main__":
    main()
