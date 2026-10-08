"""Write data/homfly/L6a4.txt (Borromean rings, all components colored R) from the
borromean_assemble.py outputs.

H_R = N_R(A, q) / den_R(q), den_R = prod_{boxes of R} (q^h - q^-h)^2 (hook lengths h),
with N_R a Laurent polynomial with integer coefficients.  Checks per R: H_R(A, q) is
invariant under (A, q) -> (1/A, 1/q) (the link is amphichiral), the numerator is not
divisible by any cyclotomic factor of den_R (the fraction is reduced), and for R = [1]
the sl_2 specialisation A = q^2 is the Jones polynomial -t^3+3t^2-2t+4-2/t+3/t^2-1/t^3,
t = q^2; transposed pairs satisfy H_{R^T}(A, q) = H_R(A, -1/q).

usage: python3 borromean_datafile.py out.txt h_R1.json h_R2.json ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from homfly.algebra.laurent import Laurent  # noqa: E402
from homfly.io.export import from_json  # noqa: E402


def conj(R):
    return tuple(sum(1 for r in R if r > i) for i in range(R[0])) if R else ()


def hooks(R):
    lt = conj(R)
    return [(R[i] - j - 1) + (lt[j] - i - 1) + 1 for i in range(len(R)) for j in range(R[i])]


def cyclotomic(d):
    """integer coefficients of Phi_d(x), lowest degree first"""
    from sympy import cyclotomic_poly, Poly, Symbol
    x = Symbol("x")
    return [int(c) for c in reversed(Poly(cyclotomic_poly(d, x), x).all_coeffs())]


def divides(N, f):
    """is every A-coefficient of N (a polynomial in q) divisible by f(q)?"""
    rows = {}
    for (i, j), c in N.terms.items():
        rows.setdefault(i, {})[j] = c
    for row in rows.values():
        lo = min(row)
        a = [0] * (max(row) - lo + 1)
        for j, c in row.items():
            a[j - lo] = c
        n = len(f) - 1
        for k in range(len(a) - 1, n - 1, -1):      # f is monic
            c = a[k]
            if c:
                for t in range(n + 1):
                    a[k - n + t] -= c * f[t]
        if any(a[:n]):
            return False
    return True


def main():
    out = sys.argv[1]
    data = []
    for f in sys.argv[2:]:
        d = json.load(open(f))
        R = tuple(d["R"])
        N = from_json(d["numerator"] if isinstance(d["numerator"], str) else json.dumps(d["numerator"]))
        hs = hooks(R)
        assert d["normalisation"] == "H * " + "*".join("(q^%d - q^-%d)^2" % (h, h) for h in hs), d["normalisation"]
        # (A, q) -> (1/A, 1/q): the denominator is invariant
        assert all(N.terms.get((-i, -j)) == c for (i, j), c in N.terms.items()), ("not amphichiral", R)
        ds = sorted({e for h in hs for e in range(1, 2 * h + 1) if (2 * h) % e == 0})
        assert not any(divides(N, cyclotomic(e)) for e in ds), ("fraction not reduced", R)
        data.append((R, N, d))
    byR = {R: N for R, N, _ in data}
    if (1,) in byR:
        import sympy as sp
        A, q = sp.symbols("A q")
        H1 = sum(c * A ** i * q ** j for (i, j), c in byR[(1,)].terms.items()) / (q - 1 / q) ** 2
        t = q ** 2
        assert sp.simplify(H1.subs(A, q ** 2) - (-t ** 3 + 3 * t ** 2 - 2 * t + 4 - 2 / t + 3 / t ** 2 - 1 / t ** 3)) == 0
    for R, N in byR.items():
        if conj(R) in byR and conj(R) != R:
            # N_{R^T}(A, q) = N_R(A, -1/q) * (-1)^{deg den} ... den(-1/q) = den(q) (even powers)
            assert byR[conj(R)].terms == {(i, -j): c * (-1) ** j for (i, j), c in N.terms.items()}, ("transposition", R)
    data.sort(key=lambda x: (sum(x[0]), [-r for r in x[0]]))
    with open(out, "w") as fh:
        fh.write("# Colored HOMFLY polynomials H_R(A, q) of the Borromean rings L6a4 = closure of the\n"
                 "#   3-braid (s1 s2^-1)^3, all three components colored R.\n"
                 "# Convention: A^{-1}H(L+) - A H(L-) = (q - 1/q) H(L0); H_R(unknot) = 1 (reduced:\n"
                 "#   unnormalised invariant / dim_q R); no framing factor needed (all linking numbers 0).\n"
                 "# H_R = N / den with den = prod over boxes of R of (q^h - q^-h)^2, h = hook length;\n"
                 "#   the fraction is reduced.  Checks: (A, q) -> (1/A, 1/q) symmetry; R = [1] at A = q^2\n"
                 "#   gives the Jones polynomial with t = q^2; H_{R^T}(A, q) = H_R(A, -1/q).\n"
                 "# Method: H_R = sum_Q dim_q(Q) Tr_Q (R1 R2^-1)^3 / dim_q(R) over R x R x R -> Q, inclusive\n"
                 "#   Racah matrices numerically at each q (scripts/borromean_pts.py); exact integer traces\n"
                 "#   from two primes (scripts/borromean_exact.py), H by dense interpolation modulo several\n"
                 "#   primes with a stable CRT lift (scripts/borromean_assemble.py).\n\n")
        for R, N, d in data:
            fh.write("R = %s\n" % list(R))
            fh.write("# primes=%d terms=%d max_coeff=%d\n" % (len(d.get("primes", ())), len(N.terms),
                                                           max(abs(c) for c in N.terms.values())))
            fh.write("den = %s\n" % " * ".join("(q^%d - q^(-%d))^2" % (h, h) for h in hooks(R)))
            fh.write("N = %s\n\n" % N.to_string(mul="*", pow_="^"))
    print("wrote %s: %s" % (out, [list(R) for R, _, _ in data]))


if __name__ == "__main__":
    main()
