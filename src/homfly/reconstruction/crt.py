"""Chinese remaindering and rational reconstruction."""
from __future__ import annotations

import math
from fractions import Fraction


def crt_pair(r1, m1, r2, m2):
    """x = r1 mod m1, x = r2 mod m2  (coprime moduli) -> (x mod m1*m2, m1*m2)."""
    inv = pow(m1, -1, m2)
    t = (r2 - r1) * inv % m2
    return (r1 + m1 * t) % (m1 * m2), m1 * m2


def crt(residues, moduli):
    r, m = 0, 1
    for ri, mi in zip(residues, moduli):
        r, m = crt_pair(r, m, ri % mi, mi)
    return r, m


def symmetric_lift(r, m):
    """Representative of r mod m in (-m/2, m/2]."""
    r %= m
    return r - m if r > m // 2 else r


def rational_reconstruction(r, m, N=None, D=None):
    """Wang's algorithm: find n/d with n = d*r mod m, |n| <= N, 0 < d <= D.

    Defaults N = D = floor(sqrt(m/2)), which makes the answer unique when it
    exists.  Returns a Fraction or None.
    """
    r %= m
    if N is None:
        N = math.isqrt(m // 2)
    if D is None:
        D = N
    r0, r1 = m, r
    s0, s1 = 0, 1
    while r1 > N:
        qt = r0 // r1
        r0, r1 = r1, r0 - qt * r1
        s0, s1 = s1, s0 - qt * s1
    if s1 == 0 or abs(s1) > D:
        return None
    if s1 < 0:
        r1, s1 = -r1, -s1
    if math.gcd(r1, s1) != 1:
        return None
    return Fraction(r1, s1)


def maximal_quotient_rational_reconstruction(r, m, T=None):
    """Monagan's MQRR: prefers the quotient with an unusually large partial
    quotient; succeeds with ~ (log2 N + log2 D + T) bits instead of 2*max.

    Returns a Fraction or None.  ``T`` is the confidence margin in bits.
    """
    r %= m
    if r == 0:
        return Fraction(0)
    if T is None:
        T = max(20, m.bit_length() // 20)
    best, bestq = None, 0
    r0, r1, s0, s1 = m, r, 0, 1
    while r1 != 0:
        qt = r0 // r1
        if qt > bestq and s1 != 0:
            bestq, best = qt, (r1, s1)
        r0, r1 = r1, r0 - qt * r1
        s0, s1 = s1, s0 - qt * s1
    if best is None or bestq.bit_length() < T:
        return None
    n, d = best
    if d < 0:
        n, d = -n, -d
    if math.gcd(n, d) != 1:
        return None
    return Fraction(n, d)
