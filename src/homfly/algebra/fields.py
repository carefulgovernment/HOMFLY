"""Coefficient fields used by the evaluation engines.

Every computational method in :mod:`homfly.methods` is written *generically*
over a field ``F``: it receives numerical values ``A0, q0`` that are elements of
``F`` and returns the value of the (reduced) HOMFLY polynomial at that point.
The same code therefore serves

* exact rational evaluation (``QQ``)                      -- debugging, small cases
* evaluation modulo a prime (``GF(p)``)                   -- the production path,
  followed by interpolation + CRT (see :mod:`homfly.reconstruction`)
* floating point / complex evaluation (``CC``)            -- quick numerical checks,
  e.g. specialisation ``A = q^N`` at roots of unity.

A field object exposes ``zero``, ``one``, ``__call__`` (coercion from int /
Fraction), ``sqrt`` (may raise :class:`NoSquareRoot`), ``random_element`` and
``characteristic``.  Elements support ``+ - * / **`` (negative powers included).
"""
from __future__ import annotations

import cmath
import random
from fractions import Fraction


class BadPoint(ArithmeticError):
    """Raised when an evaluation point is unusable (division by zero, missing
    square root, ...).  Reconstruction drivers catch it and resample."""


class NoSquareRoot(BadPoint):
    pass


# ----------------------------------------------------------------------------
# Rational numbers
# ----------------------------------------------------------------------------
class RationalField:
    characteristic = 0
    name = "QQ"

    def __call__(self, x):
        return Fraction(x)

    @property
    def zero(self):
        return Fraction(0)

    @property
    def one(self):
        return Fraction(1)

    def inv(self, x):
        if x == 0:
            raise BadPoint("division by zero in QQ")
        return 1 / Fraction(x)

    def sqrt(self, x):
        x = Fraction(x)
        if x < 0:
            raise NoSquareRoot(x)
        n, d = _isqrt_exact(x.numerator), _isqrt_exact(x.denominator)
        if n is None or d is None:
            raise NoSquareRoot(x)
        return Fraction(n, d)

    def random_element(self, rng=random, bound=10**6):
        return Fraction(rng.randint(2, bound), rng.randint(1, bound))

    def __repr__(self):
        return "QQ"


def _isqrt_exact(n):
    if n < 0:
        return None
    import math
    r = math.isqrt(n)
    return r if r * r == n else None


QQ = RationalField()


# ----------------------------------------------------------------------------
# Prime fields
# ----------------------------------------------------------------------------
class ModP:
    """Element of GF(p).  Kept deliberately small; hot loops should later be
    moved to python-flint ``nmod`` / ``nmod_mat`` or to compiled code."""

    __slots__ = ("v", "p")

    def __init__(self, v, p):
        self.v = v % p
        self.p = p

    def _c(self, o):
        if isinstance(o, ModP):
            return o.v
        if isinstance(o, int):
            return o % self.p
        if isinstance(o, Fraction):
            if o.denominator % self.p == 0:
                raise BadPoint("denominator divisible by p")
            return o.numerator * pow(o.denominator, -1, self.p) % self.p
        return NotImplemented

    def __add__(self, o):
        o = self._c(o)
        return NotImplemented if o is NotImplemented else ModP(self.v + o, self.p)

    __radd__ = __add__

    def __sub__(self, o):
        o = self._c(o)
        return NotImplemented if o is NotImplemented else ModP(self.v - o, self.p)

    def __rsub__(self, o):
        o = self._c(o)
        return NotImplemented if o is NotImplemented else ModP(o - self.v, self.p)

    def __mul__(self, o):
        o = self._c(o)
        return NotImplemented if o is NotImplemented else ModP(self.v * o, self.p)

    __rmul__ = __mul__

    def __neg__(self):
        return ModP(-self.v, self.p)

    def inverse(self):
        if self.v == 0:
            raise BadPoint("division by zero mod %d" % self.p)
        return ModP(pow(self.v, -1, self.p), self.p)

    def __truediv__(self, o):
        o = self._c(o)
        if o is NotImplemented:
            return o
        if o == 0:
            raise BadPoint("division by zero mod %d" % self.p)
        return ModP(self.v * pow(o, -1, self.p), self.p)

    def __rtruediv__(self, o):
        return ModP(self._c(o), self.p) / self

    def __pow__(self, e):
        if e < 0:
            return self.inverse() ** (-e)
        return ModP(pow(self.v, e, self.p), self.p)

    def __eq__(self, o):
        o = self._c(o)
        return o is not NotImplemented and self.v == o

    def __hash__(self):
        return hash((self.v, self.p))

    def __bool__(self):
        return self.v != 0

    def __int__(self):
        return self.v

    def __repr__(self):
        return "%d (mod %d)" % (self.v, self.p)


class PrimeField:
    def __init__(self, p):
        if not is_probable_prime(p):
            raise ValueError("%d is not prime" % p)
        self.p = p
        self.characteristic = p
        self.name = "GF(%d)" % p

    def __call__(self, x):
        if isinstance(x, ModP):
            return x
        if isinstance(x, Fraction):
            return ModP(1, self.p) * x
        return ModP(int(x), self.p)

    @property
    def zero(self):
        return ModP(0, self.p)

    @property
    def one(self):
        return ModP(1, self.p)

    def inv(self, x):
        return self(x).inverse()

    def sqrt(self, x):
        r = tonelli_shanks(int(self(x)), self.p)
        if r is None:
            raise NoSquareRoot("%s is not a square mod %d" % (x, self.p))
        return ModP(r, self.p)

    def random_element(self, rng=random):
        return ModP(rng.randrange(2, self.p - 1), self.p)

    def __repr__(self):
        return self.name

    def __eq__(self, o):
        return isinstance(o, PrimeField) and o.p == self.p

    def __hash__(self):
        return hash(("GF", self.p))


def GF(p):
    return PrimeField(p)


def is_probable_prime(n):
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for s in small:
        if n % s == 0:
            return n == s
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:  # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_below(bound, count):
    """Largest ``count`` primes below ``bound`` (descending)."""
    out, n = [], bound - 1
    while len(out) < count:
        if is_probable_prime(n):
            out.append(n)
        n -= 1
    return out


def tonelli_shanks(a, p):
    a %= p
    if a == 0:
        return 0
    if p == 2:
        return a
    if pow(a, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


# ----------------------------------------------------------------------------
# Complex numbers (numerical checks only)
# ----------------------------------------------------------------------------
class ComplexField:
    characteristic = 0
    name = "CC"

    def __call__(self, x):
        return complex(x)

    zero = 0j
    one = 1 + 0j

    def inv(self, x):
        if x == 0:
            raise BadPoint("division by zero in CC")
        return 1 / x

    def sqrt(self, x):
        return cmath.sqrt(x)

    def random_element(self, rng=random):
        return cmath.exp(2j * cmath.pi * rng.random()) * (0.8 + 0.4 * rng.random())

    def __repr__(self):
        return "CC"


CC = ComplexField()
