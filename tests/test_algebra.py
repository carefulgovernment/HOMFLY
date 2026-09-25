import random
from fractions import Fraction

from homfly.algebra.fields import GF, QQ, tonelli_shanks
from homfly.algebra.laurent import Laurent, parse_laurent
from homfly.reconstruction.crt import (crt, maximal_quotient_rational_reconstruction,
                                       rational_reconstruction, symmetric_lift)
from homfly.reconstruction.interpolation import dense_bivariate, interpolate, univariate_laurent
from homfly.reconstruction.pipeline import reconstruct_laurent


def test_parse_and_arith():
    P = parse_laurent("(2*v^2-v^4)+(v^2)*z^2", vars=("v", "z"))
    assert P.terms == {(2, 0): 2, (4, 0): -1, (2, 2): 1}
    A, q = Laurent.var("A"), Laurent.var("q")
    assert parse_laurent("A^(-2)*q**3 - 3") == A ** -2 * q ** 3 - 3
    assert (A + q) ** 2 == A * A + 2 * A * q + q * q


def test_tonelli():
    p = 1000000007
    for a in range(2, 50):
        r = tonelli_shanks(a, p)
        if r is not None:
            assert r * r % p == a


def test_crt_and_ratrec():
    ps = [1000003, 1000033, 1000037]
    x = -123456789012
    r, m = crt([x % p for p in ps], ps)
    assert symmetric_lift(r, m) == x
    f = Fraction(-355, 113)
    m = ps[0] * ps[1]
    r = f.numerator * pow(f.denominator, -1, m) % m
    assert rational_reconstruction(r, m) == f
    m = ps[0] * ps[1] * ps[2]
    r = f.numerator * pow(f.denominator, -1, m) % m
    assert maximal_quotient_rational_reconstruction(r, m) == f


def test_interpolation():
    F = GF(1000003)
    xs = [F(i) for i in range(1, 6)]
    ys = [x ** 4 - 3 * x + 7 for x in xs]
    assert [int(c) for c in interpolate(xs, ys, F)] == [7, 1000000, 0, 0, 1]
    f = lambda x: 5 * x ** -7 + x ** 9  # noqa: E731
    d = univariate_laurent(f, F, lower=-2, step=1, rng=random.Random(1))
    assert {k: int(v) for k, v in d.items()} == {-7: 5, 9: 1}


def test_bivariate_and_pipeline():
    H = parse_laurent("A^4*q^-2 - 3*A^-2*q^6 + 12345678901*A^2")
    F = GF(1000003)
    img = dense_bivariate(lambda a, q: H.evaluate([a, q], one=F.one), F, (-2, 4), (-2, 6),
                          steps=(2, 2), rng=random.Random(0))
    assert {k: int(v) for k, v in img.items()} == {k: c % 1000003 for k, c in H.terms.items()}
    G = reconstruct_laurent(lambda F, a, q: H.evaluate([a, q], one=F.one), steps=(2, 2),
                            prime_bound=2 ** 31, seed=3)
    assert G == H


def test_rational_pipeline():
    H = parse_laurent("A^2*q - q^-1") / 7
    G = reconstruct_laurent(lambda F, a, q: H.evaluate([a, q], one=F.one), steps=(1, 1),
                            rational=True, prime_bound=2 ** 31, seed=5)
    assert G == H


def test_qq_field():
    assert QQ.sqrt(Fraction(9, 4)) == Fraction(3, 2)
