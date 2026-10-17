"""Strengthened Formula II (homfly.formula2, methods.formula2) against the stored polynomials:
twist knots 3_1 .. 8_1 (|R| <= 6), the Borromean rings L6a4 (|R| <= 6 and [4,3,1]) and the eight twist knots in
R = [4,3,2,1] (data/twist_knots_4321: Tw_m = DoubleBraid(m, 1), m = -4..4), at several primes."""
import glob
import itertools
import json
import os

import pytest

from _homfly_txt import DATA, load, value
from homfly import formula2 as f2
from homfly.algebra.fields import GF
from homfly.knots.families import DoubleBraid
from homfly.methods import interpolation as I
from homfly.methods.formula2 import DoubleBraidStrong

P0, A0, Q0 = 67108859, 123457, 98765
# table knot = DoubleBraid(m, n) (KnotInfo chirality of data/homfly)
TWIST = {"3_1": (1, 1), "4_1": (1, -1), "5_2": (2, 1), "6_1": (2, -1), "7_2": (3, 1), "8_1": (3, -1)}


@pytest.fixture(scope="module")
def pt():
    return f2.Formula2Point(A0, Q0, P0)


def test_small_R_equals_naive_formula(pt):
    F = GF(P0)
    for R in [(1,), (2,), (2, 1), (3, 1), (2, 2), (3, 2, 1), (4, 2, 1), (2, 2, 2, 1)]:
        H = pt.double_braids(R, [(1, -1), (2, 3), (-2, 5)])
        for (m, n), v in H.items():
            assert v == I.paper_value(R, m, n, F, F(A0), F(Q0)).v, (R, m, n)


@pytest.mark.parametrize("knot", sorted(TWIST))
def test_twist_knots_vs_tables(pt, knot):
    table = load(knot)
    H = {R: pt.double_braid(R, *TWIST[knot]) for R in table}
    assert all(H[R] == value(table[R], A0, Q0, P0) for R in table), [R for R in table if H[R] != value(table[R], A0, Q0, P0)]


def test_borromean_vs_L6a4(pt):
    table = load("L6a4")
    assert (4, 3, 1) in table
    for R, entry in table.items():
        assert pt.borromean(R, R, R) == value(entry, A0, Q0, P0), R


@pytest.mark.parametrize("cols", [((2, 1), (3, 1), (2, 2)), ((4, 3, 2), (5, 3, 1), (4, 3, 1)),
                                  ((4, 3, 2, 1), (4, 3, 1), (3, 3, 2, 1))])
def test_borromean_tensor_symmetric(pt, cols):
    """1 + chi(R2)^T c(R1) chi(R3) does not depend on which colour carries c: T is one symmetric cubic tensor"""
    assert len({pt.contract(*perm) for perm in itertools.permutations(cols)}) == 1


def test_level_8_differs_from_naive(pt):
    """[4,3,1] is the first diagram where the naive formula fails (kernel cube); the Borromean check above pins it"""
    F = GF(P0)
    assert pt.twist_knot((4, 3, 1), -1) != I.paper_value((4, 3, 1), -1, 1, F, F(A0), F(Q0)).v


@pytest.mark.parametrize("p,A,q", [(P0, A0, Q0), (2147483629, 987654321, 123456789)])
def test_4321_twist_knots(p, A, q):
    fx = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, "twist_knots_4321", "*.json")))]
    assert len(fx) == 8
    H = f2.Formula2Point(A, q, p).double_braids((4, 3, 2, 1), [(d["m"], 1) for d in fx])
    for d in fx:
        ref = sum(c * pow(A, a % (p - 1), p) * pow(q, e % (p - 1), p) for a, e, c in d["terms"]) % p
        assert H[(d["m"], 1)] == ref, d["knot"]


def test_method_reconstructs_polynomial():
    from homfly import homfly
    H = homfly(DoubleBraid(1, -1), R=(2, 1), method="double-braid-strong")
    p = 1000003; F = GF(p); A, q = 4242, 777
    assert H.evaluate([F(A), F(q)], one=F.one).v == value(load("4_1")[(2, 1)], A, q, p)


def test_method_scope_and_interpolation_state():
    m = DoubleBraidStrong()
    nodes = I.nodes
    assert m.supports(DoubleBraid(2, 3), (4, 3, 2, 1))
    assert not m.supports(DoubleBraid(2, 3), (4, 4, 2, 1))
    assert not I.DoubleBraidInterpolation().supports(DoubleBraid(2, 3), (4, 3, 1))
    F = GF(1000003)
    m.evaluate(DoubleBraid(2, -1), (3, 1), F, F(5), F(7))
    assert I.nodes is nodes
    with pytest.raises(NotImplementedError):
        f2.double_braid((4, 4, 2, 1), 1, 1, 5, 7, 1000003)
