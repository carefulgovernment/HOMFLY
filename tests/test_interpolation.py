"""Interpolation formula for antiparallel double braids (methods.interpolation)
against the paper's closed forms, the Racah two-bridge data and Rosso--Jones."""
import pytest

from homfly.algebra.fields import GF
from homfly.algebra.quantum import qnum
from homfly.knots.families import DoubleBraid, TorusKnot
from homfly.methods import RossoJones
from homfly.methods import interpolation as I
from homfly.methods import two_bridge as tb
from homfly.reps.partitions import boxes, hook

F = GF(1000000007)
A0, q0 = F(1234567), F(7654321)
D = lambda k: A0 * q0 ** k - 1 / (A0 * q0 ** k)       # noqa: E731
br = lambda n: qnum(n, q0)                             # noqa: E731
z = lambda i, j: D(i) * D(-j)                          # noqa: E731
S = I.single


def test_E_rows_appendix():
    pt = I.Point(F, A0, q0)
    E = lambda c, x: pt.E_row(S(c))[S(x)]              # noqa: E731
    assert E((2,), (1,)) == -br(2) * D(1) / D(0)
    assert E((3,), (1,)) == br(3) * D(4) / D(0)
    assert E((3,), (2,)) == -br(3) * D(3) / D(1)
    assert E((2, 1), (1,)) == br(3)
    assert E((2, 2), (1,)) == -br(2) ** 2 * D(-1) * D(1) / (D(-2) * D(2))
    assert E((3, 1), (2, 1)) == -br(2) * br(4) / br(3) * D(1) * D(3) / (D(0) * D(2))
    assert E((3, 2), (3, 1)) == -br(3) * D(-1) * D(2) / (D(-2) * D(3))


def test_level_one_and_rectangles():
    pt = I.Point(F, A0, q0)
    for R in [(2, 1), (3, 1, 1), (4, 2)]:
        t = sum((z(hook(R, b) + b[1] - b[0], hook(R, b) - b[1] + b[0]) for b in boxes(R)), F.zero)
        assert pt.F_c(S((1,)), R) == t
    for r, s in [(2, 2), (3, 2)]:
        for lam in [(1,), (2,), (1, 1), (2, 1), (2, 2)]:
            v = F.one
            for b in boxes(lam):
                c, h = b[1] - b[0], hook(lam, b)
                v = v * br(r - c) * br(s + c) * D(r + c) * D(c - s) / br(h) ** 2
            assert pt.F_c(S(lam), (r,) * s) == v


def test_closed_formula_21():
    v = I.paper_value((2, 1), 1, -1, F, A0, q0)
    ref = (1 + (z(3, 3) + z(2, 0) + z(0, 2)) + br(3) / br(2) * z(2, 2) * (z(0, 3) + z(3, 0))
           - (q0 - 1 / q0) ** 4 * br(3) ** 2 * z(2, 2) + z(1, 1) * z(2, 2) * z(3, 3))
    assert v == ref
    assert I.paper_value((1,), 1, 1, F, A0, q0) == A0 ** 2 * (q0 ** 2 + q0 ** -2) - A0 ** 4


MN = [(1, -1), (1, 1), (2, 1), (2, -1), (-2, -3), (3, -1)]


@pytest.mark.skipif(not tb.available(), reason="Racah data not imported")
@pytest.mark.parametrize("R", [(2,), (2, 1), (3, 1), (2, 2), (3, 2), (3, 1, 1), (4, 1)])
def test_vs_racah_two_bridge(R):
    m = tb.TwoBridgeMethod()
    for mn in MN:
        K = DoubleBraid(*mn)
        assert I.DoubleBraidInterpolation().evaluate(K, R, F, A0, q0) == m.evaluate(K, R, F, A0, q0), mn


@pytest.mark.skipif(not tb.available_sbar(), reason="6-box Racah data not imported")
@pytest.mark.parametrize("R", [(4, 2), (5, 1), (3, 3)])
def test_vs_racah_six_boxes(R):
    m = tb.TwoBridgeMethod()
    for mn in [(1, -1), (2, 1)]:
        K = DoubleBraid(*mn)
        assert I.DoubleBraidInterpolation().evaluate(K, R, F, A0, q0) == m.evaluate(K, R, F, A0, q0), mn


def test_seven_boxes_trefoil_vs_rosso_jones():
    for R in [(5, 2), (4, 3)]:
        assert (I.paper_value(R, 1, 1, F, A0, q0)
                == RossoJones().evaluate(TorusKnot(2, 3), R, F, A0, q0))


def test_interpolation_systems_consistent():
    pt = I.Point(F, A0, q0)
    for c in I.channels_up_to(3):
        assert I.row_violations(pt, c) == 0, c
