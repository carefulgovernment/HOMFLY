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


def test_rows_determined_at_seven_boxes():
    """The conditions at |nu| <= 8 leave six rows at [4,2,1] (and six at [3,2,1,1]) with one free unknown; with the
    nine-box conditions added every row is determined and satisfies every condition up to ten boxes."""
    pt = I.Point(F, A0, q0)
    for R in [(4, 2, 1), (3, 2, 1, 1)]:
        for c in I.channels_of(R):
            E = pt.E_row(c)
            r, n = pt._E[("rank", c)]
            assert r == n, (R, c, r, n)
            for nu in I.nodes(c, 10):
                assert sum((e * pt.psi(x, nu) for x, e in E.items()), F.zero) == 0, (R, c, nu)


def test_free_unknown_changes_twist_knots_from_m3(monkeypatch):
    """With the old rows (no extra conditions) F_c is unchanged and so are |m| <= 2, but DB(3, 1) = 7_2 and
    DB(2, -3) at [4,2,1] change."""
    new = I.Point(F, A0, q0)
    vals = {mn: I.paper_value((4, 2, 1), *mn, F, A0, q0, point=new) for mn in [(1, -1), (2, 1), (-2, 1), (3, 1), (2, -3)]}
    monkeypatch.setattr(I, "NODE_EXTRA", 0)
    old = I.Point(F, A0, q0)
    same = {mn: I.paper_value((4, 2, 1), *mn, F, A0, q0, point=old) == v for mn, v in vals.items()}
    assert same == {(1, -1): True, (2, 1): True, (-2, 1): True, (3, 1): False, (2, -3): False}
    c = I.single((4, 2, 1))
    assert old._E[("rank", c)] == (30, 31) and new._E[("rank", c)] == (31, 31)
    assert old.F_c(c, (4, 2, 1)) == new.F_c(c, (4, 2, 1))


def test_no_claim_at_ghost_diagrams():
    """At R containing [4,3,1] or [3,2,2,1] the rows of the tops of these diagrams are not determined (rank 41 of 42)
    and the formula misses the ghost term, so the method does not claim these R."""
    m = I.DoubleBraidInterpolation()
    K = DoubleBraid(1, -1)
    assert m.supports(K, (4, 2, 1)) and m.supports(K, (5, 2, 1)) and m.supports(K, (4, 2, 2))
    for R in [(4, 3, 1), (3, 2, 2, 1), (5, 3, 1), (4, 3, 2, 1), (4, 4, 4)]:
        assert not m.supports(K, R), R
    pt = I.Point(F, A0, q0)
    c = I.single((4, 3, 1))
    pt.E_row(c)
    assert pt._E[("rank", c)] == (41, 42)
