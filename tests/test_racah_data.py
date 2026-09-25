"""Methods driven by the imported family-P Racah data (data/racah/portable).

Each check compares against an *independent* computation:
3-strand inclusive data  vs  cabling, Rosso--Jones;
two-bridge exclusive data vs  Hecke/KnotInfo (fundamental), 3-strand data.
"""
import pytest

from homfly.algebra.fields import GF
from homfly.knots.families import TorusKnot, TwoBridge
from homfly.knots.table import knot, knots
from homfly.methods import Cabling, HeckeFundamental, RossoJones
from homfly.methods.racah3 import Racah3Strand
from homfly.methods import two_bridge as tb
from homfly.racah import portable

F = GF(1000000007)
A0, q0 = F(1234567), F(7654321)
INC = portable.available()
EXC = tb.available()

pytestmark = pytest.mark.skipif(not INC or not EXC, reason="family-P data not imported")


@pytest.mark.parametrize("R", [(2,), (1, 1)])
@pytest.mark.parametrize("name", ["3_1", "4_1", "5_2", "6_2"])
def test_racah3_vs_cabling(R, name):
    b = knot(name).braid
    assert Racah3Strand().evaluate(b, R, F, A0, q0) == Cabling().evaluate(b, R, F, A0, q0)


def test_racah3_vs_rosso_jones_all_reps():
    rj, r3 = RossoJones(), Racah3Strand()
    for R in INC:
        for n in (4, 5):
            K = knot("8_19" if n == 4 else "10_124").braid   # T[3,4], T[3,5]
            assert r3.evaluate(K, R, F, A0, q0) == rj.evaluate(TorusKnot(3, n), R, F, A0, q0), R


def test_two_bridge_fundamental_all():
    h = HeckeFundamental()
    n = 0
    for rec in knots(12, two_bridge=True):
        p, q = rec.two_bridge
        K = TwoBridge(p, q, mirror=tb.chirality(p, q, rec.homfly_reference()))
        ref = rec.homfly_reference().evaluate([A0, q0], one=F.one)
        assert tb.TwoBridgeMethod().evaluate(K, (1,), F, A0, q0) == ref, rec.name
        if rec.crossings <= 7:
            assert h.evaluate(rec.braid, (1,), F, A0, q0) == ref
        n += 1
    assert n == 362


def test_two_bridge_vs_racah3():
    reps = [R for R in [(2,), (1, 1), (2, 1), (3,), (3, 1), (2, 2), (4,), (1, 1, 1, 1)]
            if R in INC and R in EXC]
    n = 0
    for rec in knots(12, two_bridge=True):
        if rec.braid.strands > 3 or rec.crossings > 10:
            continue
        p, q = rec.two_bridge
        K = TwoBridge(p, q, mirror=tb.chirality(p, q, rec.homfly_reference()))
        for R in reps:
            assert (tb.TwoBridgeMethod().evaluate(K, R, F, A0, q0)
                    == Racah3Strand().evaluate(rec.braid, R, F, A0, q0)), (rec.name, R)
        n += 1
    assert n >= 15


def test_two_bridge_six_boxes_transposition():
    if (6,) not in EXC or (1,) * 6 not in EXC:
        pytest.skip("no 6-box data")
    rec = knot("5_2")
    p, q = rec.two_bridge
    K = TwoBridge(p, q, mirror=tb.chirality(p, q, rec.homfly_reference()))
    m = tb.TwoBridgeMethod()
    # H_{R^T}(A, q) = H_R(A, -1/q);  sl_1: H_[6](A = q) = 1
    assert m.evaluate(K, (1,) * 6, F, A0, q0) == m.evaluate(K, (6,), F, A0, -(q0 ** -1))
    assert m.evaluate(K, (6,), F, q0, q0) == 1


def test_family_h_six_boxes():
    """Family H (other 6-box reps): [6] equals family P; transposition pairs."""
    SB = tb.available_sbar()
    if (6,) not in SB:
        pytest.skip("no family-H data")
    m = tb.TwoBridgeMethod()
    for name in ["3_1", "4_1", "6_1", "7_4"]:
        rec = knot(name)
        p, q = rec.two_bridge
        K = TwoBridge(p, q, mirror=tb.chirality(p, q, rec.homfly_reference()))
        cf = K.even_cf()
        assert tb.value((6,), cf, F, A0 ** -1, q0 ** -1) == tb.value_sbar((6,), cf, F, A0, q0)
        for R, RT in [((3, 3), (2, 2, 2)), ((5, 1), (2, 1, 1, 1, 1))]:
            assert m.evaluate(K, RT, F, A0, q0) == m.evaluate(K, R, F, A0, -(q0 ** -1)), (name, R)
