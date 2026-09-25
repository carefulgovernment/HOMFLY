"""Cross-validation of independent methods and against KnotInfo."""
import random

import pytest

from homfly import homfly
from homfly.algebra.fields import CC, GF, NoSquareRoot
from homfly.checks.structural import differential_expansion_symmetric_ok, special_polynomial_ok
from homfly.conventions import transpose_rep
from homfly.knots.braid import torus_braid
from homfly.knots.families import TorusKnot
from homfly.knots.table import knot, knots
from homfly.methods import Cabling, HeckeFundamental, RossoJones, RTBraid
from homfly.racah.checks import check_braid_relations
from homfly.racah.fundamental import FundamentalRacah

F = GF(1000000007)
A0, q0 = F(1234567), F(7654321)


def test_hecke_vs_knotinfo_all_up_to_9():
    h = HeckeFundamental()
    n = 0
    for rec in knots(max_crossings=9):
        if rec.braid is None:
            continue
        ref = rec.homfly_reference().evaluate([A0, q0], one=F.one)
        assert h.evaluate(rec.braid, (1,), F, A0, q0) == ref, rec.name
        n += 1
    assert n == 84


def test_hecke_vs_knotinfo_sample_12():
    h = HeckeFundamental()
    rng = random.Random(0)
    recs = [r for r in knots(max_crossings=12) if r.crossings >= 10 and r.braid.strands <= 5]
    for rec in rng.sample(recs, 25):
        ref = rec.homfly_reference().evaluate([A0, q0], one=F.one)
        assert h.evaluate(rec.braid, (1,), F, A0, q0) == ref, rec.name


@pytest.mark.parametrize("R", [(1,), (2,), (1, 1), (2, 1)])
@pytest.mark.parametrize("mn", [(2, 3), (2, 5), (3, 2)])
def test_rosso_jones_vs_cabling(R, mn):
    m, n = mn
    a = Cabling().evaluate(torus_braid(m, n), R, F, A0, q0)
    b = RossoJones().evaluate(TorusKnot(m, n), R, F, A0, q0)
    assert a == b


def test_rosso_jones_symmetry():
    rj = RossoJones()
    for R in [(1,), (2,), (2, 1), (3,)]:
        assert rj.evaluate(TorusKnot(3, 5), R, F, A0, q0) == rj.evaluate(TorusKnot(5, 3), R, F, A0, q0)
        assert rj.evaluate(TorusKnot(1, 7), R, F, A0, q0) == 1


def test_sl_n_specialisations():
    c = Cabling()
    b = knot("4_1").braid
    assert c.evaluate(b, (1, 1), F, q0 ** 2, q0) == 1   # [1,1] trivial for sl_2
    assert c.evaluate(b, (2,), F, q0, q0) == 1          # sl_1


def test_rt_engine_fundamental_numeric():
    rt, h = RTBraid(FundamentalRacah()), HeckeFundamental()
    A, q = CC(0.7 + 0.3j), CC(0.72 + 0.54j)
    for name in ["3_1", "4_1", "6_2", "9_42", "10_132"]:
        b = knot(name).braid
        x, y = rt.evaluate(b, (1,), CC, A, q), h.evaluate(b, (1,), CC, A, q)
        assert abs(x - y) < 1e-9 * abs(y)


def test_rt_engine_fundamental_modp():
    from homfly.methods.rt_braid import PathSpace
    rt, h = RTBraid(FundamentalRacah()), HeckeFundamental()
    rng = random.Random(4)
    b = knot("6_2").braid
    done = 0
    while done < 2:
        A, q = F.random_element(rng), F.random_element(rng)
        try:
            v = rt.evaluate(b, (1,), F, A, q)
            S = PathSpace(FundamentalRacah(), (1,), 4, (2, 1, 1), F, A ** -1, q ** -1)
            braid_ok = check_braid_relations(S, 4)
        except NoSquareRoot:
            continue
        assert v == h.evaluate(b, (1,), F, A, q)
        assert braid_ok
        done += 1


def test_reconstruct_colored():
    H2 = homfly("4_1", R=(2,))
    H11 = homfly("4_1", R=(1, 1))
    H1 = homfly("4_1")
    assert H1 == knot("4_1").homfly_reference()
    assert H11 == transpose_rep(H2)
    assert special_polynomial_ok(H2, H1, 2)
    assert differential_expansion_symmetric_ok(H2, 2)
    T = homfly(TorusKnot(2, 3), R=(2,))
    assert T.terms[(8, 6)] == 1 and len(T.terms) == 9
