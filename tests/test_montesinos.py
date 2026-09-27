"""Montesinos tangle calculus (methods.montesinos) against KnotInfo (fundamental)
and the 3-strand Racah methods (colored), for the families P, H and G."""
import random
from fractions import Fraction

import pytest

from homfly.algebra.fields import GF, primes_below
from homfly.compute import choose_method, presentations
from homfly.knots.families import MontesinosKnot
from homfly.knots.table import load_table
from homfly.methods import montesinos as MO
from homfly.methods.racah3 import Racah3Strand, Racah3StrandU

F = GF(primes_below(2 ** 21, 1)[0])
rng = random.Random(11)
A0, q0 = F.random_element(rng), F.random_element(rng)


def _table_montesinos(braid_index=None, limit=None):
    out = []
    for k in load_table().values():
        if not k.is_montesinos or k.montesinos.count(";") < 2:
            continue
        if braid_index is not None and k.braid_index != braid_index:
            continue
        out.append(k)
    return out[:limit]


def _oriented(k):
    fr = MontesinosKnot.from_notation(k.montesinos).fractions()
    c = MO.chirality(fr, k.homfly_reference())
    return None if c is None else [c * x for x in fr]


def test_fundamental_against_knotinfo():
    n = 0
    for k in _table_montesinos(limit=60):
        fr = MontesinosKnot.from_notation(k.montesinos).fractions()
        ref = k.homfly_reference().evaluate([A0, q0], one=F.one)
        assert ref in [MO.value([s * x for x in fr], (1,), F, A0, q0) for s in (1, -1)], k.name
        n += 1
    assert n == 60


def test_order_of_tangles_irrelevant():
    fr = [Fraction(2, 3), Fraction(2, 3), Fraction(1, 2)]
    for R in [(2,), (2, 1), (4, 2)]:
        vals = {MO.value(fr[i:] + fr[:i], R, F, A0, q0) for i in range(3)}
        assert len(vals) == 1


def test_transposition():
    fr = [Fraction(3), Fraction(-5, 2), Fraction(7, 3)]
    for R, Rt in [((3,), (1, 1, 1)), ((5, 1), (2, 1, 1, 1, 1))]:
        assert MO.value(fr, Rt, F, A0, q0) == MO.value(fr, R, F, A0, -1 / q0)


@pytest.mark.parametrize("R", [(2,), (1, 1), (2, 2), (2, 1)])
def test_colored_against_racah3(R):
    m = Racah3Strand()
    for k in _table_montesinos(braid_index=3, limit=4):
        assert MO.value(_oriented(k), R, F, A0, q0) == m.evaluate(k.braid, R, F, A0, q0), k.name


@pytest.mark.parametrize("R", [(2, 1, 1), (4, 1), (3, 2), (3, 1, 1), (2, 2, 1), (2, 1, 1, 1)])
def test_generated_G_against_racah3(R):
    """mixed S reconstructed by scripts/make_mixed_S_gtpath.sh (antiparallel vertex)"""
    m = Racah3Strand()
    for k in _table_montesinos(braid_index=3, limit=3):
        assert MO.value(_oriented(k), R, F, A0, q0) == m.evaluate(k.braid, R, F, A0, q0), k.name


def test_generated_G_transposition_parallel_vertex():
    """R and R^T were reconstructed independently; knots with a parallel vertex"""
    n = 0
    for k in _table_montesinos():
        fr = MontesinosKnot.from_notation(k.montesinos).fractions()
        if MO._types(MO._orientations(tuple(fr))[1][0][2][-1])[0] != "par":
            continue
        for R, Rt in [((4, 1), (2, 1, 1, 1)), ((3, 2), (2, 2, 1))]:
            assert MO.value(fr, R, F, A0, q0) == MO.value(fr, Rt, F, A0, -1 / q0), k.name
        n += 1
        if n == 4:
            break
    assert n == 4


@pytest.mark.parametrize("R", [(5, 1), (4, 2), (2, 2, 1, 1)])
def test_six_boxes_against_racah3U(R):
    m = Racah3StrandU()
    for k in _table_montesinos(braid_index=3, limit=2):
        assert MO.value(_oriented(k), R, F, A0, q0) == m.evaluate(k.braid, R, F, A0, q0), k.name


def test_all_29_reps_available():
    from homfly.reps.partitions import partitions
    assert all(MO.MontesinosMethod().supports(MontesinosKnot(((2, 3), (2, 3), (1, 2))), R)
               for k in range(1, 7) for R in partitions(k))


def test_8_15_presentation():
    assert any(isinstance(K, MontesinosKnot) for K in presentations("8_15"))
    m, K = choose_method("8_15", (4, 2))
    assert m.name == "montesinos" and isinstance(K, MontesinosKnot)
