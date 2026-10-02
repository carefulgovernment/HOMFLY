"""Arborescent (algebraic) knots from Conway notation: parser semantics against
KnotInfo's Montesinos notation, tangle calculus against the Montesinos engine,
H_[1] against KnotInfo and colored values against cabling."""
import random
from fractions import Fraction as Fr

import pytest

from homfly.algebra.fields import GF
from homfly.knots import conway as C
from homfly.knots.families import MontesinosKnot
from homfly.knots.table import load_table
from homfly.methods import algebraic as AL
from homfly.methods import montesinos as MO

F = GF(1048573)


def test_parse_examples():
    assert C.parse("[22;3;2-]") == ("H", [C.leaf(Fr(2, 5)), C.leaf(Fr(1, 3)), C.leaf(Fr(-1, 2))])
    assert C.parse("[3;3;2+ + ]") == ("H", [C.leaf(Fr(1, 3)), C.leaf(Fr(1, 3)), C.leaf(Fr(5, 2))])
    assert C.parse("[-4 -1;2 2 1;2]")[1][0] == C.leaf(Fr(-4, 5))
    assert C.parse("[(3;2)(21;2)]") == ("H", [("V", [C.leaf(Fr(3)), C.leaf(Fr(2))]),
                                             C.leaf(Fr(2, 3)), C.leaf(Fr(1, 2))])
    with pytest.raises(ValueError):
        C.parse("[.2.20]")


def _sum_and_fracs(fr):
    return sum(fr), sorted(f % 1 for f in fr if f % 1)


def test_conway_lists_match_montesinos_notation():
    """Flat Conway lists = the KnotInfo Montesinos fractions (up to mirror,
    order and integer shifts between tangles)."""
    bad = []
    n = 0
    for k in load_table().values():
        if (not k.conway or not k.is_montesinos or not C.is_algebraic(k.conway)
                or "(" in k.conway or k.name == "12n_221"):
            continue
        t = C.parse(k.conway)
        if t[0] != "H" or any(c[0] != "leaf" for c in t[1]):
            continue
        n += 1
        mine = [c[1] for c in t[1]]
        ref = list(MontesinosKnot.from_notation(k.montesinos).fractions())
        if _sum_and_fracs(mine) not in (_sum_and_fracs(ref), _sum_and_fracs([-x for x in ref])):
            bad.append(k.name)
    assert n > 600 and not bad


@pytest.mark.parametrize("R", [(1,), (2,), (2, 1), (3, 1), (2, 2), (3, 2, 1)])
def test_vertical_sum_identities(R):
    """V(f, 1/n) = 1/(1/f + n) and V(f, ∞) = f inside a Montesinos sum."""
    rng = random.Random(1)
    fr = [Fr(2, 5), Fr(1, 3)]
    f = Fr(2, 5)
    for X, g in [(("V", [C.leaf(f), C.leaf(Fr(1, 2))]), 1 / (1 / f + 2)),
                 (("V", [C.leaf(f), C.leaf(Fr(1, 3))]), 1 / (1 / f + 3)),
                 (("V", [C.leaf(f), C.leaf(C.INF)]), f),
                 (("V", [C.leaf(f), ("V", [C.leaf(C.INF), C.leaf(Fr(1, 2))])]), 1 / (1 / f + 2))]:
        t = AL._freeze(("H", [C.leaf(fr[0]), C.leaf(fr[1]), X]))
        A, q = F.random_element(rng), F.random_element(rng)
        assert AL.value(t, R, F, A, q) == MO.value(fr + [g], R, F, A, q)


def test_fundamental_against_knotinfo():
    t = load_table()
    for n in ["10_80", "10_153", "11n_21", "12a_123", "12n_119", "12n_87"]:
        tr = C.parse(t[n].conway)
        assert AL.chirality(tr, t[n].homfly_reference()) is not None


def test_colored_against_cabling():
    from homfly.methods.cabling_paths import CablingPaths
    t = load_table()
    rng = random.Random(7)
    cp = CablingPaths()
    for n, R in [("10_80", (2, 1)), ("10_80", (2, 2)), ("12n_119", (2, 1))]:
        tr = C.parse(t[n].conway)
        s = AL.chirality(tr, t[n].homfly_reference())
        key = AL._freeze(tr if s == 1 else C.mirror(tr))
        A, q = F.random_element(rng), F.random_element(rng)
        assert AL.value(key, R, F, A, q) == cp.evaluate(t[n].braid, R, F, A, q)
