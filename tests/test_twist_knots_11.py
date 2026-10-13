"""Stored 11-box invariants (data/twist_knots_11): R = [4,4,2,1], [4,3,3,1], [4,3,2,2]; twist knots 3_1, 4_1, 5_2, 6_1 and
DoubleBraid(+-2, 2).  Checks that are independent of how the polynomials were fitted."""
import glob
import json
import os

import pytest

from _homfly_txt import DATA

D11 = os.path.join(DATA, "twist_knots_11")


def terms(knot, R):
    d = json.load(open(os.path.join(D11, "%s_%s.json" % (knot, "".join(map(str, R))))))
    assert d["R"] == list(R)
    return {(a, e): c for a, e, c in d["terms"]}


KNOTS = ["3_1", "4_1", "5_2", "6_1", "DB_2_2", "DB_-2_2"]


def test_all_files_present():
    assert len(glob.glob(os.path.join(D11, "*.json"))) == 18


@pytest.mark.parametrize("knot", KNOTS)
def test_transposition(knot):
    """H_{R^T}(A, q) = H_R(A, 1/q): [4,4,2,1] <-> [4,3,2,2], and [4,3,3,1] is self-transposed"""
    H = terms(knot, (4, 4, 2, 1))
    assert terms(knot, (4, 3, 2, 2)) == {(a, -e): c for (a, e), c in H.items()}
    H3 = terms(knot, (4, 3, 3, 1))
    assert H3 == {(a, -e): c for (a, e), c in H3.items()}


@pytest.mark.parametrize("R", [(4, 4, 2, 1), (4, 3, 3, 1), (4, 3, 2, 2)])
def test_figure_eight_amphichiral(R):
    """4_1 is amphichiral: H(A, q) = H(1/A, 1/q)"""
    H = terms("4_1", R)
    assert H == {(-a, -e): c for (a, e), c in H.items()}


@pytest.mark.parametrize("knot,lo,hi", [("3_1", 22, 44), ("4_1", -22, 22), ("5_2", 22, 66), ("6_1", -44, 22)])
def test_A_degree_range(knot, lo, hi):
    """same A-range pattern as the stored |R| <= 10 twist knots (2|R| per twist)"""
    for R in [(4, 4, 2, 1), (4, 3, 3, 1), (4, 3, 2, 2)]:
        a = [k[0] for k in terms(knot, R)]
        assert (min(a), max(a)) == (lo, hi)
