"""Twist knots in non-rectangular R (scripts/twist_formula_check.py, docs/notes/twist_knots_formula.tex):
the channel/evolution formula against the stored data, the KNTZ closed form of the single rows,
the phi_lambda(-1) product formula, the tabulated exact weights, the pair-row identities, the reverse-tableau
formula for the single sector of the binomial matrix N = E~^{-1}, the one-cell pair facts, the union formula and the
hook family."""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "twist_formula_check", os.path.join(os.path.dirname(__file__), "..", "scripts", "twist_formula_check.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def test_formula_vs_data_small():
    ok, lines = T.check_data(nmax=4, seed=1)
    assert ok, lines


def test_kntz_single_rows():
    ok, lines = T.check_kntz(nmax=5, seed=2)
    assert ok, lines


def test_phi_minus_one_product():
    ok, lines = T.check_phi(nmax=6, seed=3)
    assert ok, lines


def test_tabulated_weights_vs_data():
    ok, lines = T.check_table(nmax=4, seed=4)
    assert ok, lines


def test_tabulated_twist_matrix():
    ok, lines = T.check_bmat(nmax=6, seed=8)
    assert ok, lines


def test_pair_row_identities():
    ok, lines = T.check_pairs(nmax=5, seed=5)
    assert ok, lines


def test_figure_eight_seven_boxes():
    F, A, q = T.random_point(seed=6)
    pt = T.I.Point(F, A, q)
    for R in [(4, 2, 1), (3, 2, 1, 1), (5, 2)]:
        tot = F.one
        for c in T.I.channels_of(R):
            tot = tot + pt.F_c(c, R)
        assert tot == T.data_value("4_1", R, F, A, q), R


@pytest.mark.parametrize("m", [1, -1, 2, -2])
def test_symmetric_seven(m):
    ok, lines = T.check_sym(rs=(7,), seed=7, knots=[m])
    assert ok, lines


def test_binomial_matrix_single_sector():
    # the reverse-tableau formula for N = E~^{-1} on single rows; no pair entries; N_{c,vac} = 1; the [1] column
    ok, lines = T.check_binom(nmax=5, seed=9)
    assert ok, lines


def test_binomial_matrix_pair_facts():
    # one-cell pairs: diagonal, first pair column, columns [1,1] and [2]
    ok, lines = T.check_pairn(nmax=5, seed=10)
    assert ok, lines


def test_union_formula_pair_rows():
    # B_c from the KNTZ rows of U, Z, Z' on all non-resonant columns
    ok, lines = T.check_union(nmax=5, seed=11)
    assert ok, lines


def test_hook_family_rows():
    # closed rows of B for {[k-1,1],[k]} and the transposed family
    ok, lines = T.check_hook(nmax=5, seed=12)
    assert ok, lines


def test_khook_law_pair_columns():
    # K-hook law for the pair columns of the pair rows of N (diagonal, first column, adjacent non-landing entries)
    ok, lines = T.check_khook(nmax=5, seed=13)
    assert ok, lines


def test_four_row_formula_single_columns():
    # every single column of a one-cell pair row of N from the four closed single rows V, Z, Z', U (merged basis)
    ok, lines = T.check_fourrow(nmax=5, seed=14)
    assert ok, lines


def test_adjacent_residue_formula():
    # every adjacent pair column of a one-cell pair row (landing entries included) on N and B
    ok, lines = T.check_adjres(nmax=5, seed=15)
    assert ok, lines
