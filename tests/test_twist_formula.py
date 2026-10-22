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


def test_gap_two_pair_columns():
    # pair columns at gap two from the adjacent residues (B) and the chain relation (N)
    ok, lines = T.check_gap2(nmax=6, seed=16)
    assert ok, lines


def test_root_formula_all_gaps():
    """Pair columns of one-cell pair rows at every gap (gaps 1-3 at six boxes) from the adjacent residues and the
    root formula, B and N against the engine."""
    ok, lines = T.check_roots(nmax=6, seed=17)
    assert ok, lines


def test_full_one_cell_pair_rows():
    """Whole one-cell pair rows of N and B (every single column, resonant ones included) from the closed formulas."""
    ok, lines = T.check_fullrow(nmax=6, seed=18)
    assert ok, lines


def test_sign_flip_kernel_pair_rows():
    """Single columns of every one-cell and non-resonant two-cell pair row of N from the sign-flip kernel."""
    ok, lines = T.check_kernel(nmax=6, seed=19)
    assert ok, lines


def test_top_weight_law():
    """w_c(R_min) = m_c(R_min) d_c rho(R_min) with one closed rho for every channel with a single minimal diagram (tops
    and resonant pairs)."""
    ok, lines = T.check_topweight(nmax=6, seed=20)
    assert ok, lines


def test_top_weight_multiplicities():
    """The multiplicity m_c(U) of a pair channel at its minimal diagram is 1 up to nine boxes and 2 for the two-cell
    separate/separate pairs, the first being the three 2+2 splittings of the corners of [4,3,2,1]."""
    ok, lines = T.check_mult(nmax=6, seed=24)
    assert ok, lines
    tops = [c for U, c in T.minimal_pairs(10) if U == (4, 3, 2, 1) and T.I.presence(c, U) == 2]
    assert sorted(tops) == sorted(T.I.pair(Z, Zp) for Z, Zp in (((3, 2, 2, 1), (4, 3, 1)), ((3, 3, 1, 1), (4, 2, 2)),
                                                                   ((3, 3, 2), (4, 2, 1, 1))))
    assert not any(T.I.presence(c, U) > 1 for n in range(2, 10) for U, c in T.minimal_pairs(n))


def test_closed_end_to_end():
    """H_R(Tw_m) from the closed rows of N alone (no vanishing conditions) for every |R| <= 6 whose channels are singles
    and one-cell pairs, against the data and Rosso-Jones."""
    ok, lines = T.check_closed(nmax=6, seed=21)
    assert ok, lines


def test_vacuum_law():
    """E~_{c,vac} = pi(R_min(c)) for every channel with one minimal diagram, and rho = (-1)^|R| pi prod {q^h}^2."""
    ok, lines = T.check_vacuum(nmax=6, seed=22)
    assert ok, lines


def test_twocell_rows():
    """Pair diagonals, first pair columns, and the partner / X* / zero entries of the non-resonant two-cell rows."""
    ok, lines = T.check_twocell(nmax=6, seed=23)
    assert ok, lines


def test_lagrange_light_columns():
    """The light columns of the non-resonant rows of N are Lagrange interpolants on the channels of one domino
    orientation (node count = dim Pi_w for w <= 9), and the explicit weight-3 columns."""
    ok, lines = T.check_lagrange(nmax=6, seed=25)
    assert ok, lines


def test_closed_light_columns():
    """The closed light columns from windowed elementary / complete functions of the composite particles: the hook
    pairs and the singles 1^w, (w) on all rows of the Lagrange families, the two-column pairs and the weight-4 two-cell
    pairs on their own family, and G = 0 / 12 Pi (|R| <= 7)."""
    ok, lines = T.check_hookcols(nmax=7, seed=29)
    assert ok, lines


def test_monomial_law():
    """Phi_c(1) = M(R_min(c)) = (-1)^|R| A^{2|R|} q^{2 kappa_R} for every channel with one minimal diagram."""
    ok, lines = T.check_monomial(nmax=6, seed=26)
    assert ok, lines


def test_ideal_identity():
    """f(U) = rho(U) phi_f(-1) and phi_f(1) = M(U) phi_f(-1) on the whole vanishing ideal I_U, |U| <= 8 (the ghosts of
    [4,3,1] and [3,2,2,1] included)."""
    ok, lines = T.check_ideal(nmax=8, seed=28)
    assert ok, lines


def test_double_braids_and_borromean():
    """The cubic form of the formula against the stored double-braid knots, their A-windows, the mirror identity and
    L6a4 (|R| <= 4)."""
    ok, lines = T.check_dbraid(nmax=4, seed=27)
    assert ok, lines
