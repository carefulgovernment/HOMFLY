"""cabling in multiplicity spaces (methods.cabling_paths) against the Racah methods"""
from homfly.compute import homfly
from homfly.knots.table import load_table


def test_matches_racah_methods():
    t = load_table()
    for name, R, other in [("8_16", (3,), "racah-3strand"), ("8_15", (2, 1), "montesinos"),
                           ("8_18", (2, 2), "racah-3strand")]:
        k = t[name]
        H = homfly(k.braid, R, method="cabling-paths")
        ref = homfly(name, R, method=other)
        assert H.terms == ref.terms, (name, R)


def test_polyhedral_knot_q1_and_transposition():
    from homfly.checks.structural import specialise_q1
    from homfly.conventions import transpose_rep
    b = load_table()["9_34"].braid
    H1 = homfly(b, (1,), method="cabling-paths")
    H2 = homfly(b, (2,), method="cabling-paths")
    H11 = homfly(b, (1, 1), method="cabling-paths")
    assert specialise_q1(H2) == specialise_q1(H1) ** 2
    assert transpose_rep(H2).terms == H11.terms
