"""High-level entry point.

    >>> from homfly import homfly
    >>> homfly("3_1")                    # fundamental, from the knot table
    >>> homfly("4_1", R=(2,))            # colored via cabling (or Racah tables)
    >>> homfly(TorusKnot(3, 4), R=(2, 1))  # Rosso--Jones

Every computation runs a method as a black box over GF(p) and reconstructs the
Laurent polynomial in (A, q) by interpolation + CRT (reconstruction.pipeline).
"""
from __future__ import annotations

from functools import lru_cache

from .knots.braid import Braid
from .knots.families import TorusKnot, TwoBridge, DoubleBraid, MontesinosKnot
from .knots.table import load_table
from .methods import (Cabling, HeckeFundamental, NotApplicable, RossoJones)
from .reconstruction.pipeline import reconstruct_laurent, ReconstructionReport
from .reps.partitions import P


def resolve_knot(k):
    """Name from the table -> Braid; other descriptions pass through."""
    if isinstance(k, str):
        rec = load_table()[k]
        if rec.braid is None:
            raise ValueError("no braid for %s (unknot?)" % k)
        return rec.braid
    if isinstance(k, (list, tuple)) and all(isinstance(a, int) for a in k):
        return Braid.from_word(k)
    return k


def default_methods(racah_store=None):
    from .methods.racah3 import Racah3Strand, Racah3StrandU
    from .methods.two_bridge import TwoBridgeMethod
    from .methods.interpolation import DoubleBraidInterpolation
    from .methods.montesinos import MontesinosMethod
    from .methods.cabling_paths import CablingPaths
    from .methods.algebraic import AlgebraicMethod
    ms = [RossoJones(), HeckeFundamental(), TwoBridgeMethod(), MontesinosMethod(), AlgebraicMethod(),
          Racah3Strand(), Racah3StrandU(), DoubleBraidInterpolation(), CablingPaths()]
    if racah_store is not None:
        from .methods.rt_braid import RTBraid
        from .methods.arborescent import Arborescent
        ms += [RTBraid(racah_store), Arborescent(racah_store)]
    ms.append(Cabling(max_strands=9))   # reference only: m*|R| > 9 is impractical
    return ms


# torus knots of the table (checked against KnotInfo in tests/test_methods.py)
TABLE_TORUS = {"3_1": (2, 3), "5_1": (2, 5), "7_1": (2, 7), "8_19": (3, 4), "9_1": (2, 9),
               "10_124": (3, 5), "11a_367": (2, 11)}


def presentations(name):
    """All descriptions of a table knot usable by some method, best first:
    3-strand braid, two-bridge / Montesinos (chirality fixed against
    KnotInfo), braid."""
    from .methods.two_bridge import chirality
    from .methods import montesinos
    rec = load_table()[name]
    out = []
    if name in TABLE_TORUS:
        out.append(TorusKnot(*TABLE_TORUS[name]))
    if rec.braid is not None and rec.braid.strands <= 3:
        out.append(rec.braid)
    if rec.two_bridge is not None:
        p, q = rec.two_bridge
        tb = TwoBridge(p, q, mirror=chirality(p, q, rec.homfly_reference()))
        out.append(tb)
        cf = tb.even_cf()
        if len(cf) == 2:              # antiparallel double braid: any R by interpolation
            out.append(DoubleBraid(-cf[0] // 2, -cf[1] // 2))
    if rec.is_montesinos and rec.montesinos.count(";") >= 2:
        mk = MontesinosKnot.from_notation(rec.montesinos)
        c = montesinos.chirality(mk.fractions(), rec.homfly_reference(), rec.braid)
        if c is not None:             # undecided by H_[1] and H_[2]
            out.append(MontesinosKnot(mk.tangles, mirror=(c < 0)))
    if rec.two_bridge is None and not any(isinstance(K, MontesinosKnot) for K in out):
        ak = algebraic_presentation(name)
        if ak is not None:
            out.append(ak)
    if rec.braid is not None and rec.braid.strands > 3:
        out.append(rec.braid)
    return out


@lru_cache(maxsize=None)
def algebraic_presentation(name):
    """N(algebraic tangle) from the Conway notation (Montesinos knots whose
    chirality the Montesinos tie-break left open: from their fractions), with
    the chirality fixed against KnotInfo; None if not algebraic or undecided."""
    from fractions import Fraction
    from .knots import conway
    from .methods import algebraic
    rec = load_table()[name]
    if rec.is_montesinos and rec.montesinos.count(";") >= 2:
        fr = MontesinosKnot.from_notation(rec.montesinos).fractions()
        tree, notation = ("H", [conway.leaf(Fraction(f)) for f in fr]), rec.montesinos
    elif rec.conway and conway.is_algebraic(rec.conway):
        try:
            tree, notation = conway.parse(rec.conway), rec.conway
        except ValueError:
            return None
    else:
        return None
    try:
        c = algebraic.chirality(tree, rec.homfly_reference(), rec.braid,
                                amphichiral="amphicheiral" in rec.symmetry)
    except ValueError:              # notation does not match the knot
        return None
    if c is None:
        return None
    return algebraic.AlgebraicKnot(tree, mirror=(c < 0), notation=notation)


def choose_method(knot, R, methods=None):
    """(method, knot description) for a knot description or a table name."""
    cands = presentations(knot) if isinstance(knot, str) else [resolve_knot(knot)]
    for m in methods or default_methods():
        for K in cands:
            if m.supports(K, R):
                return m, K
    raise NotApplicable("no available method for %s in %s" % (knot, R))


def homfly(knot, R=(1,), method=None, racah_store=None, seed=0, report=None, exclude=(), **kw):
    """Reduced colored HOMFLY H_R(K; A, q) as a Laurent polynomial.
    ``exclude``: names of methods never to use (e.g. slow fallbacks)."""
    R = P(R)
    if (knot in ("0_1",) or (isinstance(knot, TorusKnot) and knot.m == 1)
            or (isinstance(knot, DoubleBraid) and (knot.m == 0 or knot.n == 0))):
        from .algebra.laurent import Laurent
        return Laurent.const(1)
    methods = default_methods(racah_store)
    if isinstance(method, str):
        methods = [x for x in methods if x.name == method]
    elif method is not None:
        methods = [method]
    methods = [x for x in methods if x.name not in exclude]
    if isinstance(knot, TorusKnot) and not any(x.supports(knot, R) for x in methods):
        knot = knot.braid()
    m, K = choose_method(knot, R, methods)
    rep = report if report is not None else ReconstructionReport()
    rep.method = m.name
    if getattr(m, "direct", False):
        return m.polynomial(K, R, report=rep)
    if hasattr(m, "prime_bound_for"):
        kw.setdefault("prime_bound", m.prime_bound_for(R))
    elif hasattr(m, "prime_bound"):
        kw.setdefault("prime_bound", m.prime_bound)
    from .reps.partitions import conjugate
    if R == conjugate(R):             # H_R(A, q) = H_R(A, -1/q): half the q-points
        kw.setdefault("qsym", True)
    return reconstruct_laurent(m.black_box(K, R), steps=m.steps, seed=seed, report=rep, **kw)


__all__ = ["homfly", "TorusKnot", "TwoBridge", "DoubleBraid", "MontesinosKnot", "Braid"]
